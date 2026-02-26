use crate::common::{Adj, Edge, Node, Offsets, SemType, TableInfo, TableType};
use crate::encode_unique;
use crate::shallow_pca;
use crate::text_encoder;
use crate::text_index;
use crate::time_cutoffs;
use anyhow::{Result, anyhow};
use clap::Parser;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::{self, Itertools};
use parquet::file::metadata::ParquetMetaDataReader;
use polars::prelude::*;
use rkyv::rancor::Error;
use serde::Deserialize;
use serde_json::{self, Value, from_str, json};
use serde_yaml;
use half::bf16;
use pyo3::prepare_freethreaded_python;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::env::var;
use std::fs;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::io::BufWriter;
use std::io::{Seek, Write};
use std::iter;
use std::path::{Path, PathBuf};
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128_with_seed};

fn log_semantic_snapshot(
    label: &str,
    table_name: &str,
    table_type: &TableType,
    df: &DataFrame,
    semantic_types: &HashMap<String, String>,
    col_overrides: Option<&HashMap<String, String>>,
    angle_counts: Option<&HashMap<String, usize>>,
) {
    eprintln!(
        "[semantic] {} for {} ({:?})",
        label, table_name, table_type
    );
    let mut names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    if let Some(counts) = angle_counts {
        for base in counts.keys() {
            if !names.contains(base) {
                names.push(base.clone());
            }
        }
    }
    names.sort();
    for name in names {
        if let Some(overrides) = col_overrides {
            if overrides.contains_key(&name) {
                continue;
            }
        }
        let dtype_display = match df.column(&name) {
            Ok(col) => format!("{:?}", col.dtype()),
            Err(_) => "n/a".to_string(),
        };
        let sem = semantic_types
            .get(&name)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        let display = match sem {
            "shallow_c" => angle_counts
                .and_then(|m| m.get(&name))
                .map(|k| format!("shallow_c (k={})", k))
                .unwrap_or_else(|| "shallow_c".to_string()),
            "shallow" => angle_counts
                .and_then(|m| m.get(&name))
                .map(|k| format!("shallow (k={})", k))
                .unwrap_or_else(|| "shallow".to_string()),
            other => other.to_string(),
        };
        eprintln!("    {}: dtype={}, semantic={}", name, dtype_display, display);
    }
}

//Function to automatically binarize a column based on its first non-null value
pub fn make_column_boolean(df: &mut DataFrame, col_name: &str) -> PolarsResult<()> {
    let s_str = df.column(col_name)?.cast(&DataType::String)?;
    let ca = s_str.str()?;
    let first = ca
        .get(0)
        .ok_or_else(|| PolarsError::NoData("column is empty".into()))?;

    let mut mask_series = ca.equal(first).into_series();
    mask_series.rename(col_name.into());
    df.replace(col_name, mask_series)?;
    Ok(())
}

fn datetime_ns_at(series: &Series, idx: usize) -> Option<i64> {
    match series.get(idx).ok()? {
        AnyValue::Datetime(v, TimeUnit::Nanoseconds, _) => Some(v),
        AnyValue::Datetime(v, TimeUnit::Microseconds, _) => Some(v * 1_000),
        AnyValue::Datetime(v, TimeUnit::Milliseconds, _) => Some(v * 1_000_000),
        _ => None,
    }
}

fn datetime_seconds_at(series: &Series, idx: usize) -> Option<i32> {
    datetime_ns_at(series, idx).map(|v| (v / 1_000_000_000) as i32)
}

const PBAR_TEMPLATE: &str = "{percent}% {bar} {decimal_bytes}/{decimal_total_bytes} [{elapsed_precise}<{eta_precise}, {decimal_bytes_per_sec}]";

#[derive(Debug, Clone)]
struct ColStat {
    mean: f64,
    std: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Table {
    pub table_name: String,
    pub df: DataFrame,
    pub col_stats: Vec<ColStat>,
    pub pcol_name: Option<String>,
    pub fcol_name_to_ptable_name: HashMap<String, String>,
    pub tcol_name: Option<String>,
    pub node_idx_offset: i32,
    pub col_name_overrides: HashMap<String, String>,
    pub semantic_types: HashMap<String, String>,
    pub angle_component_counts: HashMap<String, usize>,
}

pub(crate) type TableMap =
    HashMap<(String, TableType), Table, BuildHasherDefault<DefaultHasher>>;

#[derive(Debug, Deserialize)]
struct TaskConfigEntry {
    db_name: String,
    #[serde(default)]
    task_name: Vec<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskConfigFile {
    dbs: Vec<TaskConfigEntry>,
}

#[derive(Debug, Clone)]
struct DbTaskSpec {
    name: String,
    tasks: HashSet<String>,
    dataset_path: String,
}

#[derive(Parser)]
pub struct Cli {
    /// YAML file listing db_name and task_name entries (e.g., configs/pyg/hpo/all-nodes.yaml)
    #[arg(value_name = "TASK_CONFIG", default_value = "../configs/pyg/hpo/all-nodes.yaml")]
    task_config: PathBuf,
    /// Optionally restrict preprocessing to a single database within the YAML file
    #[arg(long)]
    db_name: Option<String>,
    #[arg(long, default_value_t = false)]
    skip_db: bool,
    #[arg(long, default_value_t = false)]
    use_rfm_preprocessing: bool,
    #[arg(long, default_value = "all-MiniLM-L6-v2")]
    text_encoder: String,
    #[arg(long, default_value_t = 8192)]
    text_encoder_batch: usize,
    #[arg(long, default_value_t = 256)]
    text_encoder_max_length: usize,
    #[arg(long, default_value = "auto")]
    text_encoder_device: String,
    #[arg(long, default_value_t = false)]
    text_encoder_normalize: bool,
    #[arg(long)]
    text_embedding_cache_dir: Option<String>,
    #[arg(long, default_value_t = 3)]
    shallow_size: usize,
    #[arg(long)]
    pca_batch_size: Option<usize>,
    #[arg(long, default_value_t = 32768)]
    pca_row_chunk: usize,
    #[arg(long, default_value_t = true)]
    pca_whiten: bool,
    #[arg(long, default_value_t = false)]
    pca_l2_normalize_rows: bool,
    #[arg(long, default_value = "relbench/relbench/datasets/test_timecutoffs.json")]
    time_cutoffs_file: String,
    /// Output directory for preprocessed datasets
    #[arg(long)]
    output_dir: Option<String>,
}

pub(crate) type SemanticTypeMap = HashMap<String, HashMap<String, String>>;

const ANGLE_HASH_HARMONICS: usize = 2;
const ANGLE_HASH_DIM: usize = ANGLE_HASH_HARMONICS * 2;
const MIX_CONST: u64 = 0x9E37_79B1_85EB_CA87;

fn load_task_specs(config_path: &Path, db_filter: Option<&str>) -> Result<Vec<DbTaskSpec>> {
    let raw = fs::read_to_string(config_path)?;
    let parsed: TaskConfigFile = serde_yaml::from_str(&raw)?;
    let home = var("HOME")?;
    let mut specs = Vec::new();

    for entry in parsed.dbs {
        if let Some(filter) = db_filter {
            if entry.db_name != filter {
                continue;
            }
        }

        let dataset_path = entry.path.unwrap_or_else(|| {
            // Check RELBENCH_DIR environment variable first
            if let Ok(relbench_dir) = var("RELBENCH_DIR") {
                format!("{}/{}", relbench_dir, entry.db_name)
            } else {
                // Try /home/ubuntu/data/relbench first, fall back to scratch
                let data_relbench = format!("{}/data/relbench/{}", home, entry.db_name);
                let scratch_relbench = format!("{}/scratch/relbench/{}", home, entry.db_name);
                if std::path::Path::new(&data_relbench).exists() {
                    data_relbench
                } else {
                    scratch_relbench
                }
            }
        });
        let tasks: HashSet<String> = entry.task_name.into_iter().collect();

        specs.push(DbTaskSpec {
            name: entry.db_name,
            tasks,
            dataset_path,
        });
    }

    if specs.is_empty() {
        return Err(anyhow!(
            "No databases found in config {} after applying filters",
            config_path.display()
        ));
    }

    Ok(specs)
}

fn cast_col_to_bool(df: DataFrame, col_name: &str) -> Result<DataFrame, PolarsError> {
    df.lazy()
        .with_column(col(col_name).cast(DataType::Boolean).alias(col_name))
        .collect()
}

fn load_semantic_types(db_name: &str) -> Option<SemanticTypeMap> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        PathBuf::from("stypes"),
        manifest_dir.join("stypes"),
        manifest_dir.join("..").join("stypes"),
        manifest_dir.join("..").join("..").join("stypes"),
    ];

    for base in candidates {
        let path = base.join(db_name).join("stypes.json");
        if !path.exists() {
            continue;
        }
        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(err) => {
                eprintln!(
                    "Failed to read semantic types from {}: {}",
                    path.display(),
                    err
                );
                continue;
            }
        };
        match serde_json::from_str::<SemanticTypeMap>(&contents) {
            Ok(map) => return Some(map),
            Err(err) => {
                eprintln!(
                    "Failed to parse semantic types from {}: {}",
                    path.display(),
                    err
                );
            }
        }
    }

    None
}

pub(crate) fn infer_column_types(df: &DataFrame) -> HashMap<String, String> {
    df.get_columns()
        .iter()
        .map(|col| {
            let ty = match col.dtype() {
                DataType::Boolean => "boolean",
                dt if dt.is_categorical() || matches!(dt, DataType::List(_)) => "categorical",
                DataType::Datetime(_, _) => "timestamp",
                DataType::String => "text_embedded",
                _ => "numerical",
            };
            (col.name().to_string(), ty.to_string())
        })
        .collect()
}

fn mix_seed(col_seed: u64, index: u64) -> u64 {
    col_seed ^ MIX_CONST ^ index.wrapping_mul(MIX_CONST)
}

fn angle_hash(namespace: &str, value: &str, column_seed: u64) -> [f32; ANGLE_HASH_DIM] {
    let mut payload = String::with_capacity(namespace.len() + 2 + value.len());
    payload.push_str(namespace);
    payload.push_str("::");
    payload.push_str(value);
    let bytes = payload.as_bytes();
    let mut out = [0f32; ANGLE_HASH_DIM];
    let two_pi = std::f64::consts::PI * 2.0;

    for h in 0..ANGLE_HASH_HARMONICS {
        let seed = mix_seed(column_seed, h as u64);
        let hash128 = xxh3_128_with_seed(bytes, seed);
        let mantissa = hash128 & ((1u128 << 53) - 1);
        let u = mantissa as f64 / ((1u64 << 53) as f64);
        let phi = u * two_pi;
        let (sine, cosine) = phi.sin_cos();
        out[2 * h] = cosine as f32;
        out[2 * h + 1] = sine as f32;
    }

    out
}

fn apply_angle_hashing_for_categorical(
    df: &mut DataFrame,
    db_name: &str,
    table_name: &str,
    column_types: &HashMap<String, String>,
    column_overrides: &mut HashMap<String, String>,
    angle_component_counts: &mut HashMap<String, usize>,
) -> PolarsResult<()> {
    let mut categorical_columns: Vec<String> = column_types
        .iter()
        .filter_map(|(col, ty)| {
            if ty == "categorical" {
                Some(col.clone())
            } else {
                None
            }
        })
        .collect();

    categorical_columns.sort();

    for col_name in categorical_columns {
        let Ok(series) = df.column(&col_name) else {
            continue;
        };
        let string_series = match series.cast(&DataType::String) {
            Ok(series) => series,
            Err(err) => {
                eprintln!(
                    "Failed to cast categorical column {}.{} to string: {}",
                    table_name, col_name, err
                );
                continue;
            }
        };
        let string_chunked = string_series.str()?;
        let namespace = format!("{}::{}::{}", db_name, table_name, col_name);
        let column_seed = xxh3_64(namespace.as_bytes());
        let mut hashed_columns: Vec<Vec<Option<f32>>> = (0..ANGLE_HASH_DIM)
            .map(|_| Vec::with_capacity(df.height()))
            .collect();

        for value in string_chunked.into_iter() {
            if let Some(v) = value {
                let features = angle_hash(&namespace, v, column_seed);
                for (idx, feature) in features.iter().enumerate() {
                    hashed_columns[idx].push(Some(*feature));
                }
            } else {
                for column in &mut hashed_columns {
                    column.push(None);
                }
            }
        }

        let base_label = format!("{}_angle", col_name);
        let mut new_columns = Vec::with_capacity(ANGLE_HASH_DIM);
        for (idx, values) in hashed_columns.into_iter().enumerate() {
            let new_name = format!("{}_angle_{:02}", col_name, idx + 1);
            new_columns.push(Series::new(new_name.as_str().into(), values));
            column_overrides.insert(new_name, base_label.clone());
        }

        for column in new_columns {
            df.with_column(column)?;
        }
        df.drop_in_place(&col_name)?;
        angle_component_counts.insert(base_label.clone(), ANGLE_HASH_DIM);
    }

    Ok(())
}

fn output_path(pre_path: &str, stem: &str, ext: &str, use_rfm: bool) -> String {
    if use_rfm {
        format!("{}/{}_rfm.{}", pre_path, stem, ext)
    } else {
        format!("{}/{}.{}", pre_path, stem, ext)
    }
}

fn encoder_file_tag(model: &str) -> String {
    model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn write_text_vocab(pre_path: &str, use_rfm: bool, text_vec: &[String]) -> Result<()> {
    let text_path = output_path(pre_path, "text", "json", use_rfm);
    let mut writer = BufWriter::new(fs::File::create(text_path)?);
    serde_json::to_writer(&mut writer, text_vec)?;

    let mut text_map: HashMap<String, i32> = HashMap::with_capacity(text_vec.len());
    for (idx, s) in text_vec.iter().enumerate() {
        text_map.insert(s.clone(), idx as i32);
    }
    let text_map_path = output_path(pre_path, "text_map", "json", use_rfm);
    let mut writer = BufWriter::new(fs::File::create(text_map_path)?);
    serde_json::to_writer(&mut writer, &text_map)?;
    Ok(())
}

fn write_text_embeddings(
    pre_path: &str,
    shared_dir: Option<&str>,
    encoder_tag: &str,
    embeddings: &ndarray::Array2<f32>,
) -> Result<()> {
    fn write_bf16_file(path: &Path, embeddings: &ndarray::Array2<f32>) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(fs::File::create(path)?);
        for value in embeddings.iter() {
            let bf = bf16::from_f32(*value);
            writer.write_all(&bf.to_bits().to_le_bytes())?;
        }
        Ok(())
    }

    let local_path = Path::new(pre_path).join(format!("text_emb_{}.bin", encoder_tag));
    let target_path = shared_dir
        .map(|dir| Path::new(dir).join(format!("text_emb_{}.bin", encoder_tag)))
        .unwrap_or_else(|| local_path.clone());

    write_bf16_file(&target_path, embeddings)?;

    if target_path != local_path {
        if local_path.exists() {
            fs::remove_file(&local_path)?;
        }
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if fs::hard_link(&target_path, &local_path).is_err() {
            fs::copy(&target_path, &local_path)?;
        }
    }

    Ok(())
}

fn run_shallow_pipeline(
    cli: &Cli,
    _db_name: &str,
    pre_path: &str,
    table_map: &TableMap,
    semantic_types: Option<&SemanticTypeMap>,
    cutoffs: &HashMap<String, i64>,
) -> Result<(HashMap<String, i32>, shallow_pca::ShallowMap)> {
    fs::create_dir_all(pre_path)?;

    let (text_to_idx, text_vec, row_text_ids) =
        text_index::collect_text_ids_for_text_columns(table_map, semantic_types)?;

    write_text_vocab(&pre_path, cli.use_rfm_preprocessing, &text_vec)?;

    let encoder = text_encoder::TextEncoder::new(
        &cli.text_encoder,
        Some(&cli.text_encoder_device),
        cli.text_encoder_batch,
        cli.text_encoder_normalize,
        cli.text_encoder_max_length,
    )?;

    let embeddings =
        encode_unique::encode_unique_strings(&encoder, &text_vec, cli.text_encoder_batch)?;

    let encoder_tag = encoder_file_tag(&cli.text_encoder);
    write_text_embeddings(
        &pre_path,
        cli.text_embedding_cache_dir.as_deref(),
        &encoder_tag,
        &embeddings,
    )?;

    let shallow = shallow_pca::fit_and_transform_shallow(
        table_map,
        &row_text_ids,
        &embeddings,
        cli.shallow_size,
        cli.pca_l2_normalize_rows,
        cli.pca_whiten,
        cli.pca_row_chunk,
        cli.pca_batch_size,
        cutoffs,
    )?;

    Ok((text_to_idx, shallow))
}

pub fn main(cli: Cli) {
    prepare_freethreaded_python();

    let cutoffs = match time_cutoffs::load_time_cutoffs(&Some(cli.time_cutoffs_file.clone())) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("Error reading time cutoffs: {}", err);
            std::process::exit(1);
        }
    };

    let specs = match load_task_specs(&cli.task_config, cli.db_name.as_deref()) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("Error loading task specs: {}", err);
            std::process::exit(1);
        }
    };

    for spec in specs {
        eprintln!("Processing database: {}", spec.name);
        eprintln!("  Tasks: {:?}", spec.tasks);
        run_for_db(&cli, &cutoffs, &spec);
    }
}

fn run_for_db(
    cli: &Cli,
    cutoffs: &HashMap<String, i64>,
    db_spec: &DbTaskSpec,
) {
    let db_name = db_spec.name.as_str();

    // Get the validation start time for this database
    let val_time_ns = cutoffs.get(db_name).cloned();

    if val_time_ns.is_none() {
        eprintln!(
            "WARNING: No time cutoff found for database '{}' in cutoffs file '{}'.",
            db_name, cli.time_cutoffs_file
        );
        eprintln!("This may result in data leakage. All data will be treated as training data.");
    }

    let dataset_path = db_spec.dataset_path.clone();

    let dashes = dataset_path.matches("-").count();
    dbg!(dashes);

    println!("reading tables...");
    let tic = Instant::now();
    let mut table_map: TableMap = HashMap::with_hasher(BuildHasherDefault::<DefaultHasher>::new());
    let mut num_rows_sum = 0;
    let mut num_cells_sum = 0;
    let semantic_types = if cli.use_rfm_preprocessing {
        load_semantic_types(db_name)
    } else {
        None
    };

    // Support split structure: database tables at /home/ubuntu/data/rel-xxx/db/
    // and task tables at /home/ubuntu/data/relbench/rel-xxx/tasks/
    let home = var("HOME").unwrap_or_else(|_| ".".to_string());
    let db_path = format!("{}/data/{}/db/*.parquet", home, db_name);
    let task_path_relbench = format!("{}/data/relbench/{}/tasks/*/*.parquet", home, db_name);
    let task_path_scratch = format!("{}/tasks/*/*.parquet", dataset_path);

    // Try both possible locations for task tables
    let task_files: Vec<_> = glob(&task_path_relbench)
        .ok()
        .and_then(|g| Some(g.filter_map(Result::ok).collect()))
        .unwrap_or_else(Vec::new)
        .into_iter()
        .chain(
            glob(&task_path_scratch)
                .ok()
                .and_then(|g| Some(g.filter_map(Result::ok).collect()))
                .unwrap_or_else(Vec::new)
        )
        .collect();

    for (is_db_table, pq_path) in itertools::chain(
        iter::repeat(true).zip(
            glob(&db_path)
                .unwrap_or_else(|_| glob(&format!("{}/db/*.parquet", dataset_path)).unwrap())
                .filter_map(Result::ok)
                .sorted(),
        ),
        iter::repeat(false).zip(task_files.into_iter().sorted()),
    ) {
        // Skip link prediction tasks
        if pq_path.to_str().unwrap().contains("-link-prediction") {
            eprintln!("Skipping link prediction task: {:?}", pq_path);
            continue;
        }

        dbg!(&pq_path);

        let mut file = fs::File::open(&pq_path).unwrap();
        let reader = ParquetReader::new(&mut file);
        let mut df = reader.finish().unwrap();

        let mut reader = ParquetMetaDataReader::new();
        let file = fs::File::open(&pq_path).unwrap();
        reader.try_parse(&file).unwrap();
        let metadata = reader.finish().unwrap();

        let metadata = metadata
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .iter()
            .map(|kv| {
                let key = kv.key.clone();
                let value = kv.value.as_ref().unwrap().clone();
                (key, value)
            })
            .collect::<HashMap<_, _>>();
        let tmp = metadata.get("pkey_col").unwrap();
        let pcol_name = match from_str(tmp).unwrap() {
            Value::Null => None,
            Value::String(s) => Some(s),
            _ => panic!(),
        };
        let tmp = metadata.get("fkey_col_to_pkey_table").unwrap();
        let fcol_name_to_ptable_name = match from_str(tmp).unwrap() {
            Value::Object(o) => o
                .into_iter()
                .map(|(k, v)| {
                    let mut v = match v {
                        Value::String(s) => s,
                        _ => panic!(),
                    };
                    if db_name == "rel-avito" {
                        if k == "UserID" {
                            v = "UserInfo".to_string();
                        } else if k == "AdID" {
                            v = "AdsInfo".to_string();
                        }
                    }
                    (k, v)
                })
                .collect::<HashMap<_, _>>(),
            _ => panic!(),
        };

        let tmp = metadata.get("time_col").unwrap();
        let tcol_name = match from_str(tmp).unwrap() {
            Value::Null => None,
            Value::String(s) => Some(s),
            _ => panic!(),
        };

        let (table_name, table_type) = if is_db_table {
            (
                pq_path.file_stem().unwrap().to_str().unwrap().to_string(),
                TableType::Db,
            )
        } else {
            (
                pq_path
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
                match pq_path.file_stem().unwrap().to_str().unwrap() {
                    "train" => TableType::Train,
                    "val" => TableType::Val,
                    "test" => TableType::Test,
                    _ => panic!(),
                },
            )
        };

        // Skip task tables that are not in the specified task list
        if !is_db_table && !db_spec.tasks.is_empty() && !db_spec.tasks.contains(&table_name) {
            eprintln!("Skipping task not in config: {}", table_name);
            continue;
        }

        let table_key = (table_name.clone(), table_type.clone());

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Binarize columns based on first non-null value
        if db_name == "rel-stack" && table_name == "postLinks" {
            make_column_boolean(&mut df, "LinkTypeId").unwrap();
        }

        // if db_name == "rel-stack" && table_name == "badges" {
        //     make_column_boolean(&mut df, "TagBased").unwrap();

        // }
        if db_name == "rel-trial" && table_name == "studies" {
            make_column_boolean(&mut df, "has_dmc").unwrap();
        }
        // // if db_name == "rel-trial" && table_name == "designs" {
        // //     make_column_boolean(&mut df, "allocation").unwrap();
        // // }
        // if db_name == "rel-trial" && table_name == "studies" {
        //     make_column_boolean(&mut df, "is_fda_regulated_device").unwrap();
        // }

        if db_name == "rel-trial" && table_name == "eligibilities" {
            for col in ["adult", "child", "older_adult"] {
                make_column_boolean(&mut df, col).unwrap();
            }
        }

        // if db_name == "rel-event" && table_name == "users" {
        //     make_column_boolean(&mut df, "gender").unwrap();
        // }
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // stack
        if db_name == "rel-stack" {
            if table_name == "user-engagement" {
                df = cast_col_to_bool(df, "contribution").unwrap();
            }
            if table_name == "user-badge" {
                df = cast_col_to_bool(df, "WillGetBadge").unwrap();
            }
            if table_name == "posts" {
                df = df.drop("AcceptedAnswerId").unwrap();
            }
        }

        // trial
        if db_name == "rel-trial" && table_name == "study-outcome" {
            df = cast_col_to_bool(df, "outcome").unwrap();
        }

        if db_name == "rel-arxiv" && table_name == "paper-citation" {
            df = cast_col_to_bool(df, "cited").unwrap();
        }

        if db_name == "rel-avs" && table_name == "repeater" {
            df = cast_col_to_bool(df, "repeater").unwrap();
        }

        if db_name == "rel-ieeecis" && table_name == "transaction-fraud" {
            df = cast_col_to_bool(df, "isFraud").unwrap();
        }

        // f1
        if db_name == "rel-f1" {
            if table_name == "driver-dnf" {
                df = cast_col_to_bool(df, "did_not_finish").unwrap();
            }
            if table_name == "driver-top3" {
                df = cast_col_to_bool(df, "qualifying").unwrap();
            }
            if table_name == "driver-qualifying-beat-teammate" {
                df = cast_col_to_bool(df, "beats_teammate").unwrap();
            }
            if table_name == "driver-will-race" {
                df = cast_col_to_bool(df, "will_race").unwrap();
            }
            if table_name == "driver-podium" {
                df = cast_col_to_bool(df, "finished_on_podium").unwrap();
            }

            if table_name == "constructor-scores-points" {
                df = cast_col_to_bool(df, "scored_points").unwrap();
            }
        }

        // hm
        if db_name == "rel-hm" && table_name == "user-churn" {
            df = cast_col_to_bool(df, "churn").unwrap();
        }


        // event
        if db_name == "rel-event" {
            if table_name == "event_attendees" {
                df = df.drop_nulls::<String>(None).unwrap();
            }
            if table_name == "user_friends" {
                df = df.drop_nulls::<String>(None).unwrap();
                df = df.with_row_index("dummy".into(), None).unwrap();
            }
            if table_name == "user-repeat"
                || table_name == "user-ignore"
            {
                df = df.drop("index").unwrap();
                df = cast_col_to_bool(df, "target").unwrap();
            }
            if table_name == "user-attendance"
            {
                // user-attendance is a regression task, not classification
                // Keep target as numeric values (0-16), don't cast to bool
                df = df.drop("index").unwrap();
            }
        }

        // avito
        if db_name == "rel-avito" {
            if table_name == "user-visits" {
                df = cast_col_to_bool(df, "num_click").unwrap();
            }
            if table_name == "user-clicks" {
                df = cast_col_to_bool(df, "num_click").unwrap();
            }
        }

        let mut col_name_overrides: HashMap<String, String> = HashMap::new();
        let mut semantic_map: HashMap<String, String> = HashMap::new();
        let mut angle_component_counts: HashMap<String, usize> = HashMap::new();
        if cli.use_rfm_preprocessing {
            let column_types_cow: Cow<HashMap<String, String>> = if let Some(map) = semantic_types
                .as_ref()
                .and_then(|all| all.get(&table_name))
            {
                Cow::Borrowed(map)
            } else {
                let inferred = infer_column_types(&df);
                eprintln!(
                    "Semantic types missing for table {}.{}; inferring from schema.",
                    db_name, table_name
                );
                Cow::Owned(inferred)
            };
            semantic_map = column_types_cow.clone().into_owned();

            log_semantic_snapshot(
                "before preprocessing",
                &table_name,
                &table_type,
                &df,
                &semantic_map,
                None,
                None,
            );

            if let Err(err) = apply_angle_hashing_for_categorical(
                &mut df,
                db_name,
                &table_name,
                column_types_cow.as_ref(),
                &mut col_name_overrides,
                &mut angle_component_counts,
            ) {
                eprintln!(
                    "Failed to apply angle hashing for table {} column types: {}",
                    table_name, err
                );
            }

            for (derived, base) in &col_name_overrides {
                semantic_map.insert(derived.clone(), "shallow_c".to_string());
                semantic_map.insert(base.clone(), "shallow_c".to_string());
            }
            for (col_name, ty) in column_types_cow.iter() {
                if ty == "categorical" {
                    semantic_map.remove(col_name);
                }
            }

            log_semantic_snapshot(
                "after preprocessing",
                &table_name,
                &table_type,
                &df,
                &semantic_map,
                Some(&col_name_overrides),
                Some(&angle_component_counts),
            );
        }

        let num_rows = df.height() as i32;
        let num_cells = num_rows * df.width() as i32;
        table_map.insert(
            table_key,
            Table {
                table_name: table_name.to_string(),
                df,
                col_stats: Vec::new(),
                pcol_name,
                fcol_name_to_ptable_name,
                tcol_name,
                node_idx_offset: num_rows_sum,
                col_name_overrides,
                semantic_types: semantic_map,
                angle_component_counts,
            },
        );
        num_rows_sum += num_rows;
        num_cells_sum += num_cells;
    }
    println!("done in {:?}.", tic.elapsed());

    let pre_path = if let Some(ref output_dir) = cli.output_dir {
        format!("{}/{}", output_dir, db_name)
    } else if let Ok(rt_pre_dir) = var("RT_PRE_DIR") {
        format!("{}/{}", rt_pre_dir, db_name)
    } else {
        let home = var("HOME").unwrap();
        // Try /home/ubuntu/data/pre first, fall back to scratch
        let data_pre = format!("{}/data/pre", home);
        let scratch_pre = format!("{}/scratch/pre", home);
        let base_dir = if std::path::Path::new(&data_pre).exists() {
            data_pre
        } else {
            scratch_pre
        };
        format!("{}/{}", base_dir, db_name)
    };

    let (mut text_to_idx, shallow_map) = if cli.use_rfm_preprocessing {
        run_shallow_pipeline(&cli, db_name, &pre_path, &table_map, semantic_types.as_ref(), cutoffs).expect("shallow pipeline failed")
    } else {
        (HashMap::new(), HashMap::new())
    };

    if cli.use_rfm_preprocessing {
        for ((tname, ttype, col), sh) in &shallow_map {
            if let Some(table) = table_map.get_mut(&(tname.clone(), ttype.clone())) {
                table
                    .semantic_types
                    .insert(col.clone(), "shallow".to_string());
                table
                    .angle_component_counts
                    .insert(col.clone(), sh.k as usize);
            }
        }
        for ((table_name, table_type), table) in &table_map {
            log_semantic_snapshot(
                "after shallow",
                table_name,
                table_type,
                &table.df,
                &table.semantic_types,
                Some(&table.col_name_overrides),
                Some(&table.angle_component_counts),
            );
        }
    }

    println!("computing column stats (Pass 1: Fit on Train Split)...");
    let tic = Instant::now();

    // This map will store train-only stats, keyed by table name.
    let mut col_stats_map: HashMap<String, Vec<ColStat>> = HashMap::new();

    // Pass 1: Iterate all tables to compute and store train-only stats
    for ((table_name, table_type), table) in &table_map {
        // We only compute stats on Db tables and Train task tables.
        // Val/Test task tables will later borrow stats from the Train table.
        if table_type == &TableType::Val || table_type == &TableType::Test {
            continue;
        }

        // 1. Get the time column for this table, if it exists.
        let tcol = table.tcol_name.as_ref().and_then(|name| {
            table.df.column(name).ok().map(|c| c.as_materialized_series().clone())
        });

        // 2. Create a train_mask based on the time cutoff.
        let train_mask = if let (Some(val_time), Some(tcol)) = (val_time_ns, tcol) {
            // If we have a cutoff time and a time column, create a mask.
            let mask_values: Vec<bool> = (0..tcol.len())
                .map(|i| datetime_ns_at(&tcol, i).map(|ts| ts < val_time).unwrap_or(false))
                .collect();
            Series::new("mask".into(), mask_values)
                .cast(&DataType::Boolean).unwrap().bool().unwrap().clone()
        } else {
            // If no cutoff or no time col, assume all rows are train (legacy behavior).
            Series::new("mask".into(), vec![true; table.df.height()])
                .cast(&DataType::Boolean).unwrap().bool().unwrap().clone()
        };

        let mut table_stats: Vec<ColStat> = Vec::with_capacity(table.df.width());

        // 3. Iterate columns and compute stats using the train_mask
        for col in table.df.iter() {
            let col = col.rechunk();
            match col.dtype() {
                DataType::Boolean => {
                    let col_float = col.cast(&DataType::Float64).unwrap();
                    // Apply train_mask first, then filter nulls
                    let col_train = col_float.filter(&train_mask).unwrap();
                    let col_train = col_train.filter(&col_train.is_not_null()).unwrap();

                    let col_mean = col_train.mean().unwrap_or(0.0);
                    let mut col_std = col_train.std(1).unwrap_or(0.0);
                    if col_std == 0.0 { col_std = 1.0; }
                    table_stats.push(ColStat { mean: col_mean, std: col_std });
                }
                DataType::UInt32 | DataType::Int32 | DataType::Int64 |
                DataType::Float64 | DataType::Float32 => {
                    let col_float = col.cast(&DataType::Float64).unwrap();
                    // Apply train_mask first, then filter nulls and NaNs
                    let col_train = col_float.filter(&train_mask).unwrap();
                    let col_train = col_train.filter(&col_train.is_not_null()).unwrap();
                    let col_train = col_train.filter(&col_train.is_not_nan().unwrap()).unwrap();

                    let mean = col_train.mean().unwrap_or(0.0);
                    let std = col_train.std(1).unwrap_or(1.0);
                    let std = if std == 0.0 { 1.0 } else { std };
                    table_stats.push(ColStat { mean, std });
                }
                DataType::Datetime(u, _) => {
                    // This now correctly computes per-column stats for Datetime
                    assert!(*u == TimeUnit::Nanoseconds);
                    let col_float = col.cast(&DataType::Float64).unwrap();
                    // Apply train_mask first, then filter nulls and NaNs
                    let col_train = col_float.filter(&train_mask).unwrap();
                    let col_train = col_train.filter(&col_train.is_not_null()).unwrap();
                    let col_train = col_train.filter(&col_train.is_not_nan().unwrap()).unwrap();

                    let mean = col_train.mean().unwrap_or(0.0);
                    let std = col_train.std(1).unwrap_or(1.0);
                    let std = if std == 0.0 { 1.0 } else { std };
                    table_stats.push(ColStat { mean, std });
                }
                _ => {
                    // Other types (String, etc.) get placeholder stats
                    table_stats.push(ColStat { mean: 0.0, std: 0.0 });
                }
            }
        }
        // Store the computed train-only stats for this table
        col_stats_map.insert(table_name.clone(), table_stats);
    }

    // Pass 1.5: Apply the correct train-only stats to ALL tables in table_map
    for ((table_name, table_type), table) in table_map.iter_mut() {
        // Val and Test tables get the stats from their corresponding Train table.
        // Db and Train tables get the stats computed from themselves.
        // (We stored all computed stats under their base table_name)
        if let Some(stats) = col_stats_map.get(table_name) {
            table.col_stats = stats.clone();
        } else {
            // Fallback for safety (e.g., a Val table with no Train table)
            eprintln!("Warning: No train stats found for table {}. Using default stats.", table_name);
            table.col_stats = vec![ColStat { mean: 0.0, std: 1.0 }; table.df.width()];
        }
    }
    println!("done in {:?}.", tic.elapsed());

    println!("making node vector...");
    let tic = Instant::now();
    let pbar = ProgressBar::new(num_cells_sum as u64).with_style(
        ProgressStyle::default_bar()
            .template(PBAR_TEMPLATE)
            .unwrap(),
    );
    let mut column_name_to_idx: Vec<(String, i32)> = Vec::new();
    let mut seen_column_names: HashSet<String> = HashSet::new();
    let mut missing_shallow: HashSet<(String, TableType, String)> = HashSet::new();
    let mut node_vec = (0..num_rows_sum)
        .map(|_| Node::default())
        .collect::<Vec<_>>();
    let mut p2f_adj = Adj {
        adj: vec![Vec::new(); num_rows_sum as usize],
    };

    for ((_table_name, table_type), table) in &table_map {
        if cli.skip_db && table_type == &TableType::Db {
            println!(
                "skipping table {} of type {:?}",
                table.table_name, table_type
            );
            continue;
        }

        let table_name_idx = if cli.use_rfm_preprocessing {
            *text_to_idx
                .get(&table.table_name)
                .unwrap_or_else(|| panic!("missing table name idx for {}", table.table_name))
        } else {
            let l = text_to_idx.len() as i32;
            *text_to_idx
                .entry(table.table_name.clone())
                .or_insert_with(|| l)
        };

        let column_names = table.df.get_column_names();
        let mut angle_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, col_name) in column_names.iter().enumerate() {
            if let Some(base) = table
                .col_name_overrides
                .get(&col_name.to_string())
            {
                angle_groups.entry(base.clone()).or_default().push(idx);
            }
        }
        let mut processed_angle_bases: HashSet<String> = HashSet::new();
        let mut col_idx = 0usize;
        let width = table.df.width();
        while col_idx < width {
            let raw_col_series = table
                .df
                .select_at_idx(col_idx)
                .unwrap()
                .clone()
                .rechunk();
            let col_stat = &table.col_stats[col_idx];
            let col_name_key = raw_col_series.name().to_string();
            let semantic_label = table
                .semantic_types
                .get(&col_name_key)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            let display_col_name = table
                .col_name_overrides
                .get(&col_name_key)
                .cloned()
                .unwrap_or_else(|| col_name_key.clone());
            let col_name = format!("{} of {}", display_col_name, table.table_name.clone());
            let col_name_idx = if cli.use_rfm_preprocessing {
                let idx = *text_to_idx
                    .get(&col_name)
                    .unwrap_or_else(|| panic!("missing column name idx for {}", col_name));
                if seen_column_names.insert(col_name.clone()) {
                    column_name_to_idx.push((col_name.clone(), idx));
                }
                idx
            } else {
                let l = text_to_idx.len() as i32;
                let entry = text_to_idx.entry(col_name.clone()).or_insert_with(|| {
                    column_name_to_idx.push((col_name.clone(), l));
                    l
                });
                *entry
            };

            if let Some(base) = table.col_name_overrides.get(&col_name_key) {
                if processed_angle_bases.insert(base.clone()) {
                    if let Some(indices) = angle_groups.get(base) {
                        let k = indices.len();
                        let angle_stats: Vec<&ColStat> =
                            indices.iter().map(|&i| &table.col_stats[i]).collect();
                        let angle_series: Vec<_> = indices
                            .iter()
                            .map(|&i| {
                                table
                                    .df
                                    .select_at_idx(i)
                                    .unwrap()
                                    .clone()
                                    .f32()
                                    .unwrap()
                                    .clone()
                            })
                            .collect();

                        for r in 0..raw_col_series.len() {
                            pbar.inc(k as u64);
                            let node_idx = table.node_idx_offset + r as i32;
                            let node = &mut node_vec[node_idx as usize];
                            node.is_task_node = table_type != &TableType::Db;
                            node.node_idx = node_idx;
                            node.table_name_idx = table_name_idx;

                            let mut features = Vec::with_capacity(k);
                            let mut has_null = false;
                            for (chunk, stat) in angle_series.iter().zip(&angle_stats) {
                                match chunk.get(r) {
                                    Some(mut val) => {
                                        if val.is_nan() {
                                            has_null = true;
                                            break;
                                        }
                                        let mut std = stat.std;
                                        if std == 0.0 {
                                            std = 1.0;
                                        }
                                        val = ((val as f64 - stat.mean) / std) as f32;
                                        if !val.is_finite() {
                                            has_null = true;
                                            break;
                                        }
                                        features.push(val);
                                    }
                                    None => {
                                        has_null = true;
                                        break;
                                    }
                                }
                            }
                            if has_null {
                                continue;
                            }

                            let shallow_start = node.shallow_storage.len();
                            node.shallow_storage.extend_from_slice(&features);
                            node.boolean_values.push(0.0);
                            node.number_values.push(0.0);
                            node.text_values.push(0);
                            node.datetime_values.push(0.0);
                            node.sem_types
                                .push(SemType::ShallowCategorical(k as u8));
                            node.col_name_idxs.push(col_name_idx);
                            node.class_value_idx.push(-1);
                            node.shallow_offsets.push(shallow_start as u32);
                        }
                    }
                }
                col_idx += 1;
                continue;
            }

            if raw_col_series.name() == table.pcol_name.as_deref().unwrap_or("") {
                pbar.inc(raw_col_series.len() as u64);
                col_idx += 1;
                continue;
            }

            let raw_name_string = raw_col_series.name().to_string();

            if table
                .fcol_name_to_ptable_name
                .contains_key(&raw_name_string)
            {
                let ptable_name = table
                    .fcol_name_to_ptable_name
                    .get(&raw_name_string)
                    .unwrap();
                let ptable_offset = table_map
                    .get(&(ptable_name.to_string(), TableType::Db))
                    .unwrap_or_else(|| {
                        dbg!(ptable_name.to_string());
                        dbg!(table_map.keys());
                        panic!()
                    })
                    .node_idx_offset;

                // Check if this is a List column (link prediction task)
                // If so, skip it as we don't handle link prediction in this preprocessing
                let first_val = raw_col_series.get(0).unwrap();
                if matches!(first_val, AnyValue::List(_)) {
                    eprintln!("Skipping List-type column '{}' (link prediction task not supported in preprocessing)", raw_name_string);
                    pbar.inc(raw_col_series.len() as u64);
                    col_idx += 1;
                    continue;
                }

                for r in 0..raw_col_series.len() {
                    let val = raw_col_series.get(r).unwrap();
                    pbar.inc(1);
                    if val == AnyValue::Null {
                        continue;
                    }

                    let node_idx = table.node_idx_offset + r as i32;
                    let node = node_vec.get_mut(node_idx as usize).unwrap();
                    node.is_task_node = table_type != &TableType::Db;
                    node.node_idx = node_idx;
                    node.table_name_idx = table_name_idx;

                    let AnyValue::Int64(val) = val else {
                        dbg!(val);
                        dbg!(raw_col_series.name());
                        panic!()
                    };
                    let pnode_idx = ptable_offset + val as i32;

                    node.f2p_nbr_idxs.push(pnode_idx);

                    let timestamp = if let Some(c) = &table.tcol_name {
                        table
                            .df
                            .column(c)
                            .ok()
                            .and_then(|col| datetime_seconds_at(&col.as_materialized_series(), r))
                    } else {
                        None
                    };
                    node.timestamp = timestamp;

                    let ptable_name_idx = if cli.use_rfm_preprocessing {
                        *text_to_idx
                            .get(ptable_name)
                            .unwrap_or_else(|| panic!("missing parent table idx for {}", ptable_name))
                    } else {
                        let l = text_to_idx.len() as i32;
                        *text_to_idx
                            .entry(ptable_name.to_string())
                            .or_insert_with(|| l)
                    };

                    let ptable = &table_map[&(ptable_name.to_string(), TableType::Db)];
                    let ptimestamp = if ptable.tcol_name.is_some() {
                        let tcol_name = ptable.tcol_name.as_ref().unwrap();
                        ptable
                            .df
                            .column(tcol_name)
                            .ok()
                            .and_then(|col| datetime_seconds_at(&col.as_materialized_series(), val as usize))
                    } else {
                        None
                    };

                    let f2p_edge = Edge {
                        node_idx: pnode_idx,
                        table_name_idx: ptable_name_idx,
                        table_type: TableType::Db,
                        timestamp: ptimestamp,
                    };
                    node.f2p_edges.push(f2p_edge);

                    let p2f_edge = Edge {
                        node_idx,
                        table_name_idx,
                        table_type: table_type.clone(),
                        timestamp,
                    };
                    p2f_adj.adj[pnode_idx as usize].push(p2f_edge)
                }
                col_idx += 1;
                continue;
            }

            for r in 0..raw_col_series.len() {
                pbar.inc(1);
                let node_idx = table.node_idx_offset + r as i32;
                let node = &mut node_vec[node_idx as usize];
                node.is_task_node = table_type != &TableType::Db;
                node.node_idx = node_idx;
                node.table_name_idx = table_name_idx;

                let val = raw_col_series.get(r).unwrap();
                match val {
                    AnyValue::Null => {}
                    AnyValue::Boolean(v) => {
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let val_float = ((if v { 1.0 } else { 0.0 }) - col_stat.mean) / std;
                        node.boolean_values.push(val_float as f32);
                        node.number_values.push(0.0);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Boolean);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::UInt32(v) => {
                        let value = v as f64;
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let normalized = (value - col_stat.mean) / std;
                        node.boolean_values.push(0.0);
                        node.number_values.push(normalized as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::Int32(v) => {
                        let value = v as f64;
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let normalized = (value - col_stat.mean) / std;
                        node.boolean_values.push(0.0);
                        node.number_values.push(normalized as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::Int64(v) => {
                        let value = v as f64;
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let normalized = (value - col_stat.mean) / std;
                        node.boolean_values.push(0.0);
                        node.number_values.push(normalized as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::Float32(v) => {
                        let value = v as f64;
                        if value.is_nan() {
                            continue;
                        }
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let normalized = (value - col_stat.mean) / std;
                        if !normalized.is_finite() {
                            dbg!(&table.table_name);
                            dbg!(raw_col_series.name());
                            dbg!(col_stat);
                            panic!();
                        }
                        node.boolean_values.push(0.0);
                        node.number_values.push(normalized as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::Float64(v) => {
                        if v.is_nan() {
                            continue;
                        }
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let normalized = (v - col_stat.mean) / std;
                        if !normalized.is_finite() {
                            dbg!(&table.table_name);
                            dbg!(raw_col_series.name());
                            dbg!(col_stat);
                            panic!();
                        }
                        node.boolean_values.push(0.0);
                        node.number_values.push(normalized as f32);
                        node.text_values.push(0);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Number);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::Datetime(v, unit, _) => {
                        assert_eq!(unit, TimeUnit::Nanoseconds);

                        // Use per-column statistics instead of global dt_mean/dt_std
                        let value = v as f64;
                        let mut std = col_stat.std;
                        if std == 0.0 {
                            std = 1.0;
                        }
                        let val = (value - col_stat.mean) / std;

                        node.boolean_values.push(0.0);
                        node.number_values.push(0.0);
                        node.text_values.push(0);
                        node.datetime_values.push(val as f32);
                        node.sem_types.push(SemType::DateTime);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(-1);
                        node.shallow_offsets.push(0);
                    }
                    AnyValue::String(val) => {
                        let treat_as_text = if cli.use_rfm_preprocessing {
                            matches!(
                                semantic_label.as_str(),
                                "text_embedded" | "shallow"
                            ) || semantic_label == "unknown"
                        } else {
                            true
                        };
                        if !treat_as_text {
                            continue;
                        }

                        let expects_shallow =
                            cli.use_rfm_preprocessing && semantic_label.as_str() == "shallow";
                        let mut used_shallow = false;

                        if cli.use_rfm_preprocessing {
                            let raw_name_string = raw_col_series.name().to_string();
                            let mut column_candidates = Vec::with_capacity(2);
                            column_candidates.push(display_col_name.clone());
                            if raw_name_string != display_col_name {
                                column_candidates.push(raw_name_string.clone());
                            }

                            for column_name in column_candidates {
                                if let Some(sh) = shallow_map.get(&(
                                    table.table_name.clone(),
                                    table_type.clone(),
                                    column_name.clone(),
                                )) {
                                    if r < sh.offsets.len() {
                                        let off = sh.offsets[r];
                                        if off >= 0 {
                                            let start = off as usize;
                                            let k = sh.k as usize;
                                            let slice = &sh.storage[start..start + k];
                                            let shallow_start = node.shallow_storage.len();
                                            node.shallow_storage.extend_from_slice(slice);
                                            node.boolean_values.push(0.0);
                                            node.number_values.push(0.0);
                                            node.text_values.push(0);
                                            node.datetime_values.push(0.0);
                                            node.sem_types.push(SemType::Shallow(sh.k));
                                            node.col_name_idxs.push(col_name_idx);
                                            let text_idx = *text_to_idx
                                                .get(val)
                                                .unwrap_or_else(|| panic!(
                                                    "text id missing for {}",
                                                    val
                                                ));
                                            node.class_value_idx.push(text_idx);
                                            node.shallow_offsets.push(shallow_start as u32);
                                            used_shallow = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if used_shallow {
                            continue;
                        }

                        if expects_shallow {
                            let key = (
                                table.table_name.clone(),
                                table_type.clone(),
                                display_col_name.clone(),
                            );
                            if missing_shallow.insert(key) {
                                eprintln!(
                                    "missing shallow embedding for {} ({:?}) column {}",
                                    table.table_name, table_type, display_col_name
                                );
                            }
                            continue;
                        }

                        let text_idx = if cli.use_rfm_preprocessing {
                            *text_to_idx
                                .get(val)
                                .unwrap_or_else(|| panic!("text id missing for {}", val))
                        } else {
                            let l = text_to_idx.len() as i32;
                            *text_to_idx.entry(val.to_string()).or_insert_with(|| l)
                        };
                        node.boolean_values.push(0.0);
                        node.number_values.push(0.0);
                        node.text_values.push(text_idx);
                        node.datetime_values.push(0.0);
                        node.sem_types.push(SemType::Text);
                        node.col_name_idxs.push(col_name_idx);
                        node.class_value_idx.push(text_idx);
                        node.shallow_offsets.push(0);
                    }
                    _ => {
                        dbg!(&table.table_name);
                        dbg!(raw_col_series.name());
                        dbg!(val);
                        panic!()
                    }
                }
            }
            col_idx += 1;
        }
    }
    pbar.finish();
    println!("done in {:?}.", tic.elapsed());

    fs::create_dir_all(Path::new(&pre_path)).unwrap();

    if !cli.use_rfm_preprocessing {
        println!("writing out text...");
        let tic = Instant::now();
        let mut text_vec = vec![String::new(); text_to_idx.len()];
        for (k, v) in text_to_idx {
            text_vec[v as usize] = k;
        }
        let file = fs::File::create(output_path(&pre_path, "text", "json", false)).unwrap();
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &text_vec).unwrap();

        let mut text_map: HashMap<String, i32> = HashMap::new();
        for (idx, s) in text_vec.iter().enumerate() {
            text_map.insert(s.clone(), idx as i32);
        }
        let file = fs::File::create(output_path(&pre_path, "text_map", "json", false)).unwrap();
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &text_map).unwrap();
        println!("done in {:?}.", tic.elapsed());
    }

    println!("writing out column index...");
    let tic = Instant::now();
    // Write column name to index mapping collected during node processing
    let column_index: HashMap<String, i32> = column_name_to_idx.into_iter().collect();
    let file = fs::File::create(output_path(
        &pre_path,
        "column_index",
        "json",
        cli.use_rfm_preprocessing,
    ))
    .unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &column_index).unwrap();
    println!("done in {:?}.", tic.elapsed());

    if cli.use_rfm_preprocessing {
        let meta_path = output_path(&pre_path, "shallow_meta", "json", true);
        let file = fs::File::create(&meta_path).unwrap();
        let mut writer = BufWriter::new(file);
        let meta = json!({
            "shallow_dim": cli.shallow_size,
            "shallow_c_dim": ANGLE_HASH_DIM,
        });
        serde_json::to_writer(&mut writer, &meta).unwrap();
    }

    println!("writing out table info...");
    let tic = Instant::now();
    let mut table_info_map = HashMap::new();
    for (table_key, table) in &table_map {
        let key = format!("{}:{:?}", table_key.0, table_key.1);
        let column_aliases = HashMap::new();
        let mut column_display_overrides = HashMap::new();

        if cli.use_rfm_preprocessing {
            for (derived, base) in &table.col_name_overrides {
                let derived_label = format!("{} of {}", derived, table.table_name);
                let base_label = format!("{} of {}", base, table.table_name);
                let display = if let Some(count) = table.angle_component_counts.get(base) {
                    format!("{} (shallow_c k={})", base_label, count)
                } else {
                    base_label
                };
                column_display_overrides.insert(derived_label, display);
            }

            for ((tname, ttype, col), sh) in &shallow_map {
                if &table.table_name == tname && &table_key.1 == ttype {
                    let label = format!("{} of {}", col, table.table_name);
                    let shallow_label = format!("{} (shallow k={})", label, sh.k);
                    column_display_overrides.insert(label, shallow_label);
                }
            }
        }

        table_info_map.insert(
            key,
            TableInfo {
                node_idx_offset: table.node_idx_offset,
                num_nodes: table.df.height() as i32,
                column_aliases,
                column_display_overrides,
            },
        );
    }

    let file = fs::File::create(output_path(
        &pre_path,
        "table_info",
        "json",
        cli.use_rfm_preprocessing,
    ))
    .unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &table_info_map).unwrap();
    println!("done in {:?}.", tic.elapsed());

    let tic = Instant::now();
    let mut offsets = vec![0];
    let pbar = ProgressBar::new(node_vec.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template(PBAR_TEMPLATE)
            .unwrap(),
    );

    println!("writing out nodes...");
    let file = fs::File::create(output_path(
        &pre_path,
        "nodes",
        "rkyv",
        cli.use_rfm_preprocessing,
    ))
    .unwrap();
    let mut writer = BufWriter::new(file);
    for node in node_vec {
        let bytes = rkyv::to_bytes::<Error>(&node).unwrap();
        writer.write_all(&bytes).unwrap();
        offsets.push(writer.stream_position().unwrap() as i64);
        pbar.inc(1);
    }
    pbar.finish();

    println!("writing out offsets...");
    let file = fs::File::create(output_path(
        &pre_path,
        "offsets",
        "rkyv",
        cli.use_rfm_preprocessing,
    ))
    .unwrap();
    let mut writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&Offsets { offsets }).unwrap();
    writer.write_all(&bytes).unwrap();

    println!("writing out p2f_adj...");
    let file = fs::File::create(output_path(
        &pre_path,
        "p2f_adj",
        "rkyv",
        cli.use_rfm_preprocessing,
    ))
    .unwrap();
    let mut writer = BufWriter::new(file);
    let bytes = rkyv::to_bytes::<Error>(&p2f_adj).unwrap();
    writer.write_all(&bytes).unwrap();

    println!("done in {:?}.", tic.elapsed());
}
