use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedSemType, ArchivedTableType,
    Offsets,
};
use clap::Parser;
use half::bf16;
use itertools::izip;
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError};
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::seq::index;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use serde::Deserialize;
use serde_json;
use std::collections::HashMap;
use std::env::var;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::str;
use std::sync::Arc;
use std::time::Instant;

const MAX_F2P_NBRS: usize = 5;

fn embedding_file_tag(model: &str) -> String {
    model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

#[derive(Deserialize)]
struct TableInfoEntry {
    #[serde(default)]
    column_display_overrides: HashMap<String, String>,
}

fn load_column_name_remap(pre_path: &str, suffix: &str) -> PyResult<HashMap<i32, i32>> {
    let mut remap = HashMap::new();

    let text_map_path = format!("{}/text_map{}.json", pre_path, suffix);
    let table_info_path = format!("{}/table_info{}.json", pre_path, suffix);

    // Handle text_map.json
    let text_map: HashMap<String, i32> = match fs::File::open(&text_map_path) {
        Ok(file) => serde_json::from_reader(file).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to parse JSON from {}: {}",
                text_map_path, e
            ))
        })?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Text map is optional, return empty remap if not found
            return Ok(remap);
        }
        Err(e) => {
            return Err(PyFileNotFoundError::new_err(format!(
                "Failed to open text_map file {}: {}",
                text_map_path, e
            )));
        }
    };

    // Handle table_info.json
    let table_info: HashMap<String, TableInfoEntry> = match fs::File::open(&table_info_path) {
        Ok(file) => serde_json::from_reader(file).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to parse JSON from {}: {}",
                table_info_path, e
            ))
        })?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Table info is optional, return empty remap if not found
            return Ok(remap);
        }
        Err(e) => {
            return Err(PyFileNotFoundError::new_err(format!(
                "Failed to open table_info file {}: {}",
                table_info_path, e
            )));
        }
    };

    for entry in table_info.values() {
        for (derived, base) in &entry.column_display_overrides {
            if let (Some(&derived_idx), Some(&base_idx)) =
                (text_map.get(derived), text_map.get(base))
            {
                remap.insert(derived_idx, base_idx);
            }
        }
    }

    Ok(remap)
}

fn open_file(path: &Path, what: &str) -> PyResult<fs::File> {
    fs::File::open(path).map_err(|e| {
        PyFileNotFoundError::new_err(format!("Failed to open {what} at {}: {e}", path.display()))
    })
}

fn mmap_file(path: &Path, what: &str) -> PyResult<Mmap> {
    let file = open_file(path, what)?;
    unsafe {
        Mmap::map(&file).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to memory-map {what} at {}: {e}",
                path.display()
            ))
        })
    }
}

fn read_all(path: &Path, what: &str) -> PyResult<Vec<u8>> {
    let file = open_file(path, what)?;
    let mut bytes = Vec::new();
    BufReader::new(file).read_to_end(&mut bytes).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Failed to read {what} from {}: {e}",
            path.display()
        ))
    })?;
    Ok(bytes)
}

fn preflight_required_files(
    pre_path: &str,
    suffix: &str,
    embedding_model: &str,
    db_name: &str,
) -> PyResult<()> {
    let emb_tag = embedding_file_tag(embedding_model);
    let required = [
        (
            "nodes",
            PathBuf::from(format!("{}/nodes{}.rkyv", pre_path, suffix)),
        ),
        (
            "text embeddings",
            PathBuf::from(format!("{}/text_emb_{}.bin", pre_path, emb_tag)),
        ),
        (
            "offsets",
            PathBuf::from(format!("{}/offsets{}.rkyv", pre_path, suffix)),
        ),
        (
            "p2f adjacency",
            PathBuf::from(format!("{}/p2f_adj{}.rkyv", pre_path, suffix)),
        ),
    ];

    for (what, path) in required.iter() {
        if !path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "Missing required file ({what}): {}\n\
Hint: Did you run relational_transformer/scripts/preprocess_all_databases.sh {}?",
                path.display(),
                db_name
            )));
        }
    }
    Ok(())
}

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    class_value_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    shallow_values: Vec<bf16>,
    shallow_c_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    boolean_values: Vec<bf16>,
    masks: Vec<bool>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    true_batch_size: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    class_value_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    shallow_values: &'a mut [bf16],
    shallow_c_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    boolean_values: &'a mut [bf16],
    masks: &'a mut [bool],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
}

impl Vecs {
    fn new(
        batch_size: usize,
        seq_len: usize,
        true_batch_size: usize,
        d_text: usize,
        d_shallow: usize,
        d_shallow_c: usize,
    ) -> Self {
        let l = batch_size * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: vec![bf16::ZERO; l * d_text],
            sem_types: vec![0; l],
            number_values: vec![bf16::ZERO; l],
            text_values: vec![bf16::ZERO; l * d_text],
            shallow_values: vec![bf16::ZERO; l * d_shallow],
            shallow_c_values: vec![bf16::ZERO; l * d_shallow_c],
            datetime_values: vec![bf16::ZERO; l],
            boolean_values: vec![bf16::ZERO; l],
            masks: vec![false; l],
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            true_batch_size,
        }
    }

    fn chunks_exact_mut(
        &mut self,
        seq_len: usize,
        d_text: usize,
        d_shallow: usize,
        d_shallow_c: usize,
    ) -> impl Iterator<Item = Slices<'_>> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.class_value_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.shallow_values.chunks_exact_mut(seq_len * d_shallow),
            self.shallow_c_values
                .chunks_exact_mut(seq_len * d_shallow_c),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.boolean_values.chunks_exact_mut(seq_len),
            self.masks.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                shallow_values,
                shallow_c_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                shallow_values,
                shallow_c_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        let mut results = Vec::with_capacity(18);

        results.push(("node_idxs", PyArray1::from_vec(py, self.node_idxs)).into_py_any(py)?);
        results.push(("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs)).into_py_any(py)?);
        results.push(
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)?,
        );
        results
            .push(("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs)).into_py_any(py)?);
        results.push(
            (
                "class_value_idxs",
                PyArray1::from_vec(py, self.class_value_idxs),
            )
                .into_py_any(py)?,
        );
        results.push(
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)?,
        );
        results.push(("sem_types", PyArray1::from_vec(py, self.sem_types)).into_py_any(py)?);
        results
            .push(("number_values", PyArray1::from_vec(py, self.number_values)).into_py_any(py)?);
        results.push(("text_values", PyArray1::from_vec(py, self.text_values)).into_py_any(py)?);
        results.push(
            (
                "shallow_values",
                PyArray1::from_vec(py, self.shallow_values),
            )
                .into_py_any(py)?,
        );
        results.push(
            (
                "shallow_c_values",
                PyArray1::from_vec(py, self.shallow_c_values),
            )
                .into_py_any(py)?,
        );
        results.push(
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)?,
        );
        results.push(
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)?,
        );
        results.push(("masks", PyArray1::from_vec(py, self.masks)).into_py_any(py)?);
        results.push(("is_targets", PyArray1::from_vec(py, self.is_targets)).into_py_any(py)?);
        results
            .push(("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes)).into_py_any(py)?);
        results.push(("is_padding", PyArray1::from_vec(py, self.is_padding)).into_py_any(py)?);
        results.push(("true_batch_size", self.true_batch_size).into_py_any(py)?);

        Ok(results)
    }
}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Mmap,
    offsets: Vec<i64>,
    col_name_idx_remap: Arc<HashMap<i32, i32>>,
    rfm_preprocessed: bool,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
}

#[pyclass]
pub struct Sampler {
    batch_size: usize,
    seq_len: usize,
    rank: usize,
    world_size: usize,
    datasets: Vec<Dataset>,
    items: Vec<Item>,
    max_bfs_width: usize,
    epoch: u64,
    d_text: usize,
    d_shallow: usize,
    d_shallow_c: usize,
    seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,
}

#[pymethods]
impl Sampler {
    #[new]
    #[pyo3(signature = (
        dataset_tuples,
        batch_size,
        seq_len,
        rank,
        world_size,
        max_bfs_width,
        embedding_model,
        d_text,
        seed,
        target_columns,
        columns_to_drop
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_tuples: Vec<(String, i32, i32)>,
        batch_size: usize,
        seq_len: usize,
        rank: usize,
        world_size: usize,
        max_bfs_width: usize,
        embedding_model: &str,
        d_text: usize,
        seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
    ) -> PyResult<Self> {
        // eprintln!("[sampler] Starting Sampler::new");
        let mut datasets = Vec::new();
        let mut items = Vec::new();
        let mut col_name_remap_cache: HashMap<String, Arc<HashMap<i32, i32>>> = HashMap::new();

        // Determine pre_root using same logic as Python's _pre_root()
        let pre_root = if let Ok(rt_pre_dir) = var("RT_PRE_DIR") {
            rt_pre_dir
        } else {
            let home = var("HOME").map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to read HOME environment variable: {}", err))
            })?;
            let data_pre = format!("{}/data/pre", home);
            let scratch_pre = format!("{}/scratch/pre", home);
            if std::path::Path::new(&data_pre).exists() {
                data_pre
            } else {
                scratch_pre
            }
        };

        let emb_tag = embedding_file_tag(embedding_model);
        let d_shallow = 3usize;
        let d_shallow_c = 4usize;
        for (i, (db_name, node_idx_offset, num_nodes)) in dataset_tuples.into_iter().enumerate() {
            let mut parts = db_name.splitn(2, "::");
            let db_name_base = parts.next().unwrap_or_default().to_owned();
            let use_rfm_preprocessing = matches!(parts.next(), Some("rfm"));
            let pre_path = format!("{}/{}", pre_root, db_name_base);
            let suffix = if use_rfm_preprocessing { "_rfm" } else { "" };

            preflight_required_files(&pre_path, suffix, embedding_model, &db_name_base)?;

            let col_name_idx_remap = if use_rfm_preprocessing {
                let cache_key = format!("{}{}", db_name_base, suffix);
                if let Some(cached) = col_name_remap_cache.get(&cache_key) {
                    cached.clone()
                } else {
                    let remap = load_column_name_remap(&pre_path, suffix)?;
                    let arc_remap = Arc::new(remap);
                    col_name_remap_cache.insert(cache_key, arc_remap.clone());
                    arc_remap
                }
            } else {
                Arc::new(HashMap::new())
            };

            let trace_io = std::env::var("RT_TRACE_IO").ok().as_deref() == Some("1");
            let nodes_path = PathBuf::from(format!("{}/nodes{}.rkyv", pre_path, suffix));
            let text_path = PathBuf::from(format!(
                "{}/text_emb_{}.bin",
                pre_path, emb_tag
            ));
            let offsets_path = PathBuf::from(format!("{}/offsets{}.rkyv", pre_path, suffix));
            let p2f_adj_path = PathBuf::from(format!("{}/p2f_adj{}.rkyv", pre_path, suffix));

            if trace_io {
                eprintln!("[sampler] pre_root={}", pre_root);
                eprintln!(
                    "[sampler] db_name='{}'  base='{}'  use_rfm={}  suffix='{}'",
                    db_name, db_name_base, use_rfm_preprocessing, suffix
                );
                eprintln!(
                    "[sampler] embedding_model='{}'  d_text={}",
                    embedding_model, d_text
                );
                for (label, p) in [
                    ("nodes", &nodes_path),
                    ("offsets", &offsets_path),
                    ("p2f_adj", &p2f_adj_path),
                    ("text_emb", &text_path),
                ] {
                    let exists = p.exists();
                    let size = std::fs::metadata(p).map(|m| m.len()).ok();
                    eprintln!(
                        "  - {:9}  {}  exists={}  size={:?}",
                        label,
                        p.display(),
                        exists,
                        size
                    );
                }
            }

            let mmap = mmap_file(&nodes_path, "nodes")?;
            let text_mmap = mmap_file(&text_path, "text embeddings")?;
            let bytes = read_all(&offsets_path, "offsets")?;

            let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "Failed to parse offsets archive from {}: {}",
                    offsets_path.display(),
                    err
                ))
            })?;
            let offsets = rkyv::deserialize::<Offsets, Error>(archived).map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "Failed to deserialize offsets from {}: {}",
                    offsets_path.display(),
                    err
                ))
            })?;
            let offsets = offsets.offsets;

            let p2f_adj_mmap = mmap_file(&p2f_adj_path, "p2f adjacency")?;
            let target = target_columns[i];

            datasets.push(Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                offsets,
                col_name_idx_remap,
                rfm_preprocessed: use_rfm_preprocessing,
            });

            let mut matched_count = 0;
            let trace_io = std::env::var("RT_TRACE_IO").ok().as_deref() == Some("1");
            for j in node_idx_offset..node_idx_offset + num_nodes {
                let node = get_node(&datasets[i], j);
                // skip the node if target column was removed during preprocessing
                let has_target = node.col_name_idxs.iter().any(|&c| {
                    let idx: i32 = c.into();
                    datasets[i]
                        .col_name_idx_remap
                        .get(&idx)
                        .copied()
                        .unwrap_or(idx)
                        == target
                });
                if has_target {
                    items.push(Item {
                        dataset_idx: i as i32,
                        node_idx: j,
                    });
                    matched_count += 1;
                }
                if trace_io && j == node_idx_offset {
                    // Log first node for debugging
                    eprintln!(
                        "[sampler] First node j={} has {} columns, target={}, col_name_idxs={:?}",
                        j,
                        node.col_name_idxs.len(),
                        target,
                        &node.col_name_idxs.iter().map(|&c| c.into()).collect::<Vec<i32>>()[..std::cmp::min(10, node.col_name_idxs.len())]
                    );
                }
            }
            if trace_io {
                eprintln!(
                    "[sampler] Task {} ({}): target_col={}, nodes checked={}, nodes with target={}",
                    i, db_name, target, num_nodes, matched_count
                );
            }
        }

        let epoch = 0;
        Ok(Self {
            batch_size,
            seq_len,
            rank,
            world_size,
            datasets,
            items,
            max_bfs_width,
            epoch,
            d_text,
            d_shallow,
            d_shallow_c,
            seed,
            target_columns,
            columns_to_drop,
        })
    }

    fn len_py(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn batch_py<'a>(&self, py: Python<'a>, batch_idx: usize) -> PyResult<Vec<PyObject>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    fn set_feature_dims(&mut self, d_shallow: usize, d_shallow_c: usize) -> PyResult<()> {
        self.d_shallow = d_shallow.max(1);
        self.d_shallow_c = d_shallow_c.max(1);
        Ok(())
    }

    fn feature_dims_py(&self) -> PyResult<(usize, usize, usize)> {
        Ok((self.d_text, self.d_shallow, self.d_shallow_c))
    }

    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));
        self.items.shuffle(&mut rng);
    }
}

impl Sampler {
    fn len(&self) -> usize {
        self.items.len().div_ceil(self.batch_size * self.world_size)
    }

    fn batch(&self, batch_idx: usize) -> Vecs {
        let true_batch_size = self.batch_size.min(
            self.items.len()
                - self.rank * self.batch_size
                - batch_idx * self.batch_size * self.world_size,
        );

        let mut vecs = Vecs::new(
            self.batch_size,
            self.seq_len,
            true_batch_size,
            self.d_text,
            self.d_shallow,
            self.d_shallow_c,
        );
        vecs.chunks_exact_mut(self.seq_len, self.d_text, self.d_shallow, self.d_shallow_c)
            .enumerate()
            .for_each(|(i, slices)| {
                let j =
                    batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size + i;
                // when self.batch_size > true_batch_size, this will wrap around
                let j = j % self.items.len();
                let item = &self.items[j];
                self.seq(item, slices);
            });
        vecs
    }

    fn seq(&self, item: &Item, slices: Slices) {
        let dataset = &self.datasets[item.dataset_idx as usize];
        let target_column = self.target_columns[item.dataset_idx as usize];
        //define let columns to drop which is a vector of i32 and is at the same index as target_columns
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];
        let seed_node_idx = item.node_idx;

        let mut visited = vec![false; dataset.offsets.len() - 1];

        let mut f2p_ftr = vec![(0, seed_node_idx)];
        let seed_node = get_node(dataset, seed_node_idx);
        let mut p2f_ftr = Vec::<Vec<_>>::new();

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch
                .wrapping_add(seed_node_idx as u64)
                .wrapping_add(self.seed),
        );
        loop {
            // select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for (i, node) in p2f_ftr.iter().enumerate() {
                    if !node.is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    return;
                } else {
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    p2f_ftr[depth].swap(r, l - 1);
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            if visited[node_idx as usize] {
                continue;
            }
            visited[node_idx as usize] = true;

            let node = get_node(dataset, node_idx);

            for edge in node.f2p_edges.iter() {
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            let p2f_edges = get_p2f_edges(dataset, node_idx);

            // temporary storage for db edges to be subsampled
            let mut db_p2f_ftr: Vec<i32> = Vec::new();

            for edge in p2f_edges.iter() {
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != seed_node.table_name_idx
                    && edge.table_type != ArchivedTableType::Db
                {
                    continue;
                }

                // temporal constraint
                if edge.timestamp.is_some()
                    && seed_node.timestamp.is_some()
                    && edge.timestamp > seed_node.timestamp
                {
                    continue;
                }

                if edge.table_type == ArchivedTableType::Db {
                    db_p2f_ftr.push(edge.node_idx.into());
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            let idxs = if db_p2f_ftr.len() > self.max_bfs_width {
                index::sample(&mut rng, db_p2f_ftr.len(), self.max_bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            for idx in idxs.iter() {
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(db_p2f_ftr[*idx]);
            }

            let num_cells = node.col_name_idxs.len();
            for cell_i in 0..num_cells {
                let raw_col_idx: i32 = node.col_name_idxs[cell_i].into();
                let col_idx = dataset
                    .col_name_idx_remap
                    .get(&raw_col_idx)
                    .copied()
                    .unwrap_or(raw_col_idx);
                if (node.node_idx == seed_node_idx && columns_to_drop.contains(&col_idx))
                    || (node.timestamp == seed_node.timestamp && columns_to_drop.contains(&col_idx))
                {
                    continue; // do not add this cell to the sequence
                }
                if dataset.rfm_preprocessed
                    && matches!(node.sem_types[cell_i], ArchivedSemType::Text)
                {
                    continue;
                }

                slices.node_idxs[seq_i] = node.node_idx.into();

                assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
                for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
                    slices.f2p_nbr_idxs[seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
                }

                slices.table_name_idxs[seq_i] = node.table_name_idx.into();
                slices.col_name_idxs[seq_i] = col_idx;
                slices.class_value_idxs[seq_i] = node.class_value_idx[cell_i].into();
                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(dataset, col_idx, self.d_text));

                let text_slice = &mut slices.text_values
                    [seq_i * self.d_text..(seq_i + 1) * self.d_text];
                text_slice.fill(bf16::ZERO);
                let shallow_slice = &mut slices.shallow_values
                    [seq_i * self.d_shallow..(seq_i + 1) * self.d_shallow];
                shallow_slice.fill(bf16::ZERO);
                let shallow_c_slice = &mut slices.shallow_c_values
                    [seq_i * self.d_shallow_c..(seq_i + 1) * self.d_shallow_c];
                shallow_c_slice.fill(bf16::ZERO);

                let sem_type = &node.sem_types[cell_i];
                let sem_code = match sem_type {
                    ArchivedSemType::Number => 0,
                    ArchivedSemType::Text => {
                        let text_idx: i32 = node.text_values[cell_i].into();
                        text_slice.copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));
                        1
                    }
                    ArchivedSemType::DateTime => 2,
                    ArchivedSemType::Boolean => 3,
                    ArchivedSemType::Shallow(k) => {
                        let k = *k as usize;
                        assert!(
                            k <= self.d_shallow,
                            "shallow width {} exceeds buffer {}",
                            k,
                            self.d_shallow
                        );
                        let offset: u32 = node.shallow_offsets[cell_i].into();
                        let offset = offset as usize;
                        let storage = node.shallow_storage.as_slice();
                        let end = offset + k;
                        if end <= storage.len() {
                            let dst_slice = &mut shallow_slice[..k];
                            let src_slice = &storage[offset..end];
                            for (dst, src) in dst_slice.iter_mut().zip(src_slice.iter()) {
                                *dst = bf16::from_f32((*src).into());
                            }
                        }
                        4
                    }
                    ArchivedSemType::ShallowCategorical(k) => {
                        let k = *k as usize;
                        assert!(
                            k <= self.d_shallow_c,
                            "shallow_c width {} exceeds buffer {}",
                            k,
                            self.d_shallow_c
                        );
                        let offset: u32 = node.shallow_offsets[cell_i].into();
                        let offset = offset as usize;
                        let storage = node.shallow_storage.as_slice();
                        let end = offset + k;
                        if end <= storage.len() {
                            let dst_slice = &mut shallow_c_slice[..k];
                            let src_slice = &storage[offset..end];
                            for (dst, src) in dst_slice.iter_mut().zip(src_slice.iter()) {
                                *dst = bf16::from_f32((*src).into());
                            }
                        }
                        5
                    }
                };
                slices.sem_types[seq_i] = sem_code;

                slices.number_values[seq_i] = bf16::from_f32(node.number_values[cell_i].into());

                slices.datetime_values[seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());

                slices.boolean_values[seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

                let is_target = seed_node_idx == node.node_idx && col_idx == target_column;
                slices.is_targets[seq_i] = is_target;

                slices.masks[seq_i] = is_target;

                slices.is_task_nodes[seq_i] = node.is_task_node || (col_idx == target_column);
                slices.is_padding[seq_i] = false;

                seq_i += 1;
                if seq_i >= self.seq_len {
                    break;
                }
            }
            if seq_i >= self.seq_len {
                break;
            }
        }
    }
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.p2f_adj_mmap[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(default_value = "128")]
    batch_size: usize,
    #[arg(default_value = "1024")]
    seq_len: usize,
    #[arg(default_value = "1000")]
    num_trials: usize,
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let mut sampler = Sampler::new(
        vec![(cli.db_name, 0, 10)], // dataset_tuples
        cli.batch_size,             // batch_size
        cli.seq_len,                // seq_len
        0,                          // rank
        1,                          // world_size
        256,                        // max_bfs_width
        "all-MiniLM-L12-v2",        // embedding_model
        384,                        // d_text
        0,                          // seed
        vec![-1; 1],                // target_columns
        vec![Vec::<i32>::new()],    // columns_to_drop
    )
    .expect("failed to create sampler");
    sampler.set_feature_dims(3, 4).expect("failed to set feature dims");
    println!("Sampler loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let mut rng = rand::rng();
    for _ in 0..cli.num_trials {
        let tic = Instant::now();
        let batch_idx = rng.random_range(0..sampler.len());
        let _batch = sampler.batch(batch_idx);
        let elapsed = tic.elapsed().as_millis();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / cli.num_trials as f64;
    let std = (sum_sq as f64 / cli.num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} ms,\tStd: {} ms", mean, std);
}
