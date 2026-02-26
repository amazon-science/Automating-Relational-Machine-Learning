use std::collections::{HashMap, HashSet};

use anyhow::Result;
use ndarray::Array2;

use crate::common::TableType;
use crate::gpu_ipca::GpuIpca;
use crate::pre::TableMap;
use crate::text_index::RowTextIds;

#[derive(Debug)]
pub struct ShallowColumn {
    pub k: u8,
    pub offsets: Vec<i32>,
    pub storage: Vec<f32>,
}

pub type ShallowMap = HashMap<(String, TableType, String), ShallowColumn>;

fn cutoff_for_table(
    table_name: &str,
    test_cutoffs: &HashMap<String, i64>,
    table_map: &TableMap,
) -> Option<i64> {
    if let Some(&c) = test_cutoffs.get(table_name) {
        return Some(c);
    }
    if let Some(test_table) = table_map.get(&(table_name.to_string(), TableType::Test)) {
        if let Some(tcol) = &test_table.tcol_name {
            if let Ok(series) = test_table.df.column(tcol) {
                if let Ok(dt) = series.datetime() {
                    let mut best: Option<i64> = None;
                    for idx in 0..dt.len() {
                        if let Some(v) = dt.phys.get(idx) {
                            best = Some(best.map_or(v, |curr| curr.min(v)));
                        }
                    }
                    return best;
                }
            }
        }
    }
    None
}

fn gather_embeddings(e_all: &Array2<f32>, ids: &[i32]) -> Array2<f32> {
    if ids.is_empty() {
        return Array2::<f32>::zeros((0, e_all.ncols()));
    }
    let mut out = Array2::<f32>::zeros((ids.len(), e_all.ncols()));
    for (row_idx, &id) in ids.iter().enumerate() {
        out.row_mut(row_idx).assign(&e_all.row(id as usize));
    }
    out
}

fn fit_split(
    table_name: &str,
    split: TableType,
    col: &str,
    cutoff_ns: Option<i64>,
    ipca: &GpuIpca,
    e_all: &Array2<f32>,
    row_chunk: usize,
    l2norm_rows: bool,
    row_text_ids: &RowTextIds,
    table_map: &TableMap,
) -> Result<bool> {
    let key = (table_name.to_string(), split.clone(), col.to_string());
    let Some(ids) = row_text_ids.get(&key) else {
        return Ok(false);
    };
    let Some(table) = table_map.get(&(table_name.to_string(), split.clone())) else {
        return Ok(false);
    };

    let mut selected: Vec<i32> = Vec::new();
    if let Some(ts_col) = &table.tcol_name {
        let series = table.df.column(ts_col)?;
        if let Some(cutoff) = cutoff_ns {
            let dt = series.datetime()?;
            for row in 0..table.df.height() {
                if let Some(id) = ids[row] {
                    if let Some(ts) = dt.phys.get(row) {
                        if ts < cutoff {
                            selected.push(id);
                        }
                    }
                }
            }
        } else {
            for (row, maybe_id) in ids.iter().enumerate() {
                if let Some(id) = maybe_id {
                    if table
                        .df
                        .column(ts_col)?
                        .datetime()?
                        .phys.get(row)
                        .is_some()
                    {
                        selected.push(*id);
                    }
                }
            }
        }
    } else {
        for maybe_id in ids.iter().flatten() {
            selected.push(*maybe_id);
        }
    }

    if selected.is_empty() {
        return Ok(false);
    }

    let mut start = 0usize;
    while start < selected.len() {
        let end = (start + row_chunk).min(selected.len());
        let xb = gather_embeddings(e_all, &selected[start..end]);
        ipca.partial_fit(&xb, l2norm_rows)?;
        start = end;
    }
    Ok(true)
}

fn flush_transform(
    ipca: &GpuIpca,
    e_all: &Array2<f32>,
    batch_ids: &mut Vec<i32>,
    batch_rows: &mut Vec<usize>,
    storage: &mut Vec<f32>,
    offsets: &mut [i32],
    l2norm_rows: bool,
) -> Result<()> {
    if batch_ids.is_empty() {
        return Ok(());
    }
    let xb = gather_embeddings(e_all, batch_ids);
    let zb = ipca.transform_to_cpu(&xb, l2norm_rows)?;
    for (idx, &row) in batch_rows.iter().enumerate() {
        let start = storage.len() as i32;
        offsets[row] = start;
        storage.extend(zb.row(idx).iter().copied());
    }
    batch_ids.clear();
    batch_rows.clear();
    Ok(())
}

fn transform_split(
    table_name: &str,
    split: TableType,
    col: &str,
    ipca: &GpuIpca,
    e_all: &Array2<f32>,
    row_chunk: usize,
    l2norm_rows: bool,
    row_text_ids: &RowTextIds,
    table_map: &TableMap,
    components: usize,
) -> Result<Option<ShallowColumn>> {
    let key = (table_name.to_string(), split.clone(), col.to_string());
    let Some(ids) = row_text_ids.get(&key) else {
        return Ok(None);
    };
    let Some(table) = table_map.get(&(table_name.to_string(), split.clone())) else {
        return Ok(None);
    };
    let nrows = table.df.height();
    let mut offsets = vec![-1i32; nrows];
    let mut storage: Vec<f32> = Vec::new();
    storage.reserve(ids.len() * components);

    let mut batch_rows: Vec<usize> = Vec::with_capacity(row_chunk);
    let mut batch_ids: Vec<i32> = Vec::with_capacity(row_chunk);

    for (row_idx, maybe_id) in ids.iter().enumerate() {
        if let Some(id) = maybe_id {
            batch_rows.push(row_idx);
            batch_ids.push(*id);
            if batch_ids.len() >= row_chunk {
                flush_transform(
                    ipca,
                    e_all,
                    &mut batch_ids,
                    &mut batch_rows,
                    &mut storage,
                    &mut offsets,
                    l2norm_rows,
                )?;
            }
        }
    }
    flush_transform(
        ipca,
        e_all,
        &mut batch_ids,
        &mut batch_rows,
        &mut storage,
        &mut offsets,
        l2norm_rows,
    )?;

    if offsets.iter().all(|&o| o < 0) {
        return Ok(None);
    }

    Ok(Some(ShallowColumn {
        k: components as u8,
        offsets,
        storage,
    }))
}

pub fn fit_and_transform_shallow(
    table_map: &TableMap,
    row_text_ids: &RowTextIds,
    e_all: &Array2<f32>,
    components: usize,
    l2norm_rows: bool,
    whiten: bool,
    row_chunk: usize,
    pca_batch_size: Option<usize>,
    test_cutoffs: &HashMap<String, i64>,
) -> Result<ShallowMap> {
    let mut shallow_map: ShallowMap = HashMap::new();

    let mut grouped: HashMap<String, HashSet<String>> = HashMap::new();
    for (table_name, _, col) in row_text_ids.keys() {
        grouped
            .entry(table_name.clone())
            .or_default()
            .insert(col.clone());
    }

    for (table_name, cols) in grouped.into_iter() {
        let cutoff_ns = cutoff_for_table(&table_name, test_cutoffs, table_map);
        for col in cols {
            let ipca = GpuIpca::new(components, pca_batch_size, whiten)?;
            let mut fitted = false;

            for split in [TableType::Train, TableType::Val] {
                if fit_split(
                    &table_name,
                    split.clone(),
                    &col,
                    cutoff_ns,
                    &ipca,
                    e_all,
                    row_chunk,
                    l2norm_rows,
                    row_text_ids,
                    table_map,
                )? {
                    fitted = true;
                }
            }

            if !fitted {
                if fit_split(
                    &table_name,
                    TableType::Db,
                    &col,
                    cutoff_ns,
                    &ipca,
                    e_all,
                    row_chunk,
                    l2norm_rows,
                    row_text_ids,
                    table_map,
                )? {
                    fitted = true;
                }
            }

            if !fitted {
                for split in [TableType::Train, TableType::Val, TableType::Db] {
                    if fit_split(
                        &table_name,
                        split.clone(),
                        &col,
                        None,
                        &ipca,
                        e_all,
                        row_chunk,
                        l2norm_rows,
                        row_text_ids,
                        table_map,
                    )? {
                        fitted = true;
                        break;
                    }
                }
            }

            if !fitted {
                continue;
            }

            for split in [
                TableType::Train,
                TableType::Val,
                TableType::Test,
                TableType::Db,
            ] {
                if let Some(column) = transform_split(
                    &table_name,
                    split.clone(),
                    &col,
                    &ipca,
                    e_all,
                    row_chunk,
                    l2norm_rows,
                    row_text_ids,
                    table_map,
                    components,
                )? {
                    shallow_map.insert((table_name.clone(), split, col.clone()), column);
                }
            }
        }
    }

    Ok(shallow_map)
}
