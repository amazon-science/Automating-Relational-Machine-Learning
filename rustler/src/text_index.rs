use std::collections::{HashMap, HashSet};

use crate::common::TableType;
use crate::pre::TableMap;
use anyhow::Result;
use polars::prelude::DataType;

pub type TextId = i32;
pub type RowTextIds = HashMap<(String, TableType, String), Vec<Option<TextId>>>;

pub fn collect_text_ids_for_text_columns(
    table_map: &TableMap,
    semantic_types: Option<&HashMap<String, HashMap<String, String>>>,
) -> Result<(HashMap<String, TextId>, Vec<String>, RowTextIds)> {
    let mut text_cols_per_table: HashMap<(String, TableType), HashSet<String>> = HashMap::new();

    for ((table_name, table_type), table) in table_map.iter() {
        let semantic_lookup = semantic_types
            .and_then(|m| m.get(table_name))
            .unwrap_or(&table.semantic_types);

        for col in table.df.get_column_names() {
            let col_str = col.as_str();
            let sem = semantic_lookup
                .get(col_str)
                .or_else(|| table.semantic_types.get(col_str));
            let sem_is_text = match sem.map(|s| s.as_str()) {
                Some("text_embedded") | Some("shallow") => true,
                Some("timestamp") => false,
                Some("shallow_c") | Some("categorical") => false,
                Some("unknown") | None => table
                    .df
                    .column(col_str)
                    .map(|series| matches!(series.dtype(), DataType::String))
                    .unwrap_or(false),
                Some(_) => false,
            };
            if sem_is_text {
                text_cols_per_table
                    .entry((table_name.clone(), table_type.clone()))
                    .or_default()
                    .insert(col_str.to_string());
            }
        }
    }

    let mut text_to_idx: HashMap<String, TextId> = HashMap::new();

    for ((table_name, table_type), table) in table_map.iter() {
        let key = (table_name.clone(), table_type.clone());
        if let Some(cols) = text_cols_per_table.get(&key) {
            for col in cols {
                if let Ok(series) = table.df.column(col) {
                    if let Ok(string_chunked) = series.str() {
                        for v in string_chunked.into_iter().flatten() {
                            if !text_to_idx.contains_key(v) {
                                let idx = text_to_idx.len() as TextId;
                                text_to_idx.insert(v.to_string(), idx);
                            }
                        }
                    }
                }
            }
        }
    }

    // Include table names and column labels so downstream metadata lookups stay stable.
    for ((_tname, _ttype), table) in table_map.iter() {
        if !text_to_idx.contains_key(&table.table_name) {
            let idx = text_to_idx.len() as TextId;
            text_to_idx.insert(table.table_name.clone(), idx);
        }
        for col in table.df.get_columns() {
            let override_key = col.name().to_string();
            let base = table
                .col_name_overrides
                .get(&override_key)
                .cloned()
                .unwrap_or_else(|| override_key.clone());
            let label = format!("{} of {}", base, table.table_name);
            if text_to_idx.get(&label).is_none() {
                let idx = text_to_idx.len() as TextId;
                text_to_idx.insert(label, idx);
            }
        }
    }

    let mut text_vec = vec![String::new(); text_to_idx.len()];
    for (s, idx) in &text_to_idx {
        text_vec[*idx as usize] = s.clone();
    }

    let mut row_text_ids: RowTextIds = HashMap::new();
    for ((table_name, table_type), table) in table_map.iter() {
        let key = (table_name.clone(), table_type.clone());
        if let Some(cols) = text_cols_per_table.get(&key) {
            for col in cols {
                if let Ok(series) = table.df.column(col) {
                    if let Ok(string_chunked) = series.str() {
                        let mut ids = Vec::with_capacity(series.len());
                        for v in string_chunked.into_iter() {
                            if let Some(s) = v {
                                let idx = *text_to_idx
                                    .get(s)
                                    .expect("string missing from vocabulary");
                                ids.push(Some(idx));
                            } else {
                                ids.push(None);
                            }
                        }
                        row_text_ids
                            .insert((table_name.clone(), table_type.clone(), col.clone()), ids);
                    }
                }
            }
        }
    }

    Ok((text_to_idx, text_vec, row_text_ids))
}
