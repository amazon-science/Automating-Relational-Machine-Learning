use anyhow::Result;
use ndarray::Array2;

use crate::text_encoder::TextEncoder;

pub fn encode_unique_strings(
    encoder: &TextEncoder,
    all_texts: &[String],
    batch: usize,
) -> Result<Array2<f32>> {
    let n = all_texts.len();
    if n == 0 {
        return Ok(Array2::<f32>::zeros((0, 0)));
    }

    let mut out: Option<Array2<f32>> = None;
    let mut s = 0usize;
    while s < n {
        let e = (s + batch).min(n);
        let chunk = &all_texts[s..e];
        let z = encoder.encode_batch(chunk)?;
        if out.is_none() {
            out = Some(Array2::<f32>::zeros((n, z.ncols())));
        }
        if let Some(ref mut arr) = out {
            arr.slice_mut(ndarray::s![s..e, ..]).assign(&z);
        }
        s = e;
    }

    Ok(out.expect("encoder produced no output"))
}
