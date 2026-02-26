use anyhow::{anyhow, Result};
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub struct TextEncoder {
    model: Py<PyAny>,
    device: String,
    batch_size: usize,
    normalize: bool,
    max_length: usize,
}

impl TextEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: &str,
        device: Option<&str>,
        batch_size: usize,
        normalize: bool,
        max_length: usize,
    ) -> Result<Self> {
        Python::with_gil(|py| -> Result<Self> {
            let torch = py.import("torch")?;
            let st = py.import("sentence_transformers")?;
            let st_cls = st.getattr("SentenceTransformer")?;

            let resolved_device = match device {
                Some("auto") | None => {
                    let cuda_available = torch
                        .getattr("cuda")?
                        .getattr("is_available")?
                        .call0()?
                        .is_truthy()?;
                    if cuda_available {
                        "cuda".to_string()
                    } else {
                        "cpu".to_string()
                    }
                }
                Some(d) => d.to_string(),
            };

            let model_obj = st_cls.call1((model_name,))?;
            model_obj.call_method1("to", (resolved_device.as_str(),))?;
            model_obj.setattr("max_seq_length", max_length)?;

            Ok(Self {
                model: model_obj.unbind(),
                device: resolved_device,
                batch_size,
                normalize,
                max_length,
            })
        })
    }

    pub fn encode_batch(&self, texts: &[String]) -> Result<Array2<f32>> {
        if texts.is_empty() {
            return Ok(Array2::<f32>::zeros((0, 0)));
        }

        Python::with_gil(|py| -> Result<Array2<f32>> {
            let model = self.model.bind(py);
            let items: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let py_list = PyList::new(py, &items)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", self.batch_size)?;
            kwargs.set_item("show_progress_bar", false)?;
            kwargs.set_item("convert_to_numpy", true)?;
            kwargs.set_item("device", self.device.clone())?;
            kwargs.set_item("normalize_embeddings", self.normalize)?;

            let encoded = model.call_method("encode", (py_list.as_any(),), Some(&kwargs))?;
            let out = encoded
                .cast::<PyArray2<f32>>()
                .map_err(|_| anyhow!("encode() did not return float32 ndarray"))?;

            Ok(out.to_owned_array())
        })
    }
}
