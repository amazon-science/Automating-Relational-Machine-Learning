use anyhow::{anyhow, Result};
use ndarray::{s, Array2};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

pub struct GpuIpca {
    ipca: Py<PyAny>,
    cp: Py<PyModule>,
}

impl GpuIpca {
    pub fn new(n_components: usize, batch_size: Option<usize>, whiten: bool) -> Result<Self> {
        Python::with_gil(|py| -> Result<Self> {
            let cuml = py.import("cuml.decomposition")?;
            let cp = py.import("cupy")?;
            let cls = cuml.getattr("IncrementalPCA")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("n_components", n_components)?;
            kwargs.set_item("whiten", whiten)?;
            if let Some(bs) = batch_size {
                kwargs.set_item("batch_size", bs)?;
            }
            let ipca = cls.call((), Some(&kwargs))?;
            Ok(Self {
                ipca: ipca.unbind(),
                cp: cp.unbind(),
            })
        })
    }

    fn to_gpu<'py>(&self, py: Python<'py>, x_cpu: &Array2<f32>) -> PyResult<Py<PyAny>> {
        let cp = self.cp.bind(py);
        let x_np = PyArray2::from_owned_array(py, x_cpu.to_owned());
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", cp.getattr("float32")?)?;
        let arr = cp.call_method("asarray", (x_np.as_any(),), Some(&kwargs))?;
        Ok(arr.unbind())
    }

    fn l2_norm_rows<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let cp = self.cp.bind(py);
        let linalg = cp.getattr("linalg")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("axis", 1)?;
        kwargs.set_item("keepdims", true)?;
        let norms = linalg.call_method("norm", (x,), Some(&kwargs))?;
        let epsilon = cp.call_method1("array", (1e-12f32,))?;
        let norms_eps = cp.call_method1("add", (norms.as_any(), epsilon.as_any()))?;
        let result = cp.call_method1("divide", (x.as_any(), norms_eps.as_any()))?;
        Ok(result.unbind())
    }

    pub fn partial_fit(&self, batch: &Array2<f32>, l2_normalize_rows: bool) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        Python::with_gil(|py| -> Result<()> {
            let ipca = self.ipca.bind(py);
            let x = self.to_gpu(py, batch)?;
            let x = if l2_normalize_rows {
                let bound = x.bind(py);
                self.l2_norm_rows(py, &bound)?
            } else {
                x
            };
            let bound = x.bind(py);
            ipca.call_method1("partial_fit", (bound.as_any(),))?;
            Ok(())
        })
    }

    pub fn transform_to_cpu(
        &self,
        batch: &Array2<f32>,
        l2_normalize_rows: bool,
    ) -> Result<Array2<f32>> {
        if batch.is_empty() {
            return Ok(Array2::<f32>::zeros((0, 0)));
        }
        Python::with_gil(|py| -> Result<Array2<f32>> {
            let ipca = self.ipca.bind(py);
            let cp = self.cp.bind(py);
            let x = self.to_gpu(py, batch)?;
            let x = if l2_normalize_rows {
                let bound = x.bind(py);
                self.l2_norm_rows(py, &bound)?
            } else {
                x
            };
            let z = {
                let bound = x.bind(py);
                ipca.call_method1("transform", (bound.as_any(),))?
            };
            let z_np = cp.call_method1("asnumpy", (z.as_any(),))?;
            if let Ok(arr) = z_np.cast::<PyArray2<f32>>() {
                Ok(arr.to_owned_array())
            } else {
                let casted = z_np.call_method1("astype", ("float32",))?;
                let casted_arr = casted
                    .cast::<PyArray2<f32>>()
                    .map_err(|_| anyhow!("transform() did not return float32 ndarray"))?;
                Ok(casted_arr.to_owned_array())
            }
        })
    }
}

pub fn for_each_row_batch<F>(a: &Array2<f32>, max_rows: usize, mut f: F) -> Result<()>
where
    F: FnMut(&Array2<f32>) -> Result<()>,
{
    if a.is_empty() {
        return Ok(());
    }
    let bsz = if max_rows == 0 { a.nrows() } else { max_rows.min(a.nrows()) };
    let mut s = 0usize;
    while s < a.nrows() {
        let e = (s + bsz).min(a.nrows());
        f(&a.slice(s![s..e, ..]).to_owned())?;
        s = e;
    }
    Ok(())
}
