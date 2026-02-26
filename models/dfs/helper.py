# batched_ipca_cuml.py
import numpy as np
try:
    import cupy as cp
    from cuml.decomposition import IncrementalPCA
except ImportError:
    cp = None
    IncrementalPCA = None

def gpu_ipca_two_pass(
    make_batches,                # () -> iterator of batches (NumPy or CuPy), each [B, D]
    n_components: int = 4,
    *,
    batch_size: int | None = None,   # cuML IPCA internal batch; can leave None
    whiten: bool = False,
    l2_normalize_rows: bool = False, # set True for cosine-style PCA on text embeddings
    dtype=None,
    out_dtype=np.float32,
    n_samples: int | None = None,     # if known, we preallocate output for speed,
    return_model: bool = False
):
    """
    Returns (ipca_model, Z) where Z is [N, n_components] on CPU (NumPy).
    make_batches() must return a *fresh* iterator each time (two passes).
    """
    if cp is None or IncrementalPCA is None:
        raise ImportError("cupy and cuml are required for gpu_ipca_two_pass")
    if dtype is None:
        dtype = cp.float32
    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size,
        whiten=whiten
    )

    # ---------- Pass 1: fit ----------
    for X in make_batches():
        Xg = cp.asarray(X, dtype=dtype)  # move to GPU if needed
        if l2_normalize_rows:
            norms = cp.linalg.norm(Xg, axis=1, keepdims=True) + 1e-12
            Xg = Xg / norms
        ipca.partial_fit(Xg)

    # ---------- Pass 2: transform ----------
    # Optionally preallocate (CPU) to avoid big GPU accumulation
    if n_samples is not None:
        Z_all = np.empty((n_samples, n_components), dtype=out_dtype)
        start = 0
        for X in make_batches():
            Xg = cp.asarray(X, dtype=dtype)
            if l2_normalize_rows:
                norms = cp.linalg.norm(Xg, axis=1, keepdims=True) + 1e-12
                Xg = Xg / norms
            Zb = ipca.transform(Xg)               # on GPU
            Zb = cp.asnumpy(Zb).astype(out_dtype) # back to CPU
            end = start + Zb.shape[0]
            Z_all[start:end] = Zb
            start = end
    else:
        chunks = []
        for X in make_batches():
            Xg = cp.asarray(X, dtype=dtype)
            if l2_normalize_rows:
                norms = cp.linalg.norm(Xg, axis=1, keepdims=True) + 1e-12
                Xg = Xg / norms
            Zb = ipca.transform(Xg)
            chunks.append(cp.asnumpy(Zb).astype(out_dtype))
        Z_all = np.concatenate(chunks, axis=0)

    if return_model:
        return ipca, Z_all
    else:
        return Z_all


