# rustler - context sampler for RT, written in Rust

## File Structure

- `pre.rs` - Code to preprocess datasets
- `fly.rs` - Code for on-the-fly context sampling
- `common.rs` - Data structures exchanged between `pre.rs` and `fly.rs` (using [rkyv](https://github.com/rkyv/rkyv))
- `lib.rs` - Entry point to rustler as a library
- `main.rs` - Entry point to rustler as a standalone executable

## Development Workflow

### After Making Changes to Rust Code

**Important:** You MUST rebuild and reinstall rustler after making any changes to the Rust source code.

#### Option 1: Rebuild and Install (Recommended)
```bash
pixi run rt-build
```
This uses `maturin develop` to:
- Compile the Rust code in release mode
- Build the Python wheel
- Install it as an editable package

#### Option 2: Just Compile (for checking compilation errors)
```bash
pixi run rt-compile
```
This only compiles the code without installing. Useful for checking syntax errors.

### Debugging Tips

- **Enable IO tracing:** Set `RT_TRACE_IO=1` environment variable to see detailed file access logs:
  ```bash
  RT_TRACE_IO=1 pixi run python3 -m swap.relation data.use_rfm_preprocessing=true
  ```

- **Check compiled library location:**
  ```bash
  python3 -c "import rustler; print(rustler.__file__)"
  ```

### Common Issues

**Problem:** Code changes don't take effect
- **Solution:** Run `pixi run rt-build` to rebuild and reinstall

**Problem:** Panic with "No such file or directory" but files exist
- **Solution:** The compiled binary is out of sync. Run `pixi run rt-build`

**Problem:** Module import errors
- **Solution:** Make sure you're in the AutoRDL root directory and using the pixi environment
