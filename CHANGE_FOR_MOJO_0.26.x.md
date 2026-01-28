# Mojo-workloads – Modernisation / Fixes Log

Running log of the *types* of changes needed to get the original repo code building and running against the current Mojo toolchain. This should generalise to other directories/projects.

---

## 1. Project / Tooling Configuration

### 1.1 Pixi workspace configuration

- **Channels**
  - Use Modular nightly + conda-forge:
    - `channels = ["https://conda.modular.com/max-nightly", "conda-forge"]`
- **Platform**
  - Target Apple Silicon:
    - `platforms = ["osx-arm64"]`
- **Mojo version pin**
  - Constrain to the nightly toolchain in use:
    - `mojo = ">=0.26.2.0.dev2026012705,<0.27"`
- **Python version**
  - Align to:
    - `python = "3.11.*"`
- **Legacy modular dependency**
  - The original `7-point-stencil/Mojo/pixi.toml` in the upstream repo pins a lower-level toolchain via:
    - `modular = ">=25.5.0.dev2025070105,<26"`
  - I keep that nested Pixi file for historical context, but the Apple Silicon work here is driven from the top-level Pixi workspace and its `mojo` dependency.

### 1.2 Tasks

Expose each workload as a Pixi task:

```toml
[tasks]
test-gpu    = "mojo src/test_gpu.mojo"
stencil-bench = "mojo 7-point-stencil/Mojo/laplacian.mojo"
babel-bench   = "mojo babelStream/Mojo/babelStream.mojo"
bude-bench    = "mojo miniBUDE/Mojo/miniBUDE.mojo"
hf-bench      = "mojo hartree-fock/Mojo/hartree-fock.mojo"
```

**Pattern:** task name → `mojo <relative-path-to-main.mojo>`.

### 1.3 Bulk API migration (`mojo_bulk_replacements.toml`)

Centralised text replacements for API changes:

- `from sys import sizeof` → `from sys.info import size_of`
- `sizeof[` → `size_of[` 
- `from gpu.index import block_dim, block_idx, thread_idx`
  → `from gpu import thread_idx, block_idx, block_dim`

Applied to:

- `7-point-stencil/Mojo`
- `babelStream/Mojo`
- `miniBUDE/Mojo`
- `hartree-fock/Mojo`

**Pattern:** maintain a repo-level “bulk replacements” file to upgrade older Mojo code to newer APIs.

---

## 2. 7-Point Stencil / Laplacian

**Status:** `pixi run stencil-bench` (GPU) and `pixi run stencil-cpu-bench` (CPU) both compile and run successfully on Mojo 0.26.x after the changes below.

### 2.1 Timing and numeric types

**Issue:**

- `total_elapsed: Float64 = 0.0`
- `elapsed = end - start` inferred as an integer-like type (`UInt` from `monotonic()`).
- `total_elapsed += elapsed` fails with a type mismatch.

**Fix pattern:**

- Explicitly cast timing deltas when mixing with floats:

```python
elapsed = end - start
bw_gbs: Float64 = Float64(datasize) / Float64(elapsed)
total_elapsed += Float64(elapsed)
```

**General rule:**

- Treat `monotonic()` deltas as non-float by default.
- Whenever used with `Float32`/`Float64` accumulators or divisions, wrap with explicit `Float32(...)` / `Float64(...)`.

### 2.2 GPU precision and launch configuration

**Context:**

- The original stencil Laplacian targeted double precision; on the Apple Silicon + Metal stack, compiling the `Float64` kernel currently triggers an internal GPU compiler error when creating the compute pipeline state.

**Changes:**

- Standardise the GPU path in `7-point-stencil/Mojo/laplacian.mojo` to:
  - `comptime precision = Float32`
  - `comptime dtype = DType.float32`
- Ensure the initialisation kernel `test_function_kernel` uses a full 3D grid/block configuration consistent with the Laplacian kernel (instead of hard-coding a 1D launch).

**Result:**

- `pixi run stencil-bench -- --csv` runs reliably on the M1 Pro GPU and emits CSV with header:
  - `backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs`

### 2.3 CPU stencil benchmark

**Context:**

- Add a CPU-only baseline that mirrors the Laplacian’s maths and datasize accounting, so GPU and CPU bandwidth numbers can be compared directly.

**New file / task:**

- `benchmarks/stencil_cpu.mojo`
- Pixi task:
  - `stencil-cpu-bench = "mojo benchmarks/stencil_cpu.mojo"`

**Key implementation details:**

- Modern Mojo syntax and semantics:
  - Replace `alias precision = Float32` with `comptime precision = Float32`.
  - Replace legacy `let` declarations with `var` and fix indentation so statements start at the beginning of a line.
  - Use `def main()` so argument parsing via `__int__()` can raise as needed.
- Data model:
  - Allocate `List[precision]` buffers with an explicit `capacity` and fill them via `append`.
  - Compute grid spacings and Laplacian coefficients in Float32, matching the GPU kernel.
  - Use the same theoretical datasize formula as the GPU code for fair bandwidth comparison:
    - `theoretical_fetch_size + theoretical_write_size` based on `(nx, ny, nz)` and `size_of[precision]()`.
- Timing and output:
  - Measure `monotonic()` deltas, accumulate them as `Float64`, and explicitly cast where needed.
  - Default configuration mirrors the GPU run: `L = 512`, `num_iter = 1000`, `precision = Float32`.
  - When `--csv` is passed, print rows with the same header schema as the GPU benchmark:
    - `backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs`
    - For CPU runs this is currently:
      - `backend = "cpu"`, `GPU = "CPU"`, `precision = "float32"`, `blk_x = blk_y = blk_z = 1`.

**Practical note:**

- On an M1 Pro, the full `L = 512`, `num_iter = 1000` CPU configuration takes on the order of minutes (~200 seconds). That’s acceptable for a one-off baseline, but for quick iteration it’s better to reduce `L` and/or `--iter`.

---

## 3. babelStream (identified changes; implementation pending)

### 3.1 Mutability with pointers and shared memory

**Symptoms:**

- Errors like “expression must be mutable in assignment” where writing through:
  - `UnsafePointer[Scalar[dtype]]` (`a[i] = initA`, `b[i] = ...`, etc.).
  - Shared buffers from `stack_allocation[...]()`.
  - Pointer-based buffers such as `sums[block_idx.x] = tb_sum[local_tid]`.

**Change pattern:**

- Ensure that:
  - Pointer parameters are declared **mutable** in their type (or replaced by mutable, higher-level device containers).
  - Shared-memory allocations and their element types are considered mutable.
- In newer Mojo, mutability is explicit and enforced at the type level for pointer-like access.

### 3.2 `DeviceContext.enqueue_function` vs raw pointer kernels

**Symptoms:**

- “no matching method in call to `enqueue_function`”
- Notes about candidate parameter `Ts` being `DevicePassable`, while our kernel types use `UnsafePointer[...]` parameters.

**Change pattern (one of):**

1. **Refactor kernels to use device-passable arguments:**
   - Replace raw pointers (`UnsafePointer[...]`) in kernel signatures with GPU containers expected by the current API (e.g. `Buffer[...]`, `LayoutTensor[...]`, or similar).
   - Launch with `ctx.enqueue_function[...]` using those containers directly.

2. **Or use *_unchecked* APIs for low-level kernels:**
   - If the runtime offers `enqueue_function_unchecked[...]` that accepts pointer-based kernels, switch calls accordingly when we intentionally bypass type checks.

**General rule:**

- Old code that:
  - takes `UnsafePointer` parameters, and
  - is launched via `enqueue_function`
- may need to either:
  - adopt device-safe argument types, or
  - move to an “unchecked” launch API that’s intended for such low-level kernels.

### 3.3 Collections API – `List` constructor

**Symptom:**

- `List[Int64](v1, v2, v3, v4, v5)` fails with:
  - “expected at most 0 positional arguments, got 5”
  - plus missing keyword-only args (`capacity`, `length`, `fill`, `__list_literal__`, etc.).

**Change pattern:**

- Use a 0-arg constructor plus `append` (or whatever new literal form the stdlib documents), e.g.:

```python
var kernel_data = List[Int64]()
kernel_data.append(2 * SIZE * size_of[Scalar[dtype]]())
kernel_data.append(2 * SIZE * size_of[Scalar[dtype]]())
kernel_data.append(3 * SIZE * size_of[Scalar[dtype]]())
kernel_data.append(3 * SIZE * size_of[Scalar[dtype]]())
kernel_data.append(2 * SIZE * size_of[Scalar[dtype]]())
```

**General rule:**

- Do not rely on multi-positional-argument `List[T](...)` constructors.
- Prefer explicit construction and population (`append`, literal helpers, or factory functions) that match the current API.

### 3.4 Timing and Numpy interaction

**Pattern:**

- Timing data stored in a NumPy array:
  - `kernel_timings = np.zeros(Python.tuple(5, num_iter), dtype="float32")`
  - Writes:
    - `kernel_timings[i][k] = Float32(end - start)`

**Rule:**

- Same as in the Laplacian code:
  - Always cast timing deltas from `monotonic()` into the intended float type before storing into numeric arrays/accumulators.

### 3.5 Open design choice (not yet resolved)

We have **not** yet updated `babelStream` to compile on Mojo 0.26.x, because there are two competing approaches:

1. **Low-level pointer-preserving approach**
   - Keep the existing `UnsafePointer[...]`-based kernels.
   - Update them to the new pointer mutability / address-space rules and find the correct way to write through pointers on device.
   - Use the appropriate `DeviceContext` launch API for pointer-based kernels (e.g. an `enqueue_function_unchecked`-style API if recommended for this use case).

2. **Higher-level, device-passable refactor**
   - Rewrite kernels to operate on `DevicePassable` containers (e.g. `LayoutTensor` or other buffer abstractions) instead of raw pointers.
   - Call `ctx.enqueue_function[...]` with those containers, mirroring working patterns in other benchmarks.

At this point, we have only documented the issues and options; no code changes have been applied yet to avoid committing to the wrong pattern without confirming the intended 0.26.x GPU idioms.

---

## 4. General Migration Themes (So Far)

1. **API surface drift:**
   - `sys.sizeof` → `sys.info.size_of`
   - GPU index imports reorganised (`gpu.index` → flat `gpu` imports).
   - `DeviceContext.enqueue_function` now expects `DevicePassable` types rather than arbitrary pointer-based kernels.

2. **Stricter type system / mutability:**
   - Mutability of pointers and shared-memory buffers must be explicit and consistent with writes.
   - Numeric operations across integer-like time deltas and floats require explicit casts.

3. **Standard library changes:**
   - `List` constructor semantics changed; no longer supports “Python-style” multi-element positional construction.

4. **Tooling standardisation:**
   - Pixi workspace config centralises:
     - channels,
     - platforms,
     - Mojo version pin,
     - Python version,
     - standard task names/paths.
   - A single “bulk replacements” config is used to apply systematic mechanical edits across all benchmark directories.
