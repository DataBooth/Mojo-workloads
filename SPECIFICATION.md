# Refined Structured Plan for Redoing Mojo Benchmarks on Apple Silicon (Pixi-Integrated)

This document outlines a detailed, step-by-step plan to adapt and rerun the four Mojo benchmarks (seven-point stencil, BabelStream, miniBUDE, and Hartree–Fock) from the GitHub repo on your M1 MacBook with 16 GB unified memory. The focus is on maximizing GPU usage via Mojo's Metal backend (available since late 2025) while providing CPU fallbacks for limitations like atomics and reductions. This gives a "consumer" perspective: smaller problem sizes for laptop-friendly runs, relative GPU vs. CPU speedups, and scaled comparisons to the paper's high-end GPU results (e.g., % of theoretical peak performance on M1 vs. H100/MI300A).

Estimated total time: 4–8 hours (assuming your experience with Python and Mojo). All adaptations are based on Mojo's latest nightly builds (as of January 2026), which include improved Apple Silicon support. Since you'll be using Pixi exclusively for environment management/installs and Pixi tasks for benchmarking runs/jobs, this plan integrates Pixi centrally (e.g., for Mojo updates, env setup, and task-based execution).

## Prerequisites
- **Hardware/Software**: M1 MacBook (16 GB), macOS 15.7.3, Xcode 26.2 – all compatible.
- **Pixi**: Install if not already (via `curl -fsSL https://pixi.sh/install.sh | sh`). Pixi will handle all env management and Mojo installs/updates. Add Mojo from Modular's conda channel as needed.
- **Repo**: Clone if not done: `git clone https://github.com/tdehoff/Mojo-workloads`.
- **Tools**: Use Activity Monitor for GPU/memory monitoring; Mojo's `Timer` or Python interop for timing.
- **General Tips**: Start with small iterations (e.g., 10) for testing; scale to 100+ for stable metrics. Compute bandwidth (GB/s) = bytes transferred / time; GFLOP/s = FLOPs / time. Compare to paper by normalizing to hardware peaks (M1: ~400 GB/s bandwidth, ~1.3 TFLOPS FP64).

## Phase 1: Pixi-Based Environment Setup and Verification (30–45 minutes)
1. **Configure Pixi in this repository**:
   - Ensure you are in the forked repo root (e.g., `Mojo-workloads`). If you have not already initialised Pixi here, run: `pixi init .`.
   - In `pixi.toml`, configure channels to include Modular's nightly channel and conda-forge, for example:
     ```toml
     [workspace]
     channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
     ```
   - Add Mojo (latest nightly) to this project: `pixi add mojo`.
   - Verify: `pixi run mojo --version` (should show a recent nightly build, e.g. `0.26.x.dev...`).

2. **Test GPU detection with a Pixi task**:
   - Add a task to `pixi.toml` for a simple GPU capability probe:
     ```toml
     [tasks]
     test-gpu = "mojo test_gpu.mojo"
     ```
   - Create `test_gpu.mojo` in the project:
     ```mojo
     from sys import has_accelerator, has_apple_gpu_accelerator

     fn main():
         print("Has accelerator:", has_accelerator())
         print("Has Apple GPU:", has_apple_gpu_accelerator())
     ```
   - Run: `pixi run test-gpu` – on an Apple Silicon Mac you should normally see `True` for both.

3. **Define general Pixi tasks for this repo**:
   - Keep the benchmarks in this forked `Mojo-workloads` repository; do not copy or symlink them into a separate Pixi project.
   - Define general tasks in `pixi.toml` for common jobs (e.g., build/run patterns; specific benchmark tasks in Phase 3).

## Phase 2: General Adaptations for All Benchmarks
Apply these modifications to each benchmark's `main.mojo` file. Use Pixi tasks for management (e.g., running benchmarks as jobs).

- **GPU Detection and Context**:
  ```mojo
  from sys import has_accelerator, has_apple_gpu_accelerator
  from gpu.host import DeviceContext

  fn main():
      if not has_accelerator():
          print("No compatible accelerator found; running on CPU only")
          return

      let use_apple_gpu = has_apple_gpu_accelerator()
      if use_apple_gpu:
          print("Using Apple GPU (Metal)")
          let ctx = DeviceContext(api="metal")  # Explicit Metal backend on Apple Silicon
      else:
          print("Using default GPU backend")
          let ctx = DeviceContext()

      # CPU-only fallback paths for benchmarks can mirror the existing CPU implementations
  ```
- **Memory Allocation** (Leverage unified memory):
  ```mojo
  # Example GPU buffers on Apple Silicon (unified memory)
  let host_buf = ctx.enqueue_create_host_buffer[DType.float64](size)
  let dev_buf = ctx.enqueue_create_buffer[DType.float64](size)
  ctx.synchronize()
  ```
- **Kernel Launch** (Smaller blocks for M1 GPU):
  ```mojo
  # Example launch configuration tuned for M1 GPUs
  alias block_size = 128  # Reasonable starting point for M1-class GPUs

  let kernel_obj = ctx.compile_function[kernel_fn, kernel_fn]()
  ctx.enqueue_function(
      kernel_obj,
      args...,
      grid_dim=(ceildiv(N, block_size), 1, 1),
      block_dim=(block_size, 1, 1),
  )
  ctx.synchronize()
  ```
- **Problem Scaling**: Start with small sizes to fit 16 GB (e.g., 128–256 dimensions); increase to ~8–12 GB max usage.
- **Kernel Cleanup**: Remove any `print` statements inside `fn kernel[...]`.
- **CPU Fallback**: For unsupported features, use simple loops (e.g., `for i in range(N): out[i] = in1[i] + in2[i]`).

## Phase 3: Benchmark-Specific Adaptations and Pixi Tasks
For each, modify `main.mojo`, test GPU mode, fallback if needed, and compute metrics. Define Pixi tasks in `pixi.toml` for repeatable jobs (e.g., `pixi run stencil-bench --L=256`). Compare to paper with consumer scaling (e.g., M1 efficiency as fraction of its peak vs. paper's).

1. **Seven-Point Stencil (Memory-Bound, Diffusion Simulation)**:
   - **Feasibility**: High – Pure data-parallel.
   - **Adaptations**: Scale grid to L=256 (~1 GB in FP64). Use block_size=128. CPU fallback: Nested loops for grid updates.
   - **Pixi Task**: Add to `pixi.toml`: `stencil-bench = "mojo seven-point-stencil/main.mojo"`
   - **Run Command**: `pixi run stencil-bench --L=256 --iterations=100`
   - **Expected Metrics**: 200–300 GB/s (50–75% M1 peak) vs. paper's 3–5 TB/s on H100.
   - **Consumer Comparison**: Demonstrates laptop-friendly diffusion sims with good portability.

2. **BabelStream (Memory-Bound, Array Operations: Copy, Mul, Add, Triad, Dot)**:
   - **Feasibility**: High – Element-wise; manual reduction for Dot.
   - **Adaptations**: Size=2^23 (8M elements, ~64 MB). For Dot, implement tree-reduction in kernel if built-in fails. CPU fallback: Loops for array ops.
   - **Pixi Task**: Add to `pixi.toml`: `babel-bench = "mojo babelstream/main.mojo"`
   - **Run Command**: `pixi run babel-bench --size=8388608 --iterations=100`
   - **Expected Metrics**: Triad ~300 GB/s vs. paper's peaks.
   - **Consumer Comparison**: Everyday array ops with noticeable GPU speedup on a MacBook.

3. **miniBUDE (Compute-Bound, Molecular Docking)**:
   - **Feasibility**: Medium – Parallel poses; manual reductions.
   - **Adaptations**: Poses=10,000, poses_per_wi=32. Tree-reduce energies if needed. No fast-math – accept lower perf. CPU fallback: Serial per-pose computation.
   - **Pixi Task**: Add to `pixi.toml`: `bude-bench = "mojo minibude/main.mojo"`
   - **Run Command**: `pixi run bude-bench --poses=10000 --iterations=100`
   - **Expected Metrics**: ~0.5–1 TFLOPS vs. paper's tens of TFLOPS.
   - **Consumer Comparison**: Prototyping docking on consumer hardware, highlighting compute gaps.

4. **Hartree–Fock (Compute-Bound, Quantum Chemistry with Atomics)**:
   - **Feasibility**: Low on GPU – Atomics unsupported.
   - **Adaptations**: Atoms=20, gaussians=100. Use per-thread buffers + post-kernel reduction to avoid atomics. CPU fallback: Primary – Loops for matrix updates.
   - **Pixi Task**: Add to `pixi.toml`: `hf-bench = "mojo hartree-fock/main.mojo"`
   - **Run Command**: `pixi run hf-bench --atoms=20 --iterations=100`
   - **Expected Metrics**: Seconds per run (CPU-dominant) vs. paper's ms.
   - **Consumer Comparison**: Small-molecule quantum chem on a laptop, showing Mojo's ecosystem potential despite limitations.

## Phase 4: Execution, Analysis, and Comparison (1–2 hours)
1. **Run Sequence with Pixi**:
   - For each benchmark: Test GPU/CPU modes via tasks, vary sizes slightly.
   - Log results to CSV: Columns like `benchmark, mode, size, iterations, time_s, gbs_or_gflops`. Add a task for logging if desired (e.g., `log-results = "python log_script.py"`).

2. **Analysis**:
   - **Relative Speedups**: GPU vs. CPU (aim for 5–10x on memory-bound).
   - **Vs. Paper**: Normalize (e.g., M1 stencil: 70% peak efficiency vs. paper's 90% on H100). Compute informal portability metric like average efficiency.
   - **Consumer Insights**: Emphasize ease of running HPC kernels on a MacBook without discrete GPUs – great for development/prototyping, but scale limits for production.

3. **Troubleshooting**:
   - **Common Errors**: Enqueue type mismatch – ensure args match exactly. Reduction fails – manual implement.
   - **Resources**: Modular forums (developer.modular.com), GitHub issues.
   - **Enhancements**: Fork the repo and add your M1 results to the README for community sharing. Use Pixi tasks for automation (e.g., a `all-bench` task chaining all runs).

If issues arise during implementation, note the error and benchmark – we can iterate! This plan positions Mojo as a versatile tool for consumer-grade HPC exploration, with Pixi streamlining management.

## References
- **Paper**: "Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem" – Available at [arXiv:2509.21039v1](https://arxiv.org/html/2509.21039v1).
- **Repository**: Source code for the benchmarks – Available at [github.com/tdehoff/Mojo-workloads](https://github.com/tdehoff/Mojo-workloads).
- **Companion Video**: Presentation or demo related to the paper – Available at [youtube.com/watch?v=HKqeMg9NZ8s](https://www.youtube.com/watch?v=HKqeMg9NZ8s).