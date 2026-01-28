+++
title = "Running Mojo’s GPU Benchmarks on a Consumer Apple Silicon MacBook"
date = 2026-01-28
description = "Notes from trying to reproduce the Mojo-workloads GPU benchmarks (from the Mojo HPC paper) on an M1 Pro MacBook using Pixi and Metal."
draft = true
tags = ["mojo", "gpu", "apple-silicon", "hpc", "benchmarking"]
+++

Over the last few days I’ve been trying to reproduce the Mojo GPU benchmarks from the *Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem* paper on a fairly ordinary Apple Silicon laptop: an M1 Pro MacBook with 16 GB of unified memory.

The original results in the paper target serious GPUs (think H100 / MI300A), and the companion repo, [`tdehoff/Mojo-workloads`](https://github.com/tdehoff/Mojo-workloads), is written against earlier Mojo APIs. My goal here isn’t to “beat” those numbers, but to see:

- how far I can get on a consumer MacBook,
- what it takes to get the code compiling on **Mojo 0.26.x**, and
- how the experience feels when you stick to Pixi and the Metal backend.

This post is a running log of that process, focused on the **seven‑point stencil** so far, with some notes about **BabelStream**, **miniBUDE**, and **Hartree–Fock** where I’ve hit more fundamental roadblocks.

---

## Setup: Pixi, channels, and Mojo 0.26.x

I’m treating this as a proper, Pixi-managed project rather than a loose collection of scripts.

Key pieces in `pixi.toml`:

```toml
[workspace]
name = "Mojo-workloads"
version = "0.1.0"
authors = ["Michael Booth <michael@databooth.com.au>"]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
platforms = ["osx-arm64"]

[tasks]
test-gpu     = "mojo src/test_gpu.mojo"
stencil-bench = "mojo 7-point-stencil/Mojo/laplacian.mojo"
babel-bench   = "mojo babelStream/Mojo/babelStream.mojo"
bude-bench    = "mojo miniBUDE/Mojo/miniBUDE.mojo"
hf-bench      = "mojo hartree-fock/Mojo/hartree-fock.mojo"

[dependencies]
mojo = ">=0.26.2.0.dev2026012705,<0.27"
python = "3.11.*"
```

A few things worth calling out:

- I pin Mojo to a **specific nightly band**. The repo uses GPU APIs that have changed a few times; pinning avoids chasing moving targets mid‑port.
- I pull Mojo from Modular’s `max-nightly` channel, and everything else from `conda-forge`.
- I use Pixi tasks for all the benchmarks, so `pixi run stencil-bench` *is* the canonical way to drive the seven‑point stencil, etc.
- The original `7-point-stencil/Mojo/pixi.toml` still declares a low-level `modular` dependency (`modular = ">=25.5.0.dev2025070105,<26"`) for the paper’s toolchain; I leave that file in place for provenance, but all my day-to-day runs go through the top-level Pixi workspace.

I’ve also added a small `mojo_bulk_replacements.toml` with mechanical API migrations that apply across all benchmarks, e.g.:

- `from sys import sizeof` → `from sys.info import size_of`
- `sizeof[...]` → `size_of[...]`
- `from gpu.index import block_dim, block_idx, thread_idx` → `from gpu import thread_idx, block_idx, block_dim`

That one file is doing a lot of boring but essential work to bring the original code up to Mojo 0.26.x.

---

## Seven‑point stencil: first benchmark over the line

The seven‑point stencil lives under `7-point-stencil/Mojo/laplacian.mojo`. Conceptually, it’s the same 3D finite‑difference Laplacian from the AMD lab notes: initialise a test function on a 3D grid, then run a 7‑point stencil repeatedly and report an effective memory bandwidth.

The structure of the Mojo version is:

- Layouts via `Layout` / `LayoutTensor`
- One kernel to initialise the test function
- One kernel to apply the Laplacian
- A warmup call, then a timed loop over `num_iter` launches

The code already uses the modern `layout` and `LayoutTensor` APIs, and the GPU launch side is via `DeviceContext.enqueue_function_unchecked[...]`, which still works in 0.26.x.

### Fixing a timing type mismatch

The first problem I hit wasn’t GPU‑specific at all; it was a type mismatch in the timing:

```mojo
comptime L = 512
comptime num_iter = 1000
...

# Timing:
total_elapsed: Float64 = 0.0

for _ in range(num_iter):
    start = monotonic()
    ctx.enqueue_function_unchecked[laplacian_kernel](...)
    ctx.synchronize()
    end = monotonic()

    elapsed = end - start
    bw_gbs: Float64 = Float64(datasize) / Float64(elapsed)
    ...
    total_elapsed += elapsed  # <- type mismatch here
```

On this Mojo build, `monotonic()` returns a tick count that behaves like an unsigned integer type (`UInt`), not a float. `total_elapsed` is `Float64`, so the compiler quite correctly refused to do `Float64 += UInt`.

The fix was boring but important:

```mojo
elapsed = end - start
bw_gbs: Float64 = Float64(datasize) / Float64(elapsed)
...
total_elapsed += Float64(elapsed)
```

This is the first recurring pattern in this port:

> **Timing deltas are not floats, so always cast them explicitly before mixing with `Float32` / `Float64`.**

With that change, `pixi run stencil-bench` finally compiled and ran.

### Running the stencil on an M1 Pro (Float32)

The current configuration is:

- `L = 512` → a 512³ grid
- `dtype = DType.float32`
- `num_iter = 1000`
- `TBSize = 512` (block dimensions default to `512 x 1 x 1`)

A typical run on my M1 Pro looks like:

```text
------------------------------
L = 512 ; Block dimensions: 512 1 1
GPU: Apple M1 Pro
Driver: 0
Theoretical fetch size (GB): 0.5368...
Theoretical fetch size (GB): 0.5306...
Average kernel time: 11.592559 ms
Effective memory bandwidth: 92.08065277045388 GB/s
```

Important caveats:

- **This is just one problem**: `L = 512`, Float32, 1000 iterations.
- The kernel hasn’t been tuned for Apple Silicon; I’ve left the original block size as‑is.

On the GPU side I’m mostly interested in how `BW_GBs` scales with block size, so I’ve kept the original project’s `7-point-stencil/Mojo/run.sh`, which simply sweeps a list of `(blk_x, blk_y, blk_z)` triples and runs each configuration three times.

On the CPU side I’ve added a small companion benchmark in `benchmarks/stencil_cpu.mojo`, exposed as `pixi run stencil-cpu-bench`. It mirrors the GPU Laplacian’s problem size and datasize calculation (`L = 512`, `num_iter = 1000`, Float32) and writes CSV lines with the same header schema as the GPU run (`backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs`). On my M1 Pro that full configuration takes on the order of a few minutes (~200 seconds), which is fine for a baseline but too slow for tight iteration, so for day-to-day testing I dial `L` and `--iter` back down.

Still, it’s a good sanity check: the stencil runs end‑to‑end on both GPU and CPU, and we get plausible effective bandwidth numbers for each on a laptop.

---

## Trying to switch the stencil to Float64

The paper spends a lot of time talking about double‑precision performance, so a natural question is: “can we just flip this to `Float64` and re‑run?”

The code makes that deceptively easy:

```mojo
comptime precision = Float32
comptime dtype = DType.float32
```

I changed it to:

```mojo
comptime precision = Float64
comptime dtype = DType.float64
```

and ran:

```bash
pixi run stencil-bench
```

Unfortunately, this doesn’t currently work on my setup. The Mojo GPU compiler (via Metal) fails when trying to JIT the double‑precision kernel:

```text
Failed to create compute pipeline state (GPU machine code generation):
Compiler encountered an internal error
To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
```

From my perspective as a user:

- The **Float32** stencil kernel runs fine and reports bandwidth.
- The **Float64** version has all the same structure, but the GPU backend for this exact Mojo+Metal combination hits an internal error before we ever get to runtime.

So for now:

- Float32 is my **working configuration** for the stencil on M1 Pro.
- Float64 looks blocked on the underlying toolchain rather than anything in this particular Mojo source file.

---

## BabelStream: where things start getting hairy

The next benchmark in the repo is **BabelStream**, which implements the usual STREAM‑style operations: Copy, Mul, Add, Triad, Dot.

The original Mojo implementation uses:

- Raw `UnsafePointer[Scalar[dtype]]` arguments in all the kernels, and
- `ctx.enqueue_function[...]` to launch them.

On 0.26.x that runs head‑first into two sets of issues:

1. **Mutability rules on `UnsafePointer`**  
   Assignments like `a[i] = initA` and `sums[block_idx.x] = tb_sum[local_tid]` now produce:

   > expression must be mutable in assignment

   This is the compiler being stricter about what counts as a mutable lvalue when you’re going through pointers and address spaces.

2. **`DeviceContext.enqueue_function` no longer likes these kernel signatures**  
   Every kernel launch comes back with:

   > no matching method in call to `enqueue_function`

   along with notes that the `Ts` parameter expects a `DevicePassable` type, but the function has `UnsafePointer[Float64, origin]` parameters instead.

I’ve fixed the easy, mechanical bits (like migrating `List[Int64](...)` to the new `List` constructor), but I’ve deliberately stopped short of rewriting the kernels, because at this point there are at least two viable approaches:

- **Keep the low‑level style**  
  Stay with `UnsafePointer[...]`, but:
  - update the mutability / address‑space usage to match 0.26.x,
  - and launch via whatever flavour of `enqueue_function_unchecked[...]` is intended for pointer‑based kernels.

- **Refactor to higher‑level device‑passable containers**  
  Follow the pattern from the stencil and Hartree–Fock:
  - change kernel parameters to `LayoutTensor`/buffer types that are explicitly `DevicePassable`,
  - and pass those into `ctx.enqueue_function[...]`.

I’ve logged those options (and the fact that BabelStream *does not* currently compile) in `CHANGE_FOR_MOJO_0.26.x.md`, but I haven’t committed to either design yet.

---

## miniBUDE and Hartree–Fock: pointers, atomics, and more modern Mojo

The story for **miniBUDE** and **Hartree–Fock** is similar but more intense:

- Both use **pointer‑heavy kernels** that talk directly to device memory.
- Hartree–Fock in particular leans on **atomics**, which are always a bit fraught on GPUs, and doubly so when you’re crossing multiple Mojo versions and a relatively new Metal backend.

The good news is that the high‑level structure (host‑side setup, use of `LayoutTensor` on the device, etc.) is broadly similar to the stencil benchmark. The pain is all in the details:

- How do you correctly express **mutable views of device memory** in 0.26.x?
- What’s the sanctioned way to launch kernels that still look pointer‑centric?
- To what extent are **atomics and double precision** actually supported (or performant) on the combination of Mojo 0.26.x + Apple’s Metal drivers today?

These are questions for proper documentation and examples, not something I want to guess my way through.

For now, those benchmarks are in the “documented but not ported” bucket.

---

## CHANGE_FOR_MOJO_0.26.x.md: keeping track of the churn

Because this is all happening on a nightly toolchain, I’ve started keeping a living document at the repo root:

- `CHANGE_FOR_MOJO_0.26.x.md`

It captures:

- The Pixi and channel setup,
- The mechanical API migrations (e.g. `sys.sizeof` → `sys.info.size_of`, `gpu.index` → flat `gpu` imports),
- The fixes we’ve already applied (like the stencil timing cast and the `List` constructor change in BabelStream),
- And the open design questions (like “which way should we modernise the BabelStream kernels?”).

It’s not meant to be polished prose; it’s a memory aid that explains *why* each code change was made and what that implies for other benchmarks.

---

## Where this leaves us (for now)

So far, on a consumer M1 Pro MacBook:

- **Seven‑point stencil**
  - **Float32**: compiles and runs via Metal, reports a plausible effective bandwidth (~92 GB/s) for `L = 512` with minimal tuning.
  - **Float64**: same code shape, but the GPU JIT currently hits an internal error when generating machine code.

- **BabelStream**
  - Still blocked on pointer mutability rules and `enqueue_function` expecting `DevicePassable` types.
  - Requires a non‑trivial refactor or updated idioms from the modern Mojo GPU API.

- **miniBUDE / Hartree–Fock**
  - Conceptually portable, but entangled with pointer arithmetic and atomics.
  - I’m deliberately holding off on speculative rewrites until it’s clearer what the “modern Mojo way” is for these patterns.

From a “consumer HPC” perspective, this is still a promising story:

- It’s already possible to run a non‑trivial 3D stencil on the laptop GPU using Mojo and Metal, with very little host‑side ceremony.
- The friction points are in exactly the places you’d expect: low‑level pointer use, address spaces, atomics, and double‑precision support on a relatively young toolchain.

In future posts, my plan is to:

1. Stabilise a **Float32 stencil baseline** across a range of `L` values and compare achieved GB/s to the M1’s theoretical bandwidth.
2. Decide on a direction for **BabelStream** (pointer‑preserving vs device‑passable refactor) and get at least Copy/Mul/Add/Triad running.
3. Explore how far **miniBUDE** can be pushed on a laptop GPU before atomics and precision become the limiting factor.

If you’re attempting something similar, or have an official pointer to “the correct” 0.26.x idioms for pointer‑heavy GPU kernels, I’d love to hear from you.
