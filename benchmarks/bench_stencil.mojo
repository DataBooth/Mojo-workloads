"""BenchSuite wrapper for the 7-point stencil benchmark.

Currently this focuses on GPU timings for a range of L values using the
existing Laplacian implementation in 7-point-stencil/Mojo/laplacian.mojo.

CPU-only variants can be added later once a pure-CPU stencil is factored out.
"""

from benchsuite import BenchResult, run_benchmarks, auto_benchmark
from time import perf_counter
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from sys.info import size_of

# NOTE: This is a trimmed-down, size-parameterised version of the GPU path
# from 7-point-stencil/Mojo/laplacian.mojo, specialised to Float32.

alias precision = Float32
alias dtype = DType.float32

fn gpu_stencil_once(L: Int):
    if not has_accelerator():
        return

    var nx = L
    var ny = L
    var nz = L

    let layout = Layout.row_major(nx, ny, nz)
    var ctx = DeviceContext()

    let total_points = nx * ny * nz
    let d_u = ctx.enqueue_create_buffer[dtype](total_points)
    let d_f = ctx.enqueue_create_buffer[dtype](total_points)

    var u_tensor = LayoutTensor[dtype, layout](d_u)
    var f_tensor = LayoutTensor[dtype, layout](d_f)

    # Simple grid spacing based on nx, ny, nz
    let hx = 1.0 / (nx - 1)
    let hy = 1.0 / (ny - 1)
    let hz = 1.0 / (nz - 1)

    # Initialise u with a simple function on device: here we just zero it via host for now
    # (we're primarily interested in launch cost + memory behaviour).
    with d_u.map_to_host() as h_u:
        for i in range(total_points):
            h_u[i] = precision(0.0)
    ctx.synchronize()

    # Compute Laplacian stencil once on the GPU using a simple kernel inlined here.

    fn laplacian_step(
        f: LayoutTensor[mut=True, dtype, layout],
        u: LayoutTensor[mut=False, dtype, layout],
        nx: Int, ny: Int, nz: Int,
        invhx2: precision, invhy2: precision, invhz2: precision, invhxyz2: precision,
    ):
        var k = thread_idx.x + block_idx.x * block_dim.x
        var j = thread_idx.y + block_idx.y * block_dim.y
        var i = thread_idx.z + block_idx.z * block_dim.z

        if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
            f[i, j, k] = u[i, j, k] * invhxyz2 
                   + (u[i - 1, j    , k    ] + u[i + 1, j    , k    ]) * invhx2
                   + (u[i    , j - 1, k    ] + u[i    , j + 1, k    ]) * invhy2
                   + (u[i    , j    , k - 1] + u[i    , j    , k + 1]) * invhz2

    let invhx2 = 1.0 / hx / hx
    let invhy2 = 1.0 / hy / hy
    let invhz2 = 1.0 / hz / hz
    let invhxyz2 = -2.0 * (invhx2 + invhy2 + invhz2)

    # Choose a simple block/grid configuration similar to the main benchmark
    let BLK_X = 512
    let BLK_Y = 1
    let BLK_Z = 1

    ctx.enqueue_function_unchecked[laplacian_step](
        f_tensor, u_tensor, nx, ny, nz,
        invhx2, invhy2, invhz2, invhxyz2,
        grid_dim = (ceildiv(nx, BLK_X), ceildiv(ny, BLK_Y), ceildiv(nz, BLK_Z)),
        block_dim = (BLK_X, BLK_Y, BLK_Z)
    )
    ctx.synchronize()

fn bench_stencil_gpu_size(L: Int) -> BenchResult:
    return auto_benchmark[gpu_stencil_once]("stencil_gpu_L_" + String(L))

def main():
    # Keep sizes modest to avoid memory pressure but still get scaling data.
    let sizes = [128, 192, 256, 384, 512]
    var results = List[BenchResult]()

    for L in sizes:
        results.append(bench_stencil_gpu_size(L))

    run_benchmarks(results, "stencil_gpu_sizes")