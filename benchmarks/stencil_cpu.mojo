"""CPU-only 7-point stencil benchmark.

This is a simple CPU version of the Laplacian stencil used in
7-point-stencil/Mojo/laplacian.mojo. It runs a single stencil sweep over
an L^3 grid and reports effective memory bandwidth as CSV.

Run via:
  pixi run stencil-cpu-bench -- --L 128 --iter 10
"""

from sys import argv
from time import monotonic
from sys.info import size_of

comptime precision = Float32

fn idx(i: Int, j: Int, k: Int, L: Int) -> Int:
    return (k * L + j) * L + i

fn stencil_cpu_once(L: Int):
    var n = L * L * L
    var u = List[precision](capacity=n)
    var f = List[precision](capacity=n)

    # Simple initial condition: u[i] = i as float
    for t in range(n):
        u.append(precision(t))
        f.append(precision(0.0))

    var hx = 1.0 / (L - 1)
    var hy = hx
    var hz = hx

    var invhx2 = 1.0 / hx / hx
    var invhy2 = 1.0 / hy / hy
    var invhz2 = 1.0 / hz / hz
    var invhxyz2 = -2.0 * (invhx2 + invhy2 + invhz2)

    for k in range(1, L - 1):
        for j in range(1, L - 1):
            for i in range(1, L - 1):
                var c = idx(i, j, k, L)
                var xm = idx(i - 1, j, k, L)
                var xp = idx(i + 1, j, k, L)
                var ym = idx(i, j - 1, k, L)
                var yp = idx(i, j + 1, k, L)
                var zm = idx(i, j, k - 1, L)
                var zp = idx(i, j, k + 1, L)

                f[c] = u[c] * precision(invhxyz2) 
                     + (u[xm] + u[xp]) * precision(invhx2) 
                     + (u[ym] + u[yp]) * precision(invhy2) 
                     + (u[zm] + u[zp]) * precision(invhz2)

def main():
    var L: Int = 512
    var num_iter: Int = 1000
    var csv_output = False

    var args = argv()
    var i = 0
    while i < len(args):
        var arg = args[i]
        if arg == "--L" and i + 1 < len(args):
            L = args[i + 1].__int__()
            i += 1
        elif arg == "--iter" and i + 1 < len(args):
            num_iter = args[i + 1].__int__()
            i += 1
        elif arg == "--csv":
            csv_output = True
        i += 1

    # Bytes moved per iteration: match GPU Laplacian theoretical model
    var nx = L
    var ny = L
    var nz = L
    var theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * size_of[precision]()
    var theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * size_of[precision]()
    var datasize = theoretical_fetch_size + theoretical_write_size

    var total_elapsed: Float64 = 0.0

    if csv_output:
        print("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs")

    for it in range(num_iter):
        var start = monotonic()
        stencil_cpu_once(L)
        var end = monotonic()
        var elapsed = end - start
        total_elapsed += Float64(elapsed)

        var bw_gbs = Float64(datasize) / Float64(elapsed)
        if csv_output:
            print(
                "cpu,", "CPU", ",", "float32", ",", L, ",", 1, ",", 1, ",", 1, ",", bw_gbs
            )

    if not csv_output:
        var avg_ns = total_elapsed / Float64(num_iter)
        var avg_s = avg_ns * 1e-9
        var bw_gbs = Float64(datasize) / (avg_ns)
        print("CPU stencil L =", L)
        print("Avg time per iter:", avg_s, "s")
        print("Effective bandwidth:", bw_gbs, "GB/s")