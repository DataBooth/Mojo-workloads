from sys import has_accelerator, has_apple_gpu_accelerator
from gpu.host import DeviceContext

fn main() raises:
    if not has_accelerator():
        print("No compatible accelerator found; running on CPU only")
        return

    var use_apple_gpu = has_apple_gpu_accelerator()
    var ctx = DeviceContext()  # default backend
    if use_apple_gpu:
        print("Using Apple GPU (Metal)")
        ctx = DeviceContext(api="metal")  # Explicit Metal backend on Apple Silicon
    else:
        print("Using default GPU backend")

    # Report device details via Mojo GPU API
    print("Device API:", ctx.api())
    print("Device name:", ctx.name())

    # If available, also report memory information
    try:
        var (free_bytes, total_bytes) = ctx.get_memory_info()
        print("Device memory (free / total bytes):", free_bytes, "/", total_bytes)
    except:
        # Older Mojo builds may not support get_memory_info(); ignore if unavailable.
        pass
