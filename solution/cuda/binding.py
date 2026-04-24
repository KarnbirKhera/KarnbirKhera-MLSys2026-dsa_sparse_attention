# Torch binding path — the actual C++ entry point is registered in kernel.cu
# via PYBIND11_MODULE. flashinfer-bench's torch build compiles kernel.cu with
# torch.utils.cpp_extension and reads the entry function name from config.toml.
# This file is kept as a no-op so pack_solution.py's file discovery finds it
# in the expected location. Do NOT use tvm.ffi register_func here — it
# conflicts with config.toml's `binding = "torch"`.
