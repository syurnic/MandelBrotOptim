# The AVX-512 Mandelbrot renderer

## Roadmap
* (Done) Simple scalar implementation (Baseline, 0.22s)
* (Done) Multithreading using openMP (around x10 speedup, 0.02s)
* (Almost Done) AVX-512 Vectorization and some additional optimization (0.006s)
* (Plan) Make zoom in-out

## Technical highlights
* SIMD Vectorization: Manually utilized using 512-bit registers
* Instriction Level Implementation
* Multi-threading: Dynamic scheduling via OpenMP.
## Hardware
* CPU: x86-64 CPU with AVX-512 support (Tested on ryzen AI 7 350 laptop)
* Compiler: Used GCC 13.3
