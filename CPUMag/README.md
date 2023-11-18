# CPUMag 🧲

**C++-based** CPU-parallelised MMC driver using MPI using the checkerboard algorithm, bound to Python with Cython. 🧲
Operating with Zeeman, uniaxial anisotropy, exchange, and DMI interactions, this driving algorithm is the Metropolis Monte Carlo algorithm. 🔄

- `CPU_driver.py` contains a CPU-parallelised MMC driver 🖥️
- `CPU_demo.py` displays the functionality of this driver 📚
- `include`: contains all of the `.h` files of the C++ source code 📂
- `src`: contains all of the `.cpp` files of the C++ source code 📂
- `main.cpp` can be used as main if source code is compiled manually 🎯
- `setup.py`, `CPU_driver_wrapper.cpp` and `.pyx` are for the creation of Cython extension 📦
- `utilities.py` contains useful auxiliary functions 🎯

## Compiling and running C++ source code and HPC usage

To compile and run the C++ code on `p` processes, from the root directory run:

```
mpic++ -std=c++11 -Iinclude main.cpp src/*.cpp -o main
mpiexec -n p main
```


## Cython extnesion

To use the CPU driver from Python, the user must build the Cython extension:

```
python setup.py build_ext --inplace
pip install -e .
```

**NOTE:** By default, only `#define TO_FILE` and `#define VERBOSE` conditional directives are in use in the source code, but the user can change this if they wish.

A copy of the shared object `simulation_wrapper.so` is automatically placed into the `tests` and `results` folders.
