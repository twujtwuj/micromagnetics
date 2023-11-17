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
