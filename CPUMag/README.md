# MPI (project root directory)

This is the **project root directory**. For code builds and compilations, please nagivate to this folder. 
For more information on installations, builds, compilations, and HPC runs, please see the README in the **repository root directory** (level above).

This directory contains:
- `cpusimpy`: custom Python package that wraps the MPI C++ code via a shared object build with Cython.
- `include`: contains all of the `.h` files of the C++ source code.
- `out`: contains the results from a small system with all interactions present (so that `tests/test_simulation.py` is immediately possible)
- `src`: conatins all of the `.cpp` files of the C++ source code.
- `tests`: contains various unit tests to ensure correect functionality of our driver, intended to be run with Pytest.
- `automatic_job_generation.sh`: quickly queue multiple jobs on a PBS HPC (like the College's one)
- `main.cpp`: main if source code is compiled manually.
- `setup.py`, `simulation_wrapper.pyx`: for creation of Cython extension and `cpusimpy` package.
- `ubermag_integration.ipynb`: demonstration of the functionality of our driver.