# Thomas William Unitt-Jones (ACSE-twu18)
# setup.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from shutil import copyfile
import os

# Cython extension with Extension
"""
This Cython Extension 'simulation_wrapper' serves as a binding between the functionality of the MPI CPU-parallelized MMC algorithm's 
'run_simulation' wrapper function from the 'run_simulation.cpp' file and Python
It allows this function to be called directly from Python.

**Note**: A compatible version of MPI must be installed, and the binding must be built by the user.
For more information, please refer to the project's README.

"""

# Default include directories
include_dirs = ["include"]

# Add additional directories for specific environments
if os.environ.get("GITHUB_ACTIONS"):  # Running on GitHub Actions
    include_dirs.append("/usr/include/mpi")
else:  # Running locally
    include_dirs.append("/opt/homebrew/Cellar/open-mpi/4.1.5/include")

extension = Extension(
    name="CPU_driver_wrapper",
    sources=[
        "CPU_driver_wrapper.pyx",
        "src/Simulation.cpp",
        "src/ContiguousArray4D.cpp",
        "src/Timer.cpp",
    ],
    include_dirs=include_dirs,
    libraries=["mpi"],
    library_dirs=[
        "/opt/homebrew/Cellar/open-mpi/4.1.5/lib",
        "/opt/homebrew/opt/libevent/lib",
    ],
    language="c++",
    extra_compile_args=["-std=c++11"],
)

setup(
    ext_modules=cythonize([extension]),
)

# Rename the generated .so file
os.rename("./CPU_driver_wrapper.cpython-310-darwin.so", "./CPU_driver_wrapper.so")
