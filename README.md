# Micromagnetics 🧲

This repository includes some code I developed for my master's project entitled "Monte Carlo Micromagnetic Simulator with Checkerboard Parallelisation."

- `MMC_demo.ipynb` includes a demonstration of the `MMC_driver` from the `MMCMag` package, a non-parallelised micromagnetic simulator based on Ubermag syntax that uses the Metropolis Monte Carlo algorithm to sequentially update magnetic states. Install it with `pip install -e .`. 🛠️
- `MMC_project.pdf` gives some background in micromagnetic simulation in the context of this project. 📚

A GPU-parallelised version of the driver using the checkerboard algorithm with PyCUDA will follow soon. 🚀
