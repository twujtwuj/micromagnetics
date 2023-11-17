# MMCMag 🧲

This package for **Python-based** micromagnetic simulation is based on Ubermag syntax. It operates with Zeeman, uniaxial anisotropy, exchange, and DMI interactions. The driving algorithm is the Metropolis Monte Carlo algorithm, and the API makes use of `Mesh`, `System`, and `Simulation` classes. 🔄

- `MMC_driver.py` contains a non-parallelised MMC driver and the associated classes. 🚗
- `MMC_demo.ipynb` includes a demonstration of the `MMC_driver` from the `MMCMag` package, a non-parallelised micromagnetic simulator based on Ubermag syntax that uses the Metropolis Monte Carlo algorithm to sequentially update magnetic states. Install it with `pip install -e .` in the main directory. 🛠️
- (🚧**COMING SOON**🚧) `GPU_MMC_driver.py` contains a checkerboard algorithm version of the algorithm that runs on the GPU via PyCUDA. 🚀
