# MMCMag 🧲

This package for micromagnetic simulation is based on Ubermag syntax. It operates with Zeeman, uniaxial anisotropy, exchange, and DMI interactions. The driving algorithm is the Metropolis Monte Carlo algorithm, and the API makes use of `Mesh`, `System`, and `Simulation` classes. 🔄

- `MMC_driver.py` contains a non-parallelised MMC driver and the associated classes. 🚗
- (🚧**COMING SOON**🚧) `GPU_MMC_driver.py` contains a checkerboard algorithm version of the algorithm that runs on the GPU via PyCUDA. 🚀
