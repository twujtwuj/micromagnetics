# test_driver_zeeman.py
import micromagneticmodel as mm
import discretisedfield as df
import oommfc as oc
import numpy as np

# MMC driver package imports
import MMCMag.MMC_driver as mmc
from MMCMag.MMC_driver import random_unitvec

# Use Docker image
# If Ubermag cannot be installed fully, install mm, df, and oc and use this Docker image:
# docker_runner = oc.oommf.DockerOOMMFRunner(image="ubermag/oommf")

"""
This script contains unit tests the ZEEMAN interaction in our driver.

The tests include simple scenarios where a small system (3 x 3 x 3) is initialised
with random magnetisation and then subjected to different external magnetic field
conditions to test the behavior of the Zeeman interaction. 
Parallelised on two processes. 
These tests use constats similar to those for FeGe (see table in report).s

Tested Scenarios:
1. Zero external magnetic field and no other interactions present.
   Expect no change.
2. A uniform external magnetic field in the positive z-direction 
   Expect z-components to be approximately 1.
3. A uniform external magnetic field in the negative z-direction 
   Expect z-components to be approximately -1.

Note: The `rtol_test` variable is used as the relative tolerance for the assert statements.
"""


Lx, Ly, Lz = 50e-9, 50e-9, 50e-9
Nx, Ny, Nz = 5, 5, 5
dV = (Lx / Nx) * (Ly / Ny) * (Lz / Nz)

Ms = 384e3
T = 0

H = 1e5 * random_unitvec()

# MMC SYSTEM
# Domain dimensions and discretisation (mesh)
mmc_mesh = mmc.Mesh((Lx, Ly, Lz), (Nx, Ny, Nz))

# MMC simulation
mmc_system = mmc.System(Ms, T, mmc_mesh, mag_status=1, is_atomistic=False)  # Unit x-direction spins
mmc_system.add_zeeman(H)
mmc_simulation = mmc.Simulation(mmc_system)
its = 1e5

# UBERMAG SYSTEM
# Create a system for testing purposes
uber_system = mm.System(name="system_test_zeeman")

# System size and mesh
uber_region = df.Region(p1=(0, 0, 0), p2=(Lx, Ly, Lz))
uber_mesh = df.Mesh(region=uber_region, n=(Nx, Ny, Nz))
uber_system.m = df.Field(
    uber_mesh, 3, value=(1, 0, 0), norm=Ms
)  # Initialise magnetisations as the same as other system
uber_system.T = T

# Energy parameters
uber_system.energy = (
    mm.Zeeman(H=H)
)

# Minimisation driver
md = oc.MinDriver()

# TESTS
rtol_test = 1e-2


def test_driver_zeeman():
    """
    Test the behavior of the Zeeman interaction with a positive external magnetic field.

    Uses a random magnetisation initialisation. After simulation, it reads the final magnetisation
    data and checks if the z-component of the magnetisation values are close to 1.
    """
    # Drive MMC system
    mmc_simulation.run_MMC(its)  # Run simulation
    mmc_zeeman_w = mmc_system.zeeman.zeeman
    mmc_zeeman_E = mmc_zeeman_w.sum() * dV

    # Drive Ubermag system
    md.drive(uber_system)
    uber_zeeman_w = oc.compute(
        uber_system.energy.zeeman.density, uber_system
    ).array.squeeze()
    uber_zeeman_E = oc.compute(uber_system.energy.zeeman.energy, uber_system)

    assert np.allclose(mmc_zeeman_w, uber_zeeman_w, rtol=rtol_test)
    assert np.isclose(mmc_zeeman_E, uber_zeeman_E, rtol=rtol_test)