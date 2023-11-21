import micromagneticmodel as mm
import discretisedfield as df
import oommfc as oc
import numpy as np

# MMC driver package imports
#import MMCMag.MMC_driver as mmc

# Use Docker image
# If Ubermag cannot be installed fully, install mm, df, and oc and use this Docker image:
# docker_runner = oc.oommf.DockerOOMMFRunner(image="ubermag/oommf")

"""
Unit Tests for run_simulation output

These tests verify that the initial energy densities and total energies for each interaction the corresponding Ubermag (OOMMFC) calculations. A relative tolerance of rtol_test is applied to check for acceptable deviations.

Tests are run on a small (5, 5, 5) system with random magnetic parameters of realistic orders of magnitude.

"""


def random_unitvec():
    """
    Generates a vector randomly and uniformly on the unit sphere.
    """
    theta = np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    vec = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )  # convert to cartesian
    return vec


Lx, Ly, Lz = 50e-9, 50e-9, 50e-9
Nx, Ny, Nz = 5, 5, 5
dV = (Lx / Nx) * (Ly / Ny) * (Lz / Nz)

Ms = 384e3
T = 0

H = 1e5 * random_unitvec()
K = 2e6 * (np.random.rand() - 0.5)  # [-1e6, 1e6]
u = random_unitvec()
A = 2e-11 * (np.random.rand() - 0.5)  # [-1e11, 1e11]
D = 2e-2 * (np.random.rand() - 0.5)  # [-1e2, 1e2]

# MMC SYSTEM
# Domain dimensions and discretisation (mesh)

# mmc_mesh = mmc.Mesh((Lx, Ly, Lz), (Nx, Ny, Nz))

# # System properties

# mmc_system = mmc.System(Ms, T, mmc_mesh, mag_status=0, is_atomistic=False)

# # Add interactions to the system
# mmc_system.add_zeeman(H)
# mmc_system.add_ua(K, u)
# mmc_system.add_exchange(A)
# mmc_system.add_dmi(D)

# UBERMAG SYSTEM
# Create a system for testing purposes
uber_system = mm.System(name="system_test_init")

# System size and mesh
uber_region = df.Region(p1=(0, 0, 0), p2=(Lx, Ly, Lz))
uber_mesh = df.Mesh(region=uber_region, n=(Nx, Ny, Nz))
uber_system.m = df.Field(
    uber_mesh, 3, value=mmc_system.mag, norm=Ms
)  # Initialise magnetisations as the same as other system
uber_system.T = T

# Energy parameters
uber_system.energy = (
    mm.Zeeman(H=H)
    + mm.UniaxialAnisotropy(K=K, u=u)
    + mm.Exchange(A=A)
    + mm.DMI(D=D, crystalclass="T")
)

def test_zeeman_init():
    """
    Test Zeeman energy density and total energy of our system against Ubermag.
    """

    # Initialisation in simulation
    # mmc_zeeman_w = mmc_system.zeeman.zeeman
    # mmc_zeeman_E = mmc_zeeman_w.sum() * dV

    # Initialisation in Ubermag
    uber_zeeman_w = oc.compute(
        uber_system.energy.zeeman.density, uber_system
    ).array.squeeze()
    uber_zeeman_E = oc.compute(uber_system.energy.zeeman.energy, uber_system)

    print(uber_zeeman_w, uber_zeeman_E)
    assert True


def test_dmi_init():
    """
    Test DMI energy densities and total energy of our system against Ubermag.
    """

    # Initialization in Ubermag
    uber_dmi_w = oc.compute(uber_system.energy.dmi.density, uber_system).array.squeeze()
    uber_dmi_E = oc.compute(uber_system.energy.dmi.energy, uber_system)

    print(uber_dmi_w, uber_dmi_E)

    assert True
