import micromagneticmodel as mm
import discretisedfield as df
import oommfc as oc
import numpy as np

# MMC driver package imports
import MMCMag.MMC_driver as mmc

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

mmc_mesh = mmc.Mesh((Lx, Ly, Lz), (Nx, Ny, Nz))

# System properties

mmc_system = mmc.System(Ms, T, mmc_mesh, mag_status=0, is_atomistic=False)

# Add interactions to the system
mmc_system.add_zeeman(H)
mmc_system.add_ua(K, u)
mmc_system.add_exchange(A)
mmc_system.add_dmi(D)

# UBERMAG SYSTEM
# Create a system for testing purposes
uber_system = mm.System(name="system_test_init")

# System size and mesh
uber_region = df.Region(p1=(0, 0, 0), p2=(Lx, Ly, Lz))
uber_mesh = df.Mesh(region=uber_region, n=(Nx, Ny, Nz))
uber_system.m = df.Field(
    uber_mesh, nvdim=3, value=mmc_system.mag, norm=Ms
)  # Initialise magnetisations as the same as other system
uber_system.T = T

# Energy parameters
uber_system.energy = (
    mm.Zeeman(H=H)
    + mm.UniaxialAnisotropy(K=K, u=u)
    + mm.Exchange(A=A)
    + mm.DMI(D=D, crystalclass="T")
)

# TESTS
rtol_test = 1e-5


def test_unit_mag():
    """
    Test unity of magnetisation vectors of system.
    """

    assert np.allclose(np.linalg.norm(mmc_system.mag, axis=-1), 1.0, rtol=rtol_test)


def test_zeeman_init():
    """
    Test Zeeman energy density and total energy of our system against Ubermag.
    """

    # Initialisation in simulation
    mmc_zeeman_w = mmc_system.zeeman.zeeman
    mmc_zeeman_E = mmc_zeeman_w.sum() * dV

    # Initialisation in Ubermag
    uber_zeeman_w = oc.compute(
        uber_system.energy.zeeman.density, uber_system
    ).array.squeeze()
    uber_zeeman_E = oc.compute(uber_system.energy.zeeman.energy, uber_system)

    assert np.allclose(mmc_zeeman_w, uber_zeeman_w, rtol=rtol_test)
    assert np.isclose(mmc_zeeman_E, uber_zeeman_E, rtol=rtol_test)


def test_ua_init():
    """
    Test UA energy density and total energy of our system against Ubermag.
    """

    print(u, K)

    # Initialisation in simulation
    mmc_ua_w = mmc_system.ua.ua
    mmc_ua_E = mmc_ua_w.sum() * dV

    # Initialisation in Ubermag
    uber_ua_w = oc.compute(
        uber_system.energy.uniaxialanisotropy.density, uber_system
    ).array.squeeze()
    uber_ua_E = oc.compute(uber_system.energy.uniaxialanisotropy.energy, uber_system)

    assert np.allclose(mmc_ua_w, uber_ua_w, rtol=rtol_test)
    assert np.isclose(mmc_ua_E, uber_ua_E, rtol=rtol_test)


def test_exchange_init():
    """
    Test exchange energy densities and total energy of our system against Ubermag.
    """

    print(A)

    # Initialisation in simulation
    mmc_exchange_x = mmc_system.exchange.exchange_x
    mmc_exchange_y = mmc_system.exchange.exchange_y
    mmc_exchange_z = mmc_system.exchange.exchange_z
    mmc_exchange_w = (
        mmc_exchange_x[1:, :, :]
        + mmc_exchange_x[:-1, :, :]
        + mmc_exchange_y[:, 1:, :]
        + mmc_exchange_y[:, :-1, :]
        + mmc_exchange_z[:, :, 1:]
        + mmc_exchange_z[:, :, :-1]
    )
    mmc_exchange_E = mmc_exchange_w.sum() * dV

    # Initialisation in Ubermag
    uber_exchange_w = oc.compute(
        uber_system.energy.exchange.density, uber_system
    ).array.squeeze()
    uber_exchange_E = oc.compute(uber_system.energy.exchange.energy, uber_system)

    print(mmc_exchange_E, uber_exchange_E)

    assert np.allclose(mmc_exchange_w, uber_exchange_w, rtol=rtol_test)
    assert np.isclose(mmc_exchange_E, uber_exchange_E, rtol=rtol_test)


def test_exchange_init():
    """
    Test UA energy densities and total energy of our system against Ubermag.
    """

    print(A)

    # Initialisation in simulation
    mmc_exchange_x = mmc_system.exchange.exchange_x
    mmc_exchange_y = mmc_system.exchange.exchange_y
    mmc_exchange_z = mmc_system.exchange.exchange_z
    mmc_exchange_w = (
        mmc_exchange_x[1:, :, :]
        + mmc_exchange_x[:-1, :, :]
        + mmc_exchange_y[:, 1:, :]
        + mmc_exchange_y[:, :-1, :]
        + mmc_exchange_z[:, :, 1:]
        + mmc_exchange_z[:, :, :-1]
    )
    mmc_exchange_E = mmc_exchange_w.sum() * dV

    # Initialisation in Ubermag
    uber_exchange_w = oc.compute(
        uber_system.energy.exchange.density, uber_system
    ).array.squeeze()
    uber_exchange_E = oc.compute(uber_system.energy.exchange.energy, uber_system)

    print(mmc_exchange_E, uber_exchange_E)

    assert np.allclose(mmc_exchange_w, uber_exchange_w, rtol=rtol_test)
    assert np.isclose(mmc_exchange_E, uber_exchange_E, rtol=rtol_test)


def test_dmi_init():
    """
    Test DMI energy densities and total energy of our system against Ubermag.
    """

    # Initialization in simulation
    mmc_dmi_x = mmc_system.dmi.dmi_x
    mmc_dmi_y = mmc_system.dmi.dmi_y
    mmc_dmi_z = mmc_system.dmi.dmi_z
    mmc_dmi_w = (
        mmc_dmi_x[1:, :, :]
        + mmc_dmi_x[:-1, :, :]
        + mmc_dmi_y[:, 1:, :]
        + mmc_dmi_y[:, :-1, :]
        + mmc_dmi_z[:, :, 1:]
        + mmc_dmi_z[:, :, :-1]
    )
    mmc_dmi_E = mmc_dmi_w.sum() * dV

    # Initialization in Ubermag
    uber_dmi_w = oc.compute(uber_system.energy.dmi.density, uber_system).array.squeeze()
    uber_dmi_E = oc.compute(uber_system.energy.dmi.energy, uber_system)

    assert np.allclose(mmc_dmi_w, uber_dmi_w, rtol=rtol_test)
    assert np.isclose(mmc_dmi_E, uber_dmi_E, rtol=rtol_test)
