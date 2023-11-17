# Thomas William Unitt-Jones (ACSE-twu18)
# drivers.py
import subprocess
import numpy as np
import micromagneticmodel as mm
import discretisedfield as df
import oommfc as oc
import simulation_wrapper as sw
from cpusimpy.utilities import read_data_file


def driver(system, p, atomistic, init_status=0, alpha_decay=False, it_max=1e7):
    """
    Runs a CPU-parallelised MMC driver on a given micromagnetic model system.
    Updates the system.m.orientation attribute to final magnetisations. This driver is verbose;
    it prints every 10% iteration reached, along with key statistcs.

    Parameters:
    ----------
    system : object
        A micromagnetic model system object with mesh and energy parameters defined.

    p : int
        The number of processes to parallelise on.

    atomistic : bool
        Specifies if the magnetic constants should be interpreted as atomistic or continuous ones.

    init_status : int, optional
        The initial configuration of the system. Defaults to 0.
        0: Random initialisation (default)
        1: Unit x initialisation
        2: Unit y initialisation
        3: Unit z initialisation

    alpha_decay : bool, optional
        If True, the search cone angle will decrease exponentially with iterations to help with convergence.
        Defaults to False.

    it_max : float, optional
        The maximum number of iterations the simulator will run for. Defaults to 1e7.

    Side Effects:
    ------------
    Modifies system.m.orientation attribute in place.

    Examples:
    --------
    >>> driver(mySystem, 4, True, init_status=1, alpha_decay=True, it_max=5e6)

    Notes:
    -----
    It is not currently possible to use the initial magnetisation stored in the system object in the driver,
    hence the init_status variable.

    If an energy term is stated in the system object but no values are given, any constants associated with
    that intereaction are interpreted as null.

    """

    if atomistic:
        print(
            "Warning: Uberamag cannot be used to benchmark the results of atomistic simulations."
        )

    # Extract values
    Nx, Ny, Nz = system.m.mesh.n  # Assume a rectangular mesh
    Lx, Ly, Lz = np.array(system.m.mesh.region.p2) - np.array(system.m.mesh.region.p1)
    T = system.T
    Ms = np.mean(system.m.norm.array)  # Assume constant Ms
    if "Zeeman" not in str(system.energy):  # Check if Zeeman present in system
        H = np.array([0, 0, 0], dtype=np.float64)
    else:
        H = np.array(system.energy.zeeman.H, dtype=np.float64)
    if "UniaxialAnisotropy" not in str(
        system.energy
    ):  # Check if anisotropy present in system
        u = np.array([0, 0, 0], dtype=np.float64)
        K = 0
    else:
        u = np.array(system.energy.uniaxialanisotropy.u, dtype=np.float64)
        K = system.energy.uniaxialanisotropy.K
    if "Exchange" not in str(system.energy):  # Check if exchange present in system
        J = 0
    else:
        J = system.energy.exchange.A
    if "DMI" not in str(system.energy):  # Check if DMI present in system
        D = 0
    else:
        D = system.energy.dmi.D

    # Convert parameters to strings
    H_str = ", ".join(map(str, H))
    u_str = ", ".join(map(str, u))
    atomistic_str = "True" if atomistic else "False"
    alpha_decay_str = "True" if alpha_decay else "False"
    args_str = f"{Nx}, {Ny}, {Nz}, {Lx}, {Ly}, {Lz}, {T}, {Ms}, {atomistic_str}, np.array([{H_str}]), np.array([{u_str}]), {K}, {D}, {J}, {init_status}, {alpha_decay_str}, {it_max}"
    cmd = (
        f"mpiexec -n {p} python -c 'import numpy as np; "
        f"import simulation_wrapper as sw; sw.run_simulation_py({args_str})'"
    )

    # Run simulation
    subprocess.call(cmd, shell=True)

    # Extract data from files and modify system
    _, mag_end = read_data_file("out/mu_end.dat")
    system.m = df.Field(system.m.mesh, dim=3, value=mag_end, norm=Ms)
