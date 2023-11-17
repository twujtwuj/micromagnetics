# Thomas William Unitt-Jones (ACSE-twu18)
# utilities.py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def read_data_file(file_name):
    """
    Reads a .dat file and extracts metadata and main data as a properly shaped numpy array.
    This function is intended to be the reciprocal of the ContiguousArray4D::to_tab_file member function.

    Parameters:
    ----------
    file_name : str
        Relative path to the .dat file to be read.

    Returns:
    --------
    metadata : list
        A list containing the extracted metadata in the following order:
        procs: Number of processes (p)
        it: Iteration number when saved to file
        atomistic: Whether atomistic or continuous
        Nx, Ny, Nz: Dimensions of the system
        nx, ny, nz, nv: Dimensions of the returned array
        Lx, Ly, Lz: Physical dimensions of the system
        Ms, T, H, K, u, J, D: Magnetic parameters

    all_data : numpy.ndarray
        The main data shaped as a numpy array with dimensions (nx, ny, nz, nv).

    Notes:
    -----
    Metadata are extracted from the first 13 lines of the .dat file.
    The rest of the file is read to obtain the main data.

    Examples:
    --------
    >>> metadata, all_data = read_data_file("out/mu_end.dat")

    """

    data = []
    with open(file_name, "r") as file:
        # Read metadata
        procs = int(file.readline().strip())
        it = int(file.readline().strip())
        atomistic = int(file.readline().strip())
        Nx, Ny, Nz = map(int, file.readline().strip().split("\t"))
        nx, ny, nz, nv = map(int, file.readline().strip().split("\t"))
        Lx, Ly, Lz = map(float, file.readline().strip().split("\t"))
        Ms = float(file.readline().strip())
        T = float(file.readline().strip())
        H = np.array([float(value) for value in file.readline().strip().split("\t")])
        K = float(file.readline().strip())
        u = np.array([float(value) for value in file.readline().strip().split("\t")])
        J = float(file.readline().strip())
        D = float(file.readline().strip())

        # Read main data
        for line in file:
            row = [float(value) for value in line.strip().split("\t")]
            data.append(row)

    # Reshape according to array dimensions
    all_data = np.array(data)
    if nx == it:  # If energy array
        all_data = np.reshape(all_data, (-1))
        all_data = all_data.squeeze()
    elif nv == 1:
        all_data = np.reshape(all_data, (nx, ny, nz))
    else:
        all_data = np.reshape(all_data, (nx, ny, nz, nv))

    return [
        procs,
        it,
        atomistic,
        Nx,
        Ny,
        Nz,
        nx,
        ny,
        nz,
        nv,
        Lx,
        Ly,
        Lz,
        Ms,
        T,
        H,
        K,
        u,
        J,
        D,
    ], all_data


def plot_quiver(file_name):
    """
    Generates a quiver plot of the vector field in a given .dat file.
    Intended for visualizing magnetization vector fields.

    Parameters:
    ----------
    file_name : str
        Relative path to the .dat file to be read.

    Returns:
    --------
    A quiver plot of the vector field.

    Notes:
    -----
    Reads data using the read_data_file function.
    Requires that nv == 3 in the .dat file, otherwise will raise an exception.

    Examples:
    --------
    >>> plot_quiver("out/mu_end.dat")

    """

    [
        procs,
        it,
        atomistic,
        Nx,
        Ny,
        Nz,
        nx,
        ny,
        nz,
        nv,
        Lx,
        Ly,
        Lz,
        Ms,
        T,
        H,
        K,
        u,
        J,
        D,
    ], mag = read_data_file(file_name)

    # Raise exception if not a 3-dimensional vector field
    if nv != 3:
        raise Exception(
            f"There bust be 3 componenents in the fourth dimension but there are {nv}."
        )

    x, y, z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    mag_x = mag[:, :, :, 0]
    mag_y = mag[:, :, :, 1]
    mag_z = mag[:, :, :, 2]

    # Figure
    plt.rcParams["text.usetex"] = True  # LaTeX rendering
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([Lx, Ly, Lz])  # Aspect ratio
    if Nx == 1:
        ax.set_xlim([-1 / 2, 1 / 2])
    if Ny == 1:
        ax.set_ylim([-1 / 2, 1 / 2])
    if Nz == 1:
        ax.set_zlim([-1 / 2, 1 / 2])

    # Quiver
    ax.quiver(
        x,
        y,
        z,
        mag_x,
        mag_y,
        mag_z,
        length=0.5,
        normalize=True,
        pivot="tail",
        label="mag",
        alpha=0.5,
    )
    center_x, center_y, center_z = (Nx - 1) / 2, (Ny - 1) / 2, (Nz - 1) / 2
    length = min(Nx, Ny, Nz) / 2

    # Magnetic interaction vectors
    H_x, H_y, H_z = H  # Zeeman external magnetic field
    ax.quiver(
        center_x,
        center_y,
        center_z,
        H_x,
        H_y,
        H_z,
        color="b",
        length=length,
        normalize=True,
        pivot="tail",
        label="H",
    )
    u_x, u_y, u_z = u  # Anisotropy axis
    ax.quiver(
        center_x,
        center_y,
        center_z,
        u_x,
        u_y,
        u_z,
        color="r",
        length=length,
        normalize=True,
        pivot="tail",
        label="u",
    )

    # Displaying
    if atomistic == 1:
        plt.title(f"$m$ at $i = {it}$, $p = {procs}$, $T = {T}$ (atomistic)")
    else:
        plt.title(f"$m$ at $i = {it}$, $p = {procs}$, $T = {T}$ (continuous)")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.legend()
    plt.show()


def plot_energy_data(file_name):
    """
    Generates a plot of the one-dimensional energy data in a given .dat file.
    Intended for visualising the energy_tracker variable from Simulation::run.
    Helps to indicate if the simulation is still in evolution.

    Parameters:
    ----------
    file_name : str
        Relative path to the .dat file to be read.

    Returns:
    --------
    A plot of the one-dimensional energy data on axes.

    Notes:
    -----
    Reads data using the read_data_file function.
    Requires that nv == 1 in the .dat file, otherwise, the function will raise an exception.

    Examples:
    --------
    >>> plot_energy_data("out/energy_data.dat")

    """

    # sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': '0.8'})
    # colors = sns.color_palette("colorblind", 1)
    plt.rcParams["text.usetex"] = True

    [
        procs,
        it,
        atomistic,
        Nx,
        Ny,
        Nz,
        nx,
        ny,
        nz,
        nv,
        Lx,
        Ly,
        Lz,
        Ms,
        T,
        H,
        K,
        u,
        J,
        D,
    ], energy_data = read_data_file(file_name)

    # Raise exception if not a one dimensional vector
    if len(energy_data.shape) != 1:
        raise Exception(
            "This function is only suitable for plotting one-dimensional data"
        )

    # Create the plot
    plt.figure(figsize=(10, 6))
    x_values = np.arange(0, it, 100)
    plt.plot(x_values, energy_data[:it:100], label="Energy")

    # Add title and labels and display
    if atomistic == 1:
        plt.title(f"System energy for $p$ = {procs}, $T$ = {T} (atomistic)")
    else:
        plt.title(f"System energy for $p$ = {procs}, $T$ = {T} (continuous)")
    plt.xlabel("Iteration $i$")
    plt.ylabel("Energy $E$")
    plt.legend()
    plt.show()


# Run from MPI folder
if __name__ == "__main__":
    # Plot the initial, final, and energy curve
    plot_quiver("out/mu_init.dat")
    plot_quiver("out/mu_end.dat")
    # plot_energy_data("out/energy_data.dat")
    # plot_energy_data("out/average_mu_x.dat")
