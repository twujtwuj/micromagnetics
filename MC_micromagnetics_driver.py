import numpy as np
import time
import matplotlib.pyplot as plt

# GLOBALS
kB = 1.3806452e-23
mu0 = 4 * np.pi * 1e-7


class Mesh:
    """
    A class to encompass the physical dimensions of the (cuboid) domain.
    This is similar to the df.Mesh and df.Region classes from Ubermag.
    """

    def __init__(self, L=(100e-9, 100e-9, 100e-9), N=(10, 10, 10)):
        # Physical lengths
        self.Lx = L[0]
        self.Ly = L[1]
        self.Lz = L[2]
        self.L = L

        # Number of cells
        self.Nx = N[0]
        self.Ny = N[1]
        self.Nz = N[2]
        self.N = N

        # Spacings
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.d = (self.dx, self.dy, self.dz)


class Zeeman:
    """
    Stores externmal magnetic field H and initialises Zeeman energy terms for a system.mag array.
    """

    def __init__(self, system, H):
        self.H = H
        self.Hx = H[0]
        self.Hy = H[1]
        self.Hz = H[2]

        # E_Z = - \mu_0 H \cdot \mu
        self.zeeman = -mu0 * np.sum(self.H * system.mag, axis=-1)
        if not system.is_atomistic:  # Rescale for continuous interpretation
            self.zeeman = system.Ms * self.zeeman


class UA:
    """
    Stores uniaxial anisotropy axis and constant and initialises anisotropy energy terms for a system.mag array.
    """

    def __init__(self, system, u, K):
        self.K = K
        norm_u = np.linalg.norm(u)
        if norm_u != 0:
            u = u / norm_u
        self.u = u
        self.ux = u[0]
        self.uy = u[1]
        self.uz = u[2]

        # E_a = - K (u \cdot \mu)^2
        # K is equivalent in both cases
        self.ua = -self.K * np.sum(self.u * system.mag, axis=-1) ** 2


class Exchange:
    """
    Stores exchange constant and initialises exchange energy terms in three directions for a system.mag array.
    """

    def __init__(self, system, A):
        self.A = A
        Nx, Ny, Nz = system.mesh.N
        self.exchange_x = np.zeros((Nx + 1, Ny, Nz))
        self.exchange_y = np.zeros((Nx, Ny + 1, Nz))
        self.exchange_z = np.zeros((Nx, Ny, Nz + 1))
        self.exchange_x[1:-1, :, :] = -self.A * np.sum(
            system.mag[:-1, :, :] * system.mag[1:, :, :], axis=-1
        )
        self.exchange_y[:, 1:-1, :] = -self.A * np.sum(
            system.mag[:, :-1, :] * system.mag[:, 1:, :], axis=-1
        )
        self.exchange_z[:, :, 1:-1] = -self.A * np.sum(
            system.mag[:, :, 1:] * system.mag[:, :, :-1], axis=-1
        )
        if not system.is_atomistic:  # Rescale for continuous interpretation
            self.exchange_x = self.exchange_x / (system.mesh.dx**2)
            self.exchange_y = self.exchange_y / (system.mesh.dy**2)
            self.exchange_z = self.exchange_z / (system.mesh.dz**2)


class DMI:
    """
    Stores DMI constant and initialises DMI energy terms in three directions for a system.mag array.
    """

    def __init__(self, system, D):
        self.D = D
        Nx, Ny, Nz = system.mesh.N
        self.dmi_x = np.zeros((Nx + 1, Ny, Nz))
        self.dmi_y = np.zeros((Nx, Ny + 1, Nz))
        self.dmi_z = np.zeros((Nx, Ny, Nz + 1))
        self.dmi_x[1:-1, :, :] = -np.sum(
            (self.D * np.array([1, 0, 0]))
            * np.cross(system.mag[:-1, :, :], system.mag[1:, :, :], axis=-1),
            axis=-1,
        )  # assume positive r_ijk
        self.dmi_y[:, 1:-1, :] = -np.sum(
            (self.D * np.array([0, 1, 0]))
            * np.cross(system.mag[:, :-1, :], system.mag[:, 1:, :], axis=-1),
            axis=-1,
        )  # assume positive r_ijk
        self.dmi_z[:, :, 1:-1] = -np.sum(
            (self.D * np.array([0, 0, 1]))
            * np.cross(system.mag[:, :, :-1], system.mag[:, :, 1:], axis=-1),
            axis=-1,
        )  # assume positive r_ijk
        if not system.is_atomistic:  # Rescale for continuous interpretation
            self.dmi_x = self.dmi_x / (2 * system.mesh.dx)
            self.dmi_y = self.dmi_y / (2 * system.mesh.dy)
            self.dmi_z = self.dmi_z / (2 * system.mesh.dz)


class System:
    """
    Holds the system parameters, the magnetisations, and some plotting functionalities.
    Eventually this would be replaced by the mm.System class.
    """

    def __init__(self, Ms, T, mesh, mag_status=0, is_atomistic=False):
        self.mesh = mesh

        # Atomistic interpretation?
        self.is_atomistic = is_atomistic

        # Temperature
        self.T = T

        # Magnetic saturation
        self.Ms = Ms

        # Initial magnetisations
        if mag_status == 0:
            self.mag = self.random_init()
        if mag_status == 1:
            self.mag = np.zeros((mesh.Nx, mesh.Ny, mesh.Nz, 3))
            self.mag[:, :, :, 0] = 1
        if mag_status == 2:
            self.mag = np.zeros((mesh.Nx, mesh.Ny, mesh.Nz, 3))
            self.mag[:, :, :, 1] = 1
        if mag_status == 3:
            self.mag = np.zeros((mesh.Nx, mesh.Ny, mesh.Nz, 3))
            self.mag[:, :, :, 2] = 1

        self.is_zeeman = False
        self.is_ua = False
        self.is_exchange = False
        self.is_dmi = False

    # Generate random mag (unit field)
    def random_init(self):
        theta = np.pi * np.random.rand(self.mesh.Nx, self.mesh.Ny, self.mesh.Nz)
        phi = 2 * np.pi * np.random.rand(self.mesh.Nx, self.mesh.Ny, self.mesh.Nz)
        mag = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )  # convert to cartesian
        mag = np.transpose(mag, (1, 2, 3, 0))  # rearrange order of dimensions
        return mag

    # Zeemam
    def add_zeeman(self, H):
        self.zeeman = Zeeman(self, H)
        self.is_zeeman = True

    # Uniaxial anisotropy
    def add_ua(self, K, u):
        self.ua = UA(self, u, K)
        self.is_ua = True

    # Exchange
    def add_exchange(self, A):
        self.exchange = Exchange(self, A)
        self.is_exchange = True

    # DMI
    def add_dmi(self, D):
        self.dmi = DMI(self, D)
        self.is_dmi = True

    # Plotting functionality
    def plot_quiver(self):
        Lx, Ly, Lz = self.mesh.L  # sample size
        Nx, Ny, Nz = self.mesh.N

        x, y, z = np.meshgrid(
            np.arange(Nx) * (Lx / (max(Nx, 2) - 1)),
            np.arange(Ny) * (Ly / (max(Ny, 2) - 1)),
            np.arange(Nz) * (Lz / (max(Nz, 2) - 1)),
            indexing="ij",
        )
        mag_x = self.mag[:, :, :, 0]
        mag_y = self.mag[:, :, :, 1]
        mag_z = self.mag[:, :, :, 2]

        # Figure
        plt.rcParams["text.usetex"] = False  # LaTeX rendering
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        length = max(Lx / Nx, Ly / Ny, Lz / Nz)
        aspect_ratio = [Lx, Ly, Lz]
        if Nx == 1:
            ax.set_xlim([-Lx / 2, Lx / 2])
            # aspect_ratio[1] =
        if Ny == 1:
            ax.set_ylim([-Ly / 2, Ly / 2])
        if Nz == 1:
            ax.set_zlim([-Lz / 2, Lz / 2])
        ax.set_box_aspect(aspect_ratio)  # Aspect ratio

        # Quiver
        ax.quiver(
            x,
            y,
            z,
            mag_x,
            mag_y,
            mag_z,
            length=length,
            normalize=True,
            pivot="tail",
            label="mag",
            alpha=0.5,
        )
        center_x, center_y, center_z = Lx / 2, Ly / 2, Lz / 2

        # Magnetic interaction vectors
        if self.is_zeeman:
            H_x, H_y, H_z = self.zeeman.H  # Zeeman external magnetic field
            ax.quiver(
                center_x,
                center_y,
                center_z,
                H_x,
                H_y,
                H_z,
                color="b",
                length=length / 2,
                normalize=True,
                pivot="tail",
                label="H",
            )

        if self.is_ua:
            u_x, u_y, u_z = self.ua.u  # Anisotropy axis
            ax.quiver(
                center_x,
                center_y,
                center_z,
                u_x,
                u_y,
                u_z,
                color="r",
                length=length / 2,
                normalize=True,
                pivot="tail",
                label="u",
            )

        # Displaying
        if self.is_atomistic:
            plt.title(f"$m$ (atomistic)")
        else:
            plt.title(f"$m$ (continuous)")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        ax.legend()
        plt.show()


class Simulation:
    def __init__(self, system):
        self.system = system
        self.it_max = it_max
        self.total_E_tracker = None
        self.total_E = None
        self.run_time = None

    # Monte-Carlo, non-parallelised, driver
    # (purely numpy implementation with no checkerboard)
    def run_MMC(self, it_max, alpha=0.5, verbose=True):
        start = time.time()
        self.it_max = it_max

        # Simulation variables
        it = 0
        total_E_tracker = []  # fill with total energy

        # Extract parameters
        T = self.system.T
        Ms = self.system.Ms
        H = self.system.zeeman.H
        u = self.system.ua.u
        K = self.system.ua.K
        A = self.system.exchange.A
        D = self.system.dmi.D
        mag = self.system.mag
        Nx, Ny, Nz = self.system.mesh.N
        Lx, Ly, Lz = self.system.mesh.L
        dx, dy, dz = self.system.mesh.d

        # Initialise total energies
        total_E = 0
        total_E += self.system.zeeman.zeeman.sum()  # add all Zeeman energies
        total_E += self.system.ua.ua.sum()  # add all anisotropy energies
        total_E += (
            2 * self.system.exchange.exchange_x.sum()
        )  # add all exchange energies
        total_E += 2 * self.system.exchange.exchange_y.sum()
        total_E += 2 * self.system.exchange.exchange_z.sum()
        total_E += 2 * self.system.dmi.dmi_x.sum()  # add all dmi energies
        total_E += 2 * self.system.dmi.dmi_y.sum()
        total_E += 2 * self.system.dmi.dmi_z.sum()

        # Main simulation loop
        while it < it_max:
            # Uniformly generate proposal
            x = np.random.randint(Nx)
            y = np.random.randint(Ny)
            z = np.random.randint(Nz)
            mag_x, mag_y, mag_z = self.system.mag[x, y, z]

            # Find cell-vicinity energy
            zeeman_E = self.system.zeeman.zeeman[x, y, z]  # add all Zeeman energies
            ua_E = self.system.ua.ua[x, y, z]  # add all anisotropy energies
            exchange_right_E = self.system.exchange.exchange_x[x + 1, y, z]
            exchange_left_E = self.system.exchange.exchange_x[x, y, z]
            exchange_up_E = self.system.exchange.exchange_y[x, y + 1, z]
            exchange_down_E = self.system.exchange.exchange_y[x, y, z]
            exchange_front_E = self.system.exchange.exchange_z[x, y, z + 1]
            exchange_back_E = self.system.exchange.exchange_z[x, y, z]
            dmi_right_E = self.system.dmi.dmi_x[x + 1, y, z]
            dmi_left_E = self.system.dmi.dmi_x[x, y, z]
            dmi_up_E = self.system.dmi.dmi_y[x, y + 1, z]
            dmi_down_E = self.system.dmi.dmi_y[x, y, z]
            dmi_front_E = self.system.dmi.dmi_z[x, y, z + 1]
            dmi_back_E = self.system.dmi.dmi_z[x, y, z]

            local_E = (
                zeeman_E
                + ua_E
                + 2 * exchange_right_E
                + 2  # Multiply by two since this energy is the same for the neighbouring atom
                * exchange_left_E
                + 2 * exchange_up_E
                + 2 * exchange_down_E
                + 2 * exchange_front_E
                + 2 * exchange_back_E
                + 2 * dmi_right_E
                + 2 * dmi_left_E
                + 2 * dmi_up_E
                + 2 * dmi_down_E
                + 2 * dmi_front_E
                + 2 * dmi_back_E
            )

            # Find spherical coodinates of atom
            r = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
            theta = np.arccos(mag_z / r)
            phi = np.arctan2(mag_y, mag_x)

            # Propose a perturbation in sperical coordinates and convert back to cartesian
            proposal_theta = (
                theta + alpha * np.pi * (np.random.rand() - 1 / 2)
            ) % np.pi
            proposal_phi = (phi + alpha * 2 * np.pi * (np.random.rand() - 1 / 2)) % (
                2 * np.pi
            )
            proposal_mag_x = np.sin(proposal_theta) * np.cos(proposal_phi)
            proposal_mag_y = np.sin(proposal_theta) * np.sin(proposal_phi)
            proposal_mag_z = np.cos(proposal_theta)
            proposal = np.array([proposal_mag_x, proposal_mag_y, proposal_mag_z]).T

            # Calculate proposal energy
            proposed_zeeman_E = -mu0 * np.sum(H * proposal, axis=-1)  # Zeeman
            proposed_ua_E = -K * np.sum(u * proposal, axis=-1) ** 2  # anisotropy
            proposed_exchange_right_E = (
                -A * np.sum(proposal * mag[x + 1, y, z], axis=-1) if x + 1 < Nx else 0
            )  # right
            proposed_exchange_left_E = (
                -A * np.sum(proposal * mag[x - 1, y, z], axis=-1) if x > 0 else 0
            )  # left
            proposed_exchange_up_E = (
                -A * np.sum(proposal * mag[x, y + 1, z], axis=-1) if y + 1 < Ny else 0
            )  # up
            proposed_exchange_down_E = (
                -A * np.sum(proposal * mag[x, y - 1, z], axis=-1) if y > 0 else 0
            )  # down
            proposed_exchange_front_E = (
                -A * np.sum(proposal * mag[x, y, z + 1], axis=-1) if z + 1 < Nz else 0
            )  # front
            proposed_exchange_back_E = (
                -A * np.sum(proposal * mag[x, y, z - 1], axis=-1) if z > 0 else 0
            )  # back
            proposed_dmi_right_E = -np.sum(
                D * np.array([1, 0, 0]) * np.cross(proposal, mag[x + 1, y, z], axis=-1)
                if x + 1 < Nx
                else 0
            )  # right
            proposed_dmi_left_E = -np.sum(
                D * np.array([-1, 0, 0]) * np.cross(proposal, mag[x - 1, y, z], axis=-1)
                if x > 0
                else 0
            )  # left
            proposed_dmi_up_E = -np.sum(
                D * np.array([0, 1, 0]) * np.cross(proposal, mag[x, y + 1, z], axis=-1)
                if y + 1 < Ny
                else 0
            )  # up
            proposed_dmi_down_E = -np.sum(
                D * np.array([0, -1, 0]) * np.cross(proposal, mag[x, y - 1, z], axis=-1)
                if y > 0
                else 0
            )  # down
            proposed_dmi_front_E = -np.sum(
                D * np.array([0, 0, 1]) * np.cross(proposal, mag[x, y, z + 1], axis=-1)
                if z + 1 < Nz
                else 0
            )  # front
            proposed_dmi_back_E = -np.sum(
                D * np.array([0, 0, -1]) * np.cross(proposal, mag[x, y, z - 1], axis=-1)
                if z > 0
                else 0
            )  # back
            if not self.system.is_atomistic:  # Rescaling if continuous interpretation
                proposed_zeeman_E = proposed_zeeman_E * Ms
                proposed_ua_E = proposed_ua_E  # No change
                proposed_exchange_right_E = proposed_exchange_right_E / (dx**2)
                proposed_exchange_left_E = proposed_exchange_left_E / (dx**2)
                proposed_exchange_up_E = proposed_exchange_up_E / (dy**2)
                proposed_exchange_down_E = proposed_exchange_down_E / (dy**2)
                proposed_exchange_front_E = proposed_exchange_front_E / (dz**2)
                proposed_exchange_back_E = proposed_exchange_back_E / (dz**2)
                proposed_dmi_right_E = proposed_dmi_right_E / (2 * dx)
                proposed_dmi_left_E = proposed_dmi_left_E / (2 * dx)
                proposed_dmi_up_E = proposed_dmi_up_E / (2 * dy)
                proposed_dmi_down_E = proposed_dmi_down_E / (2 * dy)
                proposed_dmi_front_E = proposed_dmi_front_E / (2 * dz)
                proposed_dmi_back_E = proposed_dmi_back_E / (2 * dz)

            # Find local energy change
            proposed_local_E = (
                proposed_zeeman_E
                + proposed_ua_E
                + 2 * proposed_exchange_right_E
                + 2  # Multiply by two since this energy is the same for the neighbouring atom
                * proposed_exchange_left_E
                + 2 * proposed_exchange_up_E
                + 2 * proposed_exchange_down_E
                + 2 * proposed_exchange_front_E
                + 2 * proposed_exchange_back_E
                + 2 * proposed_dmi_right_E
                + 2 * proposed_dmi_left_E
                + 2 * proposed_dmi_up_E
                + 2 * proposed_dmi_down_E
                + 2 * proposed_dmi_front_E
                + 2 * proposed_dmi_back_E
            )

            # Calculate delta_E and compare
            delta_E = proposed_local_E - local_E
            r = np.random.rand()
            if (T == 0 and delta_E < 0.0) or (
                T != 0 and (delta_E < 0.0 or r < np.exp(-delta_E / (kB * T)))
            ):  # accept proposal
                self.system.mag[x, y, z] = proposal  # update mu
                self.system.zeeman.zeeman[x, y, z] = proposed_zeeman_E
                self.system.ua.ua[x, y, z] = proposed_ua_E
                self.system.exchange.exchange_x[x + 1, y, z] = proposed_exchange_right_E
                self.system.exchange.exchange_x[x, y, z] = proposed_exchange_left_E
                self.system.exchange.exchange_y[x, y + 1, z] = proposed_exchange_up_E
                self.system.exchange.exchange_y[x, y, z] = proposed_exchange_down_E
                self.system.exchange.exchange_z[x, y, z + 1] = proposed_exchange_front_E
                self.system.exchange.exchange_z[x, y, z] = proposed_exchange_back_E
                self.system.dmi.dmi_x[x + 1, y, z] = proposed_dmi_right_E
                self.system.dmi.dmi_x[x, y, z] = proposed_dmi_left_E
                self.system.dmi.dmi_y[x, y + 1, z] = proposed_dmi_up_E
                self.system.dmi.dmi_y[x, y, z] = proposed_dmi_down_E
                self.system.dmi.dmi_z[x, y, z + 1] = proposed_dmi_front_E
                self.system.dmi.dmi_z[x, y, z] = proposed_dmi_back_E
                total_E += delta_E  # update total energy

            # Track energy changes and acceptence rate
            total_E_tracker.append(total_E)

            if (it + 1) % (it_max // 10) == 0:
                if verbose:
                    print(f"Iteration {it + 1}/{it_max} complete")
            it += 1  # increment time

        end = time.time()
        if verbose:
            print(f"Time elapsed for MC: {end - start}s")
            print("Simulation complete")

        self.total_E_tracker = total_E_tracker
        self.total_E = total_E
        self.run_time = end - start

    def plot_energy_tracker(self):
        if self.it_max > 1000:
            plt.plot(
                self.total_E_tracker[:: self.it_max // 100],
                np.arange(1, self.it_max, self.it_max // 100),
            )
        else:
            plt.plot(self.total_E_tracker)
        plt.show()


if __name__ == "__main__":
    Lx, Ly, Lz = 40e-9, 40e-9, 5e-9  # sample size
    Nx, Ny, Nz = 20, 20, 3
    # Lx, Ly, Lz = 10e-9, 10e-9, 10e-9  # sample size
    # Nx, Ny, Nz = 1, 10, 1
    mesh = Mesh((Lx, Ly, Lz), (Nx, Ny, Nz))
    Ms = 384e3
    T = 0
    system = System(
        Ms, T, mesh, mag_status=0, is_atomistic=False
    )  # Random initialisation, continnuous interpretation of constants
    system.add_zeeman((0, 0, 3e5))
    system.add_ua(0, (0, 0, 0))
    system.add_exchange(8.78e-12)
    system.add_dmi(1.58e-3)

    it_max = 1000
    simulation = Simulation(system)
    simulation.system.plot_quiver()
    simulation.run_MMC(it_max)
    simulation.system.plot_quiver()
    simulation.plot_energy_tracker()
