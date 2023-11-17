# simulation_wrapper.pyx
# cython: language_level=3

"""
Cython Interface for External C Functions

This file serves as a Cython interface to bind the external C functions defined in "src/run_simulation.cpp" to Python.

"""

cdef extern from "src/run_simulation.cpp":
    void run_simulation_interface(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double T, 
                                double Ms, bint atomistic, double H[3], double u[3], double K, 
                                double D, double J, int init_status, bint alpha_decay, int it_max)

def run_simulation_py(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double T, 
                      double Ms, bint atomistic, 
                      double[:] H, double[:] u, double K, double D, double J,
                      int init_status, bint alpha_decay, int it_max):
   
    """
    Run simulation with the given parameters.

    Parameters:
    ----------
    Nx, Ny, Nz : int
        Dimensions of the simulation mesh.
    Lx, Ly, Lz : double
        Physical dimensions of the system.
    T : double
        Temperature.
    Ms : double
        Saturation magnetisation.
    atomistic : bool
        Whether atomistic or continuous interpretaion of constants.
    H : list of doubles
        External magnetic field [Hx, Hy, Hz].
    u : list of doubles
        Uniaxial anisotropy axis [ux, uy, uz].
    K : double
        Uniaxial anisotropy constant.
    D : double
        Dzyaloshinskii-Moriya interaction constant.
    J : double
        Exchange interaction constant.
    init_status : int
        Initialization status.
    alpha_decay : bool
        Alpha exponential decay option.
    it_max : int
        Maximum number of iterations.

    """
    run_simulation_interface(Nx, Ny, Nz, Lx, Ly, Lz, T, Ms, atomistic, &H[0], &u[0], K, D, J, init_status, alpha_decay, it_max)

