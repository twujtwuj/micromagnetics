// Thomas William Unitt-Jones (ACSE-twu18)
// run_simulation.cpp
#include "Simulation.h"
#include <mpi.h>
#include "run_simulation.h"

/**
 * @brief Run the simulation with specified parameters.
 * 
 * @param Nx Number of discretisation nodes along the x-axis.
 * @param Ny Number of discretisation nodes along the y-axis.
 * @param Nz Number of discretisation nodes along the z-axis.
 * @param Lx Physical length along the x-axis of the simulation region.
 * @param Ly Physical length along the y-axis of the simulation region.
 * @param Lz Physical length along the z-axis of the simulation region.
 * @param T Temperature of the system.
 * @param Ms Saturation magnetization.
 * @param atomistic Flag indicating whether the simulation is atomistic.
 * @param H External magnetic field [Hx, Hy, Hz].
 * @param u Uniaxial anisotropy direction [ux, uy, uz].
 * @param K Uniaxial anisotropy constant.
 * @param D DMI constant.
 * @param J Exchange constant.
 * @param init_status Initial status of the simulation.
 * @param alpha_decay Flag indicating whether exponential alpha decay is enabled.
 * @param it_max Maximum number of iterations.
 * @see run_simulation
 */
void run_simulation(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double T, 
                    double Ms, bool atomistic, double H[3], double u[3], double K, 
                    double D, double J, int init_status, bool alpha_decay, int it_max) 
{
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        MPI_Init(nullptr, nullptr);
    }

    Simulation simulation(Nx, Ny, Nz, Lx, Ly, Lz, T, Ms, atomistic, init_status, alpha_decay, it_max);
    simulation.add_zeeman(H);
    simulation.add_anisotropy(K, u);
    simulation.add_exchange(J);
    simulation.add_DMI(D);
    simulation.run();

    if (!is_initialized) {
        MPI_Finalize();
    }
}

/**
 * @brief C wrapper function to provide a pure C interface to the simulation function for Cython.
 *  
 * This function is used to encapsulate the functionality of the MPI C++ driver in order to expose it to Python via Cython.
 * For details on how to build the extension, refer to the project's README. 
 * 
 * @see run_simulation
 */
extern "C" {
    void run_simulation_interface(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double T, 
                                double Ms, bool atomistic, double H[3], double u[3], double K, 
                                double D, double J, int init_status, bool alpha_decay, int it_max) {
        run_simulation(Nx, Ny, Nz, Lx, Ly, Lz, T, Ms, atomistic, H, u, K, D, J, init_status, alpha_decay, it_max);
    }
}
