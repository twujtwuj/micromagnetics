// Thomas William Unitt-Jones (ACSE-twu18)
// main.cpp
// This file is intended for use when direct compilation from C++ is desired.
// It serves as the entry point for running the simulation from source.

#include "include/run_simulation.h"
#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[])
{
    // Initialise MPI environment
    MPI_Init(&argc, &argv);

    // Initialise simulation parameters
    int init_status = 0;   // Initialisation status
    bool alpha_decay = false;  // Flag for alpha exponential decay (search-cone angle)
    int it_max = 1e3;  // Maximum number of iterations

    // Lattice dimensions
    int Nx = 3; //40
    int Ny = 3; //40
    int Nz = 3; //5

    // Physical dimensions
    double Lx = 80e-9;
    double Ly = 80e-9;
    double Lz = 80e-9; //10e-9

    // Temperature and magnetisation
    double T = 0; 
    double Ms = 384e3;

    // Atomistic or continuum
    bool atomistic = false;

    // External magnetic field for Zeeman interaction
    double H[3] = {0, 0, 2e5}; 

    // Uniaxial anisotropy axis and constant
    double u[3] = {1, 0, 0};
    double K = 45e4; 

    // DMI constant
    double D = 1.58e-3;

    // Exchange constant
    double J = 8.78e-12;

    // Run the simulation with the specified parameters from wrappers
    run_simulation(Nx, Ny, Nz, Lx, Ly, Lz, T, Ms, atomistic, H, u, K, D, J, init_status, alpha_decay, it_max);

    // Finalise MPI environment
    MPI_Finalize();

    return 0;
}

