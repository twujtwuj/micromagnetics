// Thomas William Unitt-Jones (ACSE-twu18)
/**
 * @file run_simulation.h
 * @brief This file declares the `run_simulation` which wraps the simulation's functionality.
 * This function is then exposed to Python via Cython
 */
#pragma once

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
                    double D, double J, int init_status, bool alpha_decay, int it_max);
