// Thomas William Unitt-Jones (ACSE-twu18)
/**
 * @file Simulation.h
 * @brief Header file for the Simulation class.
 */

#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <mpi.h>
#include <random>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include "ContiguousArray4D.h"
#include "AtomData.h"

using namespace std;

/** 
 * @class Simulation
 * @brief Manages the simulation of magnetic atoms in a system.
 * 
 * Provides methods for initialisation and running various interactions
 * in the simulation.
 */
class Simulation {
public:

    /// @name Global Variables
    ///@{
    
    int Nx, Ny, Nz; ///< Lattice dimensions in x, y, and z directions.
    int Nv; ///< Dimensionality of spin space (default is 3).
    
    double Lx, Ly, Lz; ///< Physical dimensions in x, y, and z directions.
    double dx, dy, dz; ///< Lattice spacings in x, y, and z directions.
    
    double T; ///< Temperature.
    double Ms; ///< Saturation magnetisation.
    double alpha; ///< Search-cone angle.
    double kB; ///< Boltzmann constant.
    double total_E; ///< Total energy of the system.
    double mu0; ///< Permeability of free space.
    double rtol; ///< Relative tolerance for stopping criteria (if applied).
    
    bool atomistic; ///< Flag for atomistic interpretation of constants.
    bool alpha_decay; ///< Flag for exponential alpha decay (decreasing search-cone breadth).
    
    int it; ///< Current iteration.
    int it_max; ///< Maximum number of iterations cut-off.
    ///@}

    /// @name Interaction Variables
    ///@{
    ContiguousArray4D mu, zeeman, anisotropy, exchange_x, exchange_y, exchange_z, DMI_x, DMI_y, DMI_z; ///< Arrays for storing spins and energies.
    
    // Proposal variables
    int atom_pos[3]; ///< Coordinates of the selected atom for proposal.
    double proposed_mu[3]; ///< Proposed spin value for selected atom.
    
    // Current energies (before update)
    double zeeman_E; ///< Current Zeeman energy.
    double anisotropy_E; ///< Current anisotropy energy.
    double exchange_right_E, exchange_left_E, exchange_up_E, exchange_down_E, exchange_front_E, exchange_back_E; ///< Current exchange energies.
    double DMI_right_E, DMI_left_E, DMI_up_E, DMI_down_E, DMI_front_E, DMI_back_E; ///< Current DMI energies.
    
    // Proposed energies
    double proposed_local_delta_E; ///< Proposed change in local energy.
    double proposed_zeeman_E, proposed_anisotropy_E; ///< Proposed Zeeman and anisotropy energies.
    double proposed_exchange_right_E, proposed_exchange_left_E, proposed_exchange_up_E, proposed_exchange_down_E, proposed_exchange_front_E, proposed_exchange_back_E; ///< Proposed exchange energies.
    double proposed_DMI_right_E, proposed_DMI_left_E, proposed_DMI_up_E, proposed_DMI_down_E, proposed_DMI_front_E, proposed_DMI_back_E; ///< Proposed DMI energies.
    
    // Interaction-specific variables and rescalings
    double H[3], H_rescaled[3]; ///< External field and its rescaled (continuous) version.
    double K, K_rescaled, u[3]; ///< Anisotropy constants and anisotropy direction and its rescaled (continuous) version.
    double J, J_rescaled[3]; ///< Exchange constants and rescaled (continuous) version.
    double D, D_rescaled[3]; ///< DMI constants and rescaled (continuous) version.
    ///@}

    /// @name Random Generation
    ///@{
    random_device rd; ///< Random device for generating seeds.
    mt19937 gen; ///< Mersenne Twister random generator.
    ///@}

    /// @name Parallel Computing Variables
    ///@{
    MPI_Datatype MPI_AtomData; ///< Custom MPI datatype for atom data.
    int tag_num; ///< Tag number for MPI communication.
    int id; ///< Process ID for MPI.
    int p; ///< Total number of MPI processes.
    int q; ///< Number of MPI processes with updates.
    int N_red; ///< Number of red atoms for checkerboard pattern.
    int N_black; ///< Number of black atoms for checkerboard pattern.
    vector<vector<int>> red_atoms; ///< Red atoms in the checkerboard pattern.
    vector<vector<int>> black_atoms; ///< Black atoms in the checkerboard pattern.
    vector<int> atoms_list; ///< List of atoms to be distributed to processes.
    AtomData atom_data; ///< Atom data for MPI communication.
    bool is_updated; ///< Flag to indicate if atom data is updated.
    int local_update; ///< Whether an update was made or not on this process.
    ///@}

public:
    /**
     * @brief Create MPI datatype for atom data.
     */
    void create_MPI_datatype();

    /**
     * @brief Create checkerboards and update red_atoms and black_atoms.
     */
    void find_checkerboards();

    /**
     * @brief Constructor for Simulation class.
     * 
     * @param Nx Number of atoms along x-direction.
     * @param Ny Number of atoms along y-direction.
     * @param Nz Number of atoms along z-direction.
     * @param Lx Physical length along x-direction.
     * @param Ly Physical length along y-direction.
     * @param Lz Physical length along z-direction.
     * @param T Temperature.
     * @param Ms Saturation magnetisation.
     * @param atomistic Whether atomistic simulation.
     * @param unit_status Unit conversion status.
     * @param alpha_decay Whether alpha decay is enabled.
     * @param it_max Maximum number of iterations.
     * @param Nv Dimensionality of spin space (default is 3).
     * @param alpha Gilbert damping constant (default is 1.0).
     * @param rtol Relative tolerance for simulation convergence (default is 1e-10).
     */
    Simulation(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double T, double Ms, bool atomistic, int unit_status, bool alpha_decay, int it_max, int Nv=3, double alpha=1.0, double rtol=1e-10);
    
     /**
     * @brief Add Zeeman interaction.
     * 
     * @param H External magnetic field.
     */
    void add_zeeman(double* H); 

     /**
     * @brief Add uniaxial anisotropy interaction.
     * 
     * @param K Anisotropy constant.
     * @param u Anisotropy anisotropy axis.
     */
    void add_anisotropy(double K, double* u);

    /**
     * @brief Add exchange interaction.
     * 
     * @param J Exchange constant.
     */
    void add_exchange(double J);

    /**
     * @brief Add DMI.
     * 
     * @param D DMI constant.
     */
    void add_DMI(double D);
    
    /**
     * @brief Initialise energy densities for Zeeman interaction.
     */
    void initialise_zeeman();

    /**
     * @brief Initialise energy densities for uniaxial anisotropy interaction.
     */
    void initialise_anisotropy();

    /**
     * @brief Initialise energy densities for exchange interaction.
     */
    void initialise_exchange();

    /**
     * @brief Initialise energy densities for DMI interaction.
     */
    void initialise_DMI();

    /**
     * @brief Initialise the total energy of the system.
     */
    void initialise_total_E();

    /**
     * @brief Initialise random device with a seed based on the process ID.
     * 
     * @param id Process ID for seed generation.
     */
    void initialise_random_device(int id);

    /**
     * @brief Propose a perturbation to the selected atom's spin.
     * 
     * Updates the mu_proposal array with the new spin orientation.
     */
    void rand_perturbation();

    /**
     * @brief Generate a random double.
     * 
     * @return A random double value from the initialised random device.
     */
    double rand_double();

    /**
     * @brief Generate a random integer within a specified range.
     * 
     * @param min Minimum integer value.
     * @param max Maximum integer value.
     * @return A random integer between min and max from the intialised random device.
     */
    int rand_int(int min, int max);

    /**
     * @brief Generate a random sample of atom indices.
     * 
     * Uses one of two methods to select N values from N_colour without replacement.
     * Can be used in a `red` iteration or a `black` iteration to select a sample
     * of atoms of that colour.
     * 
     * @param N Number of atoms to sample (usually the number of processes used, p).
     * @param N_colour Total number of atoms of a given colour (usually the numebr of red or black atoms, R_red or R_black).
     * @return A vector containing the sampled atom indices.
     */
    vector<int> generate_random_sample(int N, int N_colour);

    /**
     * @brief Main simulation loop; runs for at most it_max iterations.
     */
    void run();
};
#endif // SIMULATION_H
