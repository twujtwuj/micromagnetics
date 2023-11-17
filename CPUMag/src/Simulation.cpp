// Thomas William Unitt-Jones (ACSE-twu18)
// Simulation.cpp
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <mpi.h>
#include <set>
#include "Simulation.h"
#include "Timer.h"
#include "ContiguousArray4D.h"
#include "AtomData.h"

using namespace std;

//#define FISHER_YATES // Chooses unbiased Fisher-Yates instead of biased set-shuffle for random sampling.
//#define DO_TIME_PROFILING // For time profiling; enables the measurement of time taken for specific operations.
//#define TIME_TEN_MILLION_UPDATES // For timing 1e7 updates used to profile the time performance.
#define TO_FILE // Whether or not data is saved in .dat files.
#define VERBOSE // Prints additional messages during the simulation to indicate progress.
//#define STOPPING_CRITERION // Whether or not to stop simulation before it_max with relative stopping criterion.
//#define TEMPERATURE_PROFILE // Enables temperature profiling; useful for simulations involving varying temperature.

/**
 * @brief Create and commit a custom MPI datatype MPI_AtomData based on the AtomData structure.
 * 
 * This function uses MPI to define a new custom datatype that maps directly to
 * the AtomData structure. This allows us to send AtomData objects through MPI.
 * The custom datatype, MPI_AtomData, is then committed so it can be used in subsequent MPI calls.
 */
void Simulation::create_MPI_datatype()
{
    AtomData temp;  // Temporary AtomData object to calculate memory offsets

    int block_lengths[17];
    MPI_Datatype block_types[17];
    MPI_Aint start_address, member_address;
    MPI_Aint block_offsets[17];

    // Define blocks for atom position (3 integers)
    block_lengths[0] = 3;
    block_types[0] = MPI_INT;

    // Define blocks for accepted spin (3 doubles)
    block_lengths[1] = 3;
    block_types[1] = MPI_DOUBLE;

    // Define blocks for the 14 energy terms and change in energy (all doubles)
    for (int i = 2; i < 17; i++)
    {
        block_lengths[i] = 1;
        block_types[i] = MPI_DOUBLE;
    }

    // Calculate the memory offsets of each member in AtomData structure
    MPI_Get_address(&temp.atom_pos, &start_address);
    block_offsets[0] = 0;
    MPI_Get_address(&temp.new_mu, &member_address);
    block_offsets[1] = member_address - start_address;
    MPI_Get_address(&temp.local_delta_E, &member_address);
    block_offsets[2] = member_address - start_address;
    MPI_Get_address(&temp.zeeman_E, &member_address);
    block_offsets[3] = member_address - start_address;
    MPI_Get_address(&temp.anisotropy_E, &member_address);
    block_offsets[4] = member_address - start_address;
    MPI_Get_address(&temp.exchange_right_E, &member_address);
    block_offsets[5] = member_address - start_address;
    MPI_Get_address(&temp.exchange_left_E, &member_address);
    block_offsets[6] = member_address - start_address;
    MPI_Get_address(&temp.exchange_up_E, &member_address);
    block_offsets[7] = member_address - start_address;
    MPI_Get_address(&temp.exchange_down_E, &member_address);
    block_offsets[8] = member_address - start_address;
    MPI_Get_address(&temp.exchange_front_E, &member_address);
    block_offsets[9] = member_address - start_address;
    MPI_Get_address(&temp.exchange_back_E, &member_address);
    block_offsets[10] = member_address - start_address;
    MPI_Get_address(&temp.DMI_right_E, &member_address);
    block_offsets[11] = member_address - start_address;
    MPI_Get_address(&temp.DMI_left_E, &member_address);
    block_offsets[12] = member_address - start_address;
    MPI_Get_address(&temp.DMI_up_E, &member_address);
    block_offsets[13] = member_address - start_address;
    MPI_Get_address(&temp.DMI_down_E, &member_address);
    block_offsets[14] = member_address - start_address;
    MPI_Get_address(&temp.DMI_front_E, &member_address);
    block_offsets[15] = member_address - start_address;
    MPI_Get_address(&temp.DMI_back_E, &member_address);
    block_offsets[16] = member_address - start_address;

    // Create and commit the custom MPI datatype
    MPI_Type_create_struct(17, block_lengths, block_offsets, block_types, &MPI_AtomData);
    MPI_Type_commit(&MPI_AtomData);

    // Optional verbose output
    #ifdef VERBOSE
        if (id == 0)
            cout << "MPI_AtomData custom datatype successfully created ..." << endl;
    #endif
}

/**
 * @brief Finds and categorises atoms into 'red' and 'black' checkerboards based on their positions.
 * 
 * This function identifies the 'red' and 'black' atoms in a 3D grid of dimensions Nx * Ny * Nz. 
 * Atoms are categorised based on the sum of their x, y, z coordinates.
 * If the sum is even (it%2 == 0), the atom is 'red'; otherwise (it%2 == 1), it is 'black'.
 * 
 * The function also sets the counts of 'red' and 'black' atoms as N_red and N_black.
 */
void Simulation::find_checkerboards() 
{
    // Total number of atoms in the grid
    int N_total = Nx * Ny * Nz;
    
    // Assuming an even distribution of 'red' and 'black' atoms initially (corrected later)
    N_red = N_total / 2;
    N_black = N_total / 2;

    // Adjust N_red and N_black if N_total is odd
    if (N_total % 2 != 0) {
        if ((Nx * Ny) % 2 == 0) {
            N_black += 1;  // Add an extra 'black' atom
        } else {
            N_red += 1;  // Add an extra 'red' atom
        }
    }

    // Temporary vector to hold atom coordinates
    vector<int> atom;

    // Loop through all atoms to categorise them as 'red' or 'black'
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                atom.clear();
                atom.push_back(i);
                atom.push_back(j);
                atom.push_back(k);
                
                // Determine if the atom is 'red' or 'black' based on its coordinates
                if ((i + j + k) % 2 == 0) {
                    red_atoms.push_back(atom);  // 'red' atom
                } else {
                    black_atoms.push_back(atom);  // 'black' atom
                }
            }
        }
    }
}

/**
 * @brief Constructs a Simulation object with specified parameters.
 * 
 * Initialises various simulation parameters and finds 'red' and 'black' checkerboard atoms. 
 * Sets up MPI environment and initialises mu (magnetic moment) based on init_status.
 * 
 * @param Nx Number of atoms in x-direction
 * @param Ny Number of atoms in y-direction
 * @param Nz Number of atoms in z-direction
 * @param Lx Length of simulation in x-direction
 * @param Ly Length of simulation in y-direction
 * @param Lz Length of simulation in z-direction
 * @param T Temperature
 * @param Ms Saturation magnetisation
 * @param atomistic Whether the simulation is atomistic or not
 * @param init_status Initialisation status for magnetic moments (0: random, 1: unit x, 2: unit y, 3: unit z)
 * @param alpha_decay Whether alpha decay is enabled
 * @param it_max Maximum number of iterations
 * @param Nv Dimensionality of spin space (default is 3)
 * @param alpha Search-cone angle
 * @param rtol Relative tolerance for stopping criteria (if applied)
 */
Simulation::Simulation(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double T, double Ms,
                       bool atomistic, int init_status, bool alpha_decay, int it_max, int Nv,
                       double alpha, double rtol)
    : mu(Nx, Ny, Nz, Nv), // Initialisation list
      Nx(Nx),
      Ny(Ny),
      Nz(Nz),
      Nv(Nv),
      Lx(Lx),
      Ly(Ly),
      Lz(Lz),
      dx(Lx / static_cast<double>(Nx)),
      dy(Ly / static_cast<double>(Ny)),
      dz(Lz / static_cast<double>(Nz)),
      T(T),
      Ms(Ms),
      atomistic(atomistic),
      alpha(alpha),
      kB(1.3806452e-23),
      total_E(0),
      it(0),
      it_max(it_max),
      rtol(rtol),
      mu0(4.0 * M_PI * 1e-7),
      alpha_decay(alpha_decay),
      tag_num(0)
{
    // Find 'red' and 'black' checkerboard atoms
    find_checkerboards();

    // MPI environment setup
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Random device and generator
    random_device rd;
    mt19937 gen; // Mersenne Twister

    // Verbose mode message, if enabled
    #ifdef VERBOSE
        if (id == 0)
            cout << p << " process(es) used ..." << endl;
    #endif

    // Override initialisation status if temperature profiling is enabled
    #ifdef TEMPERATURE_PROFILE
        init_status = 1; // Unit x direction for temperature
    #endif

    // Initialise mu as zeros
    for (int i = 0; i < Nx * Ny * Nz * Nv; i++)
        mu.array_1D[i] = 0;

    // Validate initialisation status
    if (init_status != 0 && init_status != 1 && init_status != 2 && init_status != 3)
        throw invalid_argument("Initialisation status must be one of 0: random; 1: unit x; 2: unit y; 3: unit z!");

    // Unit x-direction initialisation (1)
    if (init_status == 1)
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++)
                {
                    mu.array_4D[i][j][k][0] = 1.0;
                }  

    // Unit y-direction initialisation (2)
    if (init_status == 2)
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++)
                {
                    mu.array_4D[i][j][k][0] = 0.0;
                    mu.array_4D[i][j][k][1] = 1.0;
                }

    // Unit z-direction initialisation (3)
    if (init_status == 3)
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++)
                {
                    mu.array_4D[i][j][k][0] = 0.0;
                    mu.array_4D[i][j][k][1] = 0.0;
                    mu.array_4D[i][j][k][2] = 1.0;
                }

    // Random, uniform spherical initialisation (0)
    if (init_status == 0) {
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    double theta = M_PI * (rand() / (double)RAND_MAX);
                    double phi = 2 * M_PI * (rand() / (double)RAND_MAX);
                    
                    // Convert to Cartesian coordinates
                    double x = sin(theta) * cos(phi);
                    double y = sin(theta) * sin(phi);
                    double z = cos(theta);
                    
                    mu.array_4D[i][j][k][0] = x;
                    mu.array_4D[i][j][k][1] = y;
                    mu.array_4D[i][j][k][2] = z;
                }
            }
        }
    }  
}

/**
 * @brief Adds an external magnetic field (Zeeman field) to the simulation.
 * 
 * Rescales the Zeeman field based on whether the simulation is atomistic or not.
 * Initialises the array storing Zeeman field contributions at each lattice site.
 * 
 * @param H Pointer to an array storing the components of the external magnetic field
 */
void Simulation::add_zeeman(double* H) {
    
    // If the simulation is atomistic
    if (atomistic==true)
    {
        this->H_rescaled[0] = H[0];
        this->H_rescaled[1] = H[1];
        this->H_rescaled[2] = H[2];
    }
    else // Interpret as continuous equations
    {
        this->H_rescaled[0] = Ms * H[0];
        this->H_rescaled[1] = Ms * H[1];
        this->H_rescaled[2] = Ms * H[2]; 
    }

    // Store the original Zeeman field for printing to file
    this->H[0] = H[0];
    this->H[1] = H[1];
    this->H[2] = H[2];

    // Initialise the 4D array for the Zeeman field
    zeeman = ContiguousArray4D(Nx, Ny, Nz, 1);

    // Call the function to initialise Zeeman field at each lattice site
    initialise_zeeman();
}

/**
 * @brief Adds an anisotropy term to the simulation.
 * 
 * Rescales the anisotropy constant based on whether the simulation is atomistic or not.
 * Initialises the array storing anisotropy contributions at each lattice site.
 * 
 * @param K The anisotropy constant
 * @param u Pointer to an array storing the components of the anisotropy axis
 */
void Simulation::add_anisotropy(double K, double* u) {

    // Rescale the anisotropy constant depending on atomistic or continuous model.
    // Note: Rescaling is the same for both atomistic and continuous in this case.
    if (atomistic==true)
        this->K_rescaled = K;
    else
        this->K_rescaled = K;

    // Store the original anisotropy constant for printing to file
    this->K = K;

    // Calculate the magnitude of the anisotropy axis
    double u_magnitude = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

    // Normalise the anisotropy axis if its magnitude is not zero
    if (u_magnitude != 0) {
        this->u[0] = u[0] / u_magnitude;
        this->u[1] = u[1] / u_magnitude;
        this->u[2] = u[2] / u_magnitude;
    } else {
        // If the anisotropy axis has zero magnitude, set the anisotropy constants to zero
        this->u[0] = u[0];
        this->u[1] = u[1];
        this->u[2] = u[2];
        this->K_rescaled = 0;
        this->K = 0;
    }

    // Initialise the 4D array for the anisotropy term
    anisotropy = ContiguousArray4D(Nx, Ny, Nz, 1);

    // Call the function to initialise anisotropy at each lattice site
    initialise_anisotropy();
}


/**
 * @brief Adds an exchange interaction term to the simulation.
 * 
 * Rescales the exchange constant J based on whether the simulation is atomistic or continuous.
 * Initialises arrays for storing exchange interactions along each axis (x, y, z).
 * 
 * @param J The exchange constant
 */
void Simulation::add_exchange(double J) {

    // If the simulation is atomistic, set the rescaled exchange constants directly to J.
    if (atomistic == true) {
        this->J_rescaled[0] = J;
        this->J_rescaled[1] = J;
        this->J_rescaled[2] = J;
    } else { 
        // For the continuous case, rescale the exchange constants by the lattice spacing squared (rescaled J is A).
        this->J_rescaled[0] = J / (dx * dx);
        this->J_rescaled[1] = J / (dy * dy);
        this->J_rescaled[2] = J / (dz * dz);
    }

    // Store the original exchange constant for printing to file
    this->J = J;

    // Initialise the 4D arrays for the exchange term along each axis
    exchange_x = ContiguousArray4D(Nx + 1, Ny, Nz, 1);
    exchange_y = ContiguousArray4D(Nx, Ny + 1, Nz, 1);
    exchange_z = ContiguousArray4D(Nx, Ny, Nz + 1, 1);

    // Call the function to initialise exchange interactions at each lattice site
    initialise_exchange();
}


/**
 * @brief Adds a Dzyaloshinskii-Moriya interaction (DMI) term to the simulation.
 * 
 * Rescales the DMI constant D based on whether the simulation is atomistic or continuous.
 * Initialises arrays for storing DMI interactions along each axis (x, y, z).
 * 
 * @param D The DMI constant
 */
void Simulation::add_DMI(double D) {

    // If the simulation is atomistic, set the rescaled DMI constants directly to D.
    if (atomistic == true) {
        this->D_rescaled[0] = D;
        this->D_rescaled[1] = D;
        this->D_rescaled[2] = D;
    } else { 
        // For the continuous case, rescale the DMI constants by half of the lattice spacing.
        this->D_rescaled[0] = D / (2 * dx);
        this->D_rescaled[1] = D / (2 * dy);
        this->D_rescaled[2] = D / (2 * dz);
    }

    // Store the original DMI constant
    this->D = D;

    // Initialise the 4D arrays for the DMI term along each axis
    DMI_x = ContiguousArray4D(Nx + 1, Ny, Nz, 1);
    DMI_y = ContiguousArray4D(Nx, Ny + 1, Nz, 1);
    DMI_z = ContiguousArray4D(Nx, Ny, Nz + 1, 1);

    // Call the function to initialise DMI interactions at each lattice site
    initialise_DMI();
}


/**
 * @brief Initialises the Zeeman energy term for each lattice site in the simulation.
 * 
 * The function loops over each lattice site, calculating the initial Zeeman energy using the rescaled (atomistic or continuous) constants.
 * The calculated energies are stored in the 4D array "zeeman".
 */
void Simulation::initialise_zeeman() {
    // Loop over all the points in the x, y, and z dimensions of the lattice
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                
                // Calculate the Zeeman energy using the rescaled magnetic field and moment.
                // The calculated energy is stored in the 4D array "zeeman".
                zeeman.array_4D[i][j][k][0] = -mu0 * (
                    H_rescaled[0] * mu.array_4D[i][j][k][0] + 
                    H_rescaled[1] * mu.array_4D[i][j][k][1] + 
                    H_rescaled[2] * mu.array_4D[i][j][k][2]
                );
            }
        }
    }
}

/**
 * @brief Initialises the anisotropy energy term for each lattice site in the simulation.
 * 
 * The function loops over each lattice site, calculating the initial anisotropy energy based on the rescaled anisotropy constant (K_rescaled) 
 * and the unit vector u which specifies the direction of anisotropy. The calculated energies are stored in the 4D array "anisotropy".
 */
void Simulation::initialise_anisotropy() {
    // Loop over all lattice points in the x, y, and z dimensions
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                
                // Calculate the dot product of the unit vector u with the magnetic moment at each lattice point
                double dot_product = u[0] * mu.array_4D[i][j][k][0] + u[1] * mu.array_4D[i][j][k][1] + u[2] * mu.array_4D[i][j][k][2];
                
                // Calculate the anisotropy energy based on the sign of K_rescaled
                if (K_rescaled < 0) {
                    anisotropy.array_4D[i][j][k][0] = -K_rescaled * pow(dot_product, 2);
                } else {
                    // Note that we use a different reference level for the positive K_rescaled case
                    anisotropy.array_4D[i][j][k][0] = K_rescaled - K_rescaled * pow(dot_product, 2);
                }
            }
        }
    }
}

/**
 * @brief Initialises the exchange energy terms for interactions along the x, y, and z directions.
 * 
 * This function calculates and stores the exchange energy terms based on the rescaled exchange constant (J_rescaled) 
 * and the magnetic moments at neighboring lattice sites. The results are stored in 4D arrays "exchange_x", "exchange_y", and "exchange_z" 
 * to represent exchange interactions in the x, y, and z directions, respectively.
 */
void Simulation::initialise_exchange() 
{
    // Initialise the exchange energy term for interactions along the x-direction
    for (int i = 1; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                double dot_product_x = mu.array_4D[i-1][j][k][0] * mu.array_4D[i][j][k][0] 
                                     + mu.array_4D[i-1][j][k][1] * mu.array_4D[i][j][k][1] 
                                     + mu.array_4D[i-1][j][k][2] * mu.array_4D[i][j][k][2];
                exchange_x.array_4D[i][j][k][0] = -J_rescaled[0] * dot_product_x;

                // Add reference level in continuous models
                if (atomistic == false) 
                    exchange_x.array_4D[i][j][k][0] += J_rescaled[0];
            }
        }
    }

    // Initialise the exchange energy term for interactions along the y-direction
    for (int i = 0; i < Nx; i++) {
        for (int j = 1; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                double dot_product_y = mu.array_4D[i][j-1][k][0] * mu.array_4D[i][j][k][0] 
                                     + mu.array_4D[i][j-1][k][1] * mu.array_4D[i][j][k][1] 
                                     + mu.array_4D[i][j-1][k][2] * mu.array_4D[i][j][k][2];
                exchange_y.array_4D[i][j][k][0] = -J_rescaled[1] * dot_product_y;

                // Add reference level in continuous models
                if (atomistic == false) 
                    exchange_y.array_4D[i][j][k][0] += J_rescaled[1];
            }
        }
    }

    // Initialise the exchange energy term for interactions along the z-direction
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 1; k < Nz; k++) {
                double dot_product_z = mu.array_4D[i][j][k-1][0] * mu.array_4D[i][j][k][0] 
                                     + mu.array_4D[i][j][k-1][1] * mu.array_4D[i][j][k][1] 
                                     + mu.array_4D[i][j][k-1][2] * mu.array_4D[i][j][k][2];
                exchange_z.array_4D[i][j][k][0] = -J_rescaled[2] * dot_product_z;

                // Add reference level in continuous models
                if (atomistic == false) 
                    exchange_z.array_4D[i][j][k][0] += J_rescaled[2];
            }
        }
    }
}

/**
 * @brief Initialises the Dzyaloshinskii-Moriya interaction (DMI) energy terms for interactions along the x, y, and z directions.
 * 
 * This function calculates and stores the DMI energy terms based on the rescaled DMI constant (D_rescaled) 
 * and the magnetic moments at neighboring lattice sites. The results are stored in 4D arrays "DMI_x", "DMI_y", and "DMI_z" 
 * to represent DMI interactions in the x, y, and z directions, respectively.
 */
void Simulation::initialise_DMI() 
{
    // Initialise the DMI energy term for interactions along the x-direction
    for (int i = 1; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // Calculate the DMI for this i, j, k index along the x-axis
                DMI_x.array_4D[i][j][k][0] = -D_rescaled[0] * (mu.array_4D[i-1][j][k][1] * mu.array_4D[i][j][k][2] - mu.array_4D[i-1][j][k][2] * mu.array_4D[i][j][k][1]);
            }
        }
    }

    // Initialise the DMI energy term for interactions along the y-direction
    for (int i = 0; i < Nx; i++) {
        for (int j = 1; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // Calculate the DMI for this i, j, k index along the y-axis
                DMI_y.array_4D[i][j][k][0] = -D_rescaled[1] * (mu.array_4D[i][j-1][k][2] * mu.array_4D[i][j][k][0] - mu.array_4D[i][j-1][k][0] * mu.array_4D[i][j][k][2]);
            }
        }
    }

    // Initialise the DMI energy term for interactions along the z-direction
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 1; k < Nz; k++) {
                // Calculate the DMI for this i, j, k index along the z-axis
                DMI_z.array_4D[i][j][k][0] = -D_rescaled[2] * (mu.array_4D[i][j][k-1][0] * mu.array_4D[i][j][k][1] - mu.array_4D[i][j][k-1][1] * mu.array_4D[i][j][k][0]);
            }
        }
    }
}

/**
 * @brief Initialises the total energy of the magnetic system.
 *
 * This function calculates the total energy by summing up the Zeeman, anisotropy, exchange, and DMI energies. 
 * If `atomistic = false`, the energies are rescaled by the volume element dV before being saved to file, although
 * calculations are done wihout this rescaling to help prevent overflow.
 */
void Simulation::initialise_total_E() 
{
    total_E = 0; // Reset the total energy to zero

    // Sum up all Zeeman interaction energies
    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 0; k < Nz; k++) 
                total_E += zeeman.array_4D[i][j][k][0]; 

    // Sum up all anisotropy interaction energies
    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 0; k < Nz; k++) 
                total_E += anisotropy.array_4D[i][j][k][0]; 

    // Sum up all exchange interaction energies along x, y, and z directions
    for (int i = 0; i < Nx + 1; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 0; k < Nz; k++) 
                total_E += 2 * exchange_x.array_4D[i][j][k][0]; // Each energy must be counted twice

    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny + 1; j++) 
            for (int k = 0; k < Nz; k++) 
                total_E += 2 * exchange_y.array_4D[i][j][k][0];

    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 0; k < Nz + 1; k++) 
                total_E += 2 * exchange_z.array_4D[i][j][k][0];

    // Sum up all DMI interaction energies along x, y, and z directions
    for (int i = 1; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 0; k < Nz; k++) 
                total_E += 2 * DMI_x.array_4D[i][j][k][0]; // Each energy must be counted twice

    for (int i = 0; i < Nx; i++) 
        for (int j = 1; j < Ny; j++) 
            for (int k = 0; k < Nz; k++) 
                total_E += 2 * DMI_y.array_4D[i][j][k][0];

    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 1; k < Nz; k++) 
                total_E += 2 * DMI_z.array_4D[i][j][k][0];
}

/**
 * @brief Initialise random device with a seed based on the process ID.
 * 
 * @param id Process ID for seed generation.
 */
void Simulation::initialise_random_device(int id) {
    // Seed the random generator using a random device and the process ID
    gen.seed(rd() + id);
}

/**
 * @brief Generate a random double.
 * 
 * @return A random double value from the initialised random device.
 */
double Simulation::rand_double() {
    // Generate a random double between 0.0 and 1.0
    uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

/**
 * @brief Generate a random integer within a specified range.
 * 
 * @param min Minimum integer value.
 * @param max Maximum integer value.
 * @return A random integer between min and max from the initialised random device.
 */
int Simulation::rand_int(int min, int max) {
    // Generate a random integer between the given min and max (inclusive)
    uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

/**
 * @brief Propose a perturbation to the selected atom's spin.
 * 
 * Updates the mu_proposal array with the new spin orientation.
 */
void Simulation::rand_perturbation() {
    int x, y, z;
    x = atom_pos[0];
    y = atom_pos[1];
    z = atom_pos[2];

    // Uniform perturbation in alpha-search cone
    double theta = alpha * M_PI * rand_double();
    double phi = alpha * 2.0 * M_PI * rand_double();
    
    // Convert to Cartesian coordinates and add the perturbation
    proposed_mu[0] = mu.array_4D[x][y][z][0] + sin(theta) * cos(phi);
    proposed_mu[1] = mu.array_4D[x][y][z][1] + sin(theta) * sin(phi);
    proposed_mu[2] = mu.array_4D[x][y][z][2] + cos(theta);

    // Enforce normalization to make sure the proposed spin has unit length
    double proposed_norm = sqrt(pow(proposed_mu[0], 2) + pow(proposed_mu[1], 2) + pow(proposed_mu[2], 2));
    proposed_mu[0] /= proposed_norm;
    proposed_mu[1] /= proposed_norm;
    proposed_mu[2] /= proposed_norm;
}

/**
 * @brief Generate a random sample of atom indices.
 *
 * This function has two versions, controlled by the conditional directive FISHER_YATES.
 *
 * @note Version 1: Enabled with FISHER_YATES
 * Uses the Fisher-Yates shuffle to select N unique atom indices from 0 to N_colour-1.
 * This version is computationally efficient but modifies the order of the original list.
 *
 * @note Version 2: Disabled FISHER_YATES
 * Uses a set to store unique random integers, ensuring that all indices are unique.
 * This version may be slower if N is large and close to N_colour.
 *
 * @param N Number of atoms to sample (usually the number of processes used, p).
 * @param N_colour Total number of atoms of a given color (usually the number of red or black atoms, R_red or R_black).
 *
 * @return A vector containing the sampled atom indices.
 * @throws invalid_argument if N > N_colour
 */
#ifdef FISHER_YATES
    vector<int> Simulation::generate_random_sample(int N, int N_colour) {

        // Ensure the sample size isn't greater than the available pool
        if (N > N_colour) {
            throw invalid_argument("Sample size cannot be larger than the number of atoms of that colour!");
        }

        // Create a list of numbers to sample from
        vector<int> numbers(N_colour);
        iota(numbers.begin(), numbers.end(), 0);  // Fill with integers from 0 to N_colour - 1

        // Implement Fisher-Yates shuffle algorithm
        for (int i = 0; i < N; ++i) {
            int j = rand_int(i, N_colour - 1);
            swap(numbers[i], numbers[j]);
        }

        // Resize to the desired sample size
        numbers.resize(N);

        return numbers;
    }
#else
    vector<int> Simulation::generate_random_sample(int N, int N_colour) {

        // Ensure the sample size isn't greater than the available pool
        if (N > N_colour) {
            throw invalid_argument("Sample size cannot be larger than the number of atoms of that colour!");
        }

        // Using a set to ensure unique numbers
        set<int> sampled_numbers;

        // Keep adding unique numbers until we reach our sample size
        while (sampled_numbers.size() < N) {
            int random_number = rand_int(0, N_colour - 1);
            sampled_numbers.insert(random_number);
        }

        // Convert the set to a vector and return
        return vector<int>(sampled_numbers.begin(), sampled_numbers.end());
    }
#endif

/**
 * @brief Run the simulation based on the predefined parameters and options.
 * 
 * This function is responsible for the main logic of the simulation. It consists of the following steps:
 * 1. Initialization of random device, Timer stopwatches, energy tracker, atom list, and delta E. Creation of MPI_AtomData.
 * 2. Initialization of total energy.
 * 3. If TO_FILE is enabled, save all initial states to file.
 * 4. Execution of the main while loop.
 * 5. Free data types and, if VERBOSE is enabled, print summary.
 * 6. Save final states to file.
 *
 * It will loop through iterations, update the system state, check for stopping conditions, and manage other functionalities depending on the compile-time and run-time options provided.
 *
 * @throw invalid_argument Throws if the number of processes exceeds half of the total grid size `(Nx * Ny * Nz) / 2`.
 *
 * @note Conditonal directive 1: TIME_TEN_MILLION_UPDATES overwrites `it_max` with `(int)((1e7 + p - 1) / p)` to time for 1e7 updates.
 * @note Conditonal directive 2: DO_TIME_PROFILING enables time profiling for various portions of the code.
 * @note Conditonal directive 3: TO_FILE enables saving arrays to .dat file format.
 * @note Conditonal directive 4: VERBOSE prints updates and checkpoints at `it_max / 10` intervals.
 * @note Conditonal directive 5: STOPPING_CRITERION allows for early termination based on energy convergence.
 * @note [Conditonal directive 6: TEMPERATURE_PROFILE is a special mode for Curie temperature experiments.]
 */
void Simulation::run()
{            
    // Check if the number of processes is compatible with the problem size.
    if (id == 0) {
        if (p > (Nx * Ny * Nz) / 2) {
            throw invalid_argument("There are too many processes being used for this problem size!");
        }
    }
    
    // Initialise the random device based on the process ID.
    initialise_random_device(id);
    
    // Initialise the energy tracker.
    ContiguousArray4D energy_tracker(it_max, 1, 1, 1);
    
    // Initialise the array to store the list of chosen atoms.
    int atoms_list_array[p];
    
    // Initialise variables for energy calculations.
    double delta_E; // Variable to store the total change in energy due to proposals
    
    // Initialise the vector to store data related to all atoms.
    vector<AtomData> all_data(p); 
    
    // Initialise stopwatch timers
    Timer time_run;
    #ifdef TIME_PROFILING
        Timer time_custom;
        Timer time_initial_E;
        Timer time_atom_selection;
        Timer time_perturbation;
        Timer time_local;
        Timer time_atomdata;
        Timer time_communication;
        Timer time_update;
    #endif 

    // If timing 1e7 updates, override it_max
    #ifdef TIME_TEN_MILLION_UPDATES
        it_max = (int)((1e7 + p - 1) / p);
    #endif

    #ifdef TEMPERATURE_PROFILE
        ContiguousArray4D average_mu_xs(it_max, 1, 1, 1);
        T = 0.0;
    #endif

    // Start run time stopwatch
    if (id == 0)
        time_run.start();

    #ifdef DO_TIME_PROFILING
        if (id == 0)
        {   
            time_custom.start();
        }
    #endif

    // Create MPI_AtomData
    create_MPI_datatype();

    #ifdef DO_TIME_PROFILING
        if (id == 0)
        {
            time_custom.stop();
            time_initial_E.start();
        }
    #endif

    // Find system's total energy by summing over energy dnesity arrays
    initialise_total_E();

    #ifdef DO_TIME_PROFILING
        if (id == 0)
            time_initial_E.stop();
    #endif

    // Pause run time stopwatch for saving to file
    if (id == 0)
        time_run.stop();

    // Tab initial conditions if saving to file
    #ifdef TO_FILE
        if (id == 0)
        {
            mu.to_tab_file("mu_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            zeeman.to_tab_file("zeeman_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            anisotropy.to_tab_file("anisotropy_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            exchange_x.to_tab_file("exchange_x_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            exchange_y.to_tab_file("exchange_y_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            exchange_z.to_tab_file("exchange_z_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            DMI_x.to_tab_file("dmi_x_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            DMI_y.to_tab_file("dmi_y_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            DMI_z.to_tab_file("dmi_z_init", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
        }
    #endif

    // Restart run time stopwatch
    if (id == 0)
        time_run.start();
    
    // Main while loop
    while (it < it_max)
    {
        // Track 10% progress checkpoints
        #ifdef VERBOSE
            if (id == 0)
                if ((it + 1)%(it_max/10) == 0)
                    cout << "Iteration " << it + 1 << " reached ..." << endl;
        #endif
        
        // Update energy tracker with current system energy
        energy_tracker.array_4D[it][0][0][0] = total_E;

        #ifdef DO_TIME_PROFILING
            if (id == 0)
            {
                time_atom_selection.start();
            }
        #endif

        // Generate a vector of selected atoms on root process
        if (id == 0)
        {
            // Select atoms in the appropriate checkerboard 
            if (it%2 == 0) // Red iteration
                atoms_list = generate_random_sample(p, N_red);
            else // Black iteration
                atoms_list = generate_random_sample(p, N_black);

            // Read the vector into an array
            for (int i = 0; i < p; i++)
            {
                atoms_list_array[i] = atoms_list[i];
            }
        }

        #ifdef DO_TIME_PROFILING
            if (id == 0)
            {
                time_atom_selection.stop();
                time_communication.start();
            }
        #endif

        // Broadcast selected atoms from root process to all other processes
        MPI_Bcast(&atoms_list_array, p, MPI_INT, 0, MPI_COMM_WORLD);  

        #ifdef DO_TIME_PROFILING
            if (id == 0)
            {
                time_communication.stop();
            }
        #endif

        // Parallel region
        for (int i = 0; i < p; i++)
        {
            if (id == i)
            {   
                
                #ifdef DO_TIME_PROFILING
                    if (id == 0)
                    {
                        time_atom_selection.start();
                    }
                #endif
                
                // Find the corresponding atom on each process
                if (it%2 == 0) // Red iteration
                {
                    for (int j = 0; j < 3; j++)
                        atom_pos[j] = red_atoms[atoms_list_array[i]][j];
                } 
                else // Black iteration
                {
                    for (int j = 0; j < 3; j++)
                        atom_pos[j] = black_atoms[atoms_list_array[i]][j];
                }

                // Extract atom coordinates 
                int x, y, z;
                x = atom_pos[0];
                y = atom_pos[1];
                z = atom_pos[2];

                // Find current energies of the atom
                zeeman_E = zeeman.array_4D[x][y][z][0];
                anisotropy_E = anisotropy.array_4D[x][y][z][0];
                exchange_right_E = exchange_x.array_4D[x + 1][y][z][0];
                exchange_left_E = exchange_x.array_4D[x][y][z][0];
                exchange_up_E = exchange_y.array_4D[x][y + 1][z][0];
                exchange_down_E = exchange_y.array_4D[x][y][z][0];
                exchange_front_E = exchange_z.array_4D[x][y][z + 1][0];
                exchange_back_E = exchange_z.array_4D[x][y][z][0];
                DMI_right_E = DMI_x.array_4D[x + 1][y][z][0];
                DMI_left_E = DMI_x.array_4D[x][y][z][0];
                DMI_up_E = DMI_y.array_4D[x][y + 1][z][0];
                DMI_down_E = DMI_y.array_4D[x][y][z][0];
                DMI_front_E = DMI_z.array_4D[x][y][z + 1][0];
                DMI_back_E = DMI_z.array_4D[x][y][z][0];

                // ... and the energy contribution of this atom
                double local_E = (
                    zeeman_E +
                    anisotropy_E +
                    2 * exchange_right_E + // Multiply by two since this energy is the same for the neighbouring atom
                    2 * exchange_left_E +
                    2 * exchange_up_E +
                    2 * exchange_down_E +
                    2 * exchange_front_E +
                    2 * exchange_back_E +
                    2 * DMI_right_E +
                    2 * DMI_left_E +
                    2 * DMI_up_E +
                    2 * DMI_down_E +
                    2 * DMI_front_E +
                    2 * DMI_back_E
                );

                #ifdef DO_TIME_PROFILING
                    if (id == 0)
                    {
                        time_atom_selection.stop();
                        time_perturbation.start();
                    }
                #endif

                // Propose a perturbation
                rand_perturbation(); // Updates proposed_mu variable

                #ifdef DO_TIME_PROFILING
                    if (id == 0)
                    {
                        time_perturbation.stop();
                        time_local.start();
                    }  
                #endif

                // ... and find the associated energy change associated with this perturbation
                proposed_zeeman_E = -mu0 * (H_rescaled[0] * proposed_mu[0] + H_rescaled[1] * proposed_mu[1] + H_rescaled[2] * proposed_mu[2]);
                proposed_anisotropy_E = (K < 0) ? -K_rescaled * pow(u[0] * proposed_mu[0] + u[1] * proposed_mu[1] + u[2] * proposed_mu[2], 2) : K_rescaled - K_rescaled * pow(u[0] * proposed_mu[0] + u[1] * proposed_mu[1] + u[2] * proposed_mu[2], 2);
                if (atomistic==true)
                {
                    proposed_exchange_right_E = (x == Nx - 1) ? 0 : -J_rescaled[0] * (mu.array_4D[x + 1][y][z][0] * proposed_mu[0] + mu.array_4D[x + 1][y][z][1] * proposed_mu[1] + mu.array_4D[x + 1][y][z][2] * proposed_mu[2]);
                    proposed_exchange_left_E = (x == 0) ? 0 : -J_rescaled[0] * (mu.array_4D[x - 1][y][z][0] * proposed_mu[0] + mu.array_4D[x - 1][y][z][1] * proposed_mu[1] + mu.array_4D[x - 1][y][z][2] * proposed_mu[2]);
                    proposed_exchange_up_E = (y == Ny - 1) ? 0 : -J_rescaled[1] * (mu.array_4D[x][y + 1][z][0] * proposed_mu[0] + mu.array_4D[x][y + 1][z][1] * proposed_mu[1] + mu.array_4D[x][y + 1][z][2] * proposed_mu[2]);
                    proposed_exchange_down_E = (y == 0) ? 0 : -J_rescaled[1] * (mu.array_4D[x][y - 1][z][0] * proposed_mu[0] + mu.array_4D[x][y - 1][z][1] * proposed_mu[1] + mu.array_4D[x][y - 1][z][2] * proposed_mu[2]);
                    proposed_exchange_front_E = (z == Nz - 1) ? 0 : -J_rescaled[2] * (mu.array_4D[x][y][z + 1][0] * proposed_mu[0] + mu.array_4D[x][y][z + 1][1] * proposed_mu[1] + mu.array_4D[x][y][z + 1][2] * proposed_mu[2]);
                    proposed_exchange_back_E = (z == 0) ? 0 : -J_rescaled[2] * (mu.array_4D[x][y][z - 1][0] * proposed_mu[0] + mu.array_4D[x][y][z - 1][1] * proposed_mu[1] + mu.array_4D[x][y][z - 1][2] * proposed_mu[2]);
                }
                else // Add reference level for exchange
                {
                    proposed_exchange_right_E = (x == Nx - 1) ? 0 : J_rescaled[0] - J_rescaled[0] * (mu.array_4D[x + 1][y][z][0] * proposed_mu[0] + mu.array_4D[x + 1][y][z][1] * proposed_mu[1] + mu.array_4D[x + 1][y][z][2] * proposed_mu[2]);
                    proposed_exchange_left_E = (x == 0) ? 0 : J_rescaled[0] - J_rescaled[0] * (mu.array_4D[x - 1][y][z][0] * proposed_mu[0] + mu.array_4D[x - 1][y][z][1] * proposed_mu[1] + mu.array_4D[x - 1][y][z][2] * proposed_mu[2]);
                    proposed_exchange_up_E = (y == Ny - 1) ? 0 : J_rescaled[1] - J_rescaled[1] * (mu.array_4D[x][y + 1][z][0] * proposed_mu[0] + mu.array_4D[x][y + 1][z][1] * proposed_mu[1] + mu.array_4D[x][y + 1][z][2] * proposed_mu[2]);
                    proposed_exchange_down_E = (y == 0) ? 0 : J_rescaled[1] - J_rescaled[1] * (mu.array_4D[x][y - 1][z][0] * proposed_mu[0] + mu.array_4D[x][y - 1][z][1] * proposed_mu[1] + mu.array_4D[x][y - 1][z][2] * proposed_mu[2]);
                    proposed_exchange_front_E = (z == Nz - 1) ? 0 : J_rescaled[2] - J_rescaled[2] * (mu.array_4D[x][y][z + 1][0] * proposed_mu[0] + mu.array_4D[x][y][z + 1][1] * proposed_mu[1] + mu.array_4D[x][y][z + 1][2] * proposed_mu[2]);
                    proposed_exchange_back_E = (z == 0) ? 0 : J_rescaled[2] - J_rescaled[2] * (mu.array_4D[x][y][z - 1][0] * proposed_mu[0] + mu.array_4D[x][y][z - 1][1] * proposed_mu[1] + mu.array_4D[x][y][z - 1][2] * proposed_mu[2]);
                }
                proposed_DMI_right_E = (x == Nx - 1) ? 0 : D_rescaled[0] * (mu.array_4D[x + 1][y][z][1] * proposed_mu[2] - mu.array_4D[x + 1][y][z][2] * proposed_mu[1]);
                proposed_DMI_left_E = (x == 0) ? 0 : - D_rescaled[0] * (mu.array_4D[x - 1][y][z][1] * proposed_mu[2] - mu.array_4D[x - 1][y][z][2] * proposed_mu[1]);
                proposed_DMI_up_E = (y == Ny - 1) ? 0 : D_rescaled[1] * (mu.array_4D[x][y + 1][z][2] * proposed_mu[0] - mu.array_4D[x][y + 1][z][0] * proposed_mu[2]);
                proposed_DMI_down_E = (y == 0) ? 0 : - D_rescaled[1] * (mu.array_4D[x][y - 1][z][2] * proposed_mu[0] - mu.array_4D[x][y - 1][z][0] * proposed_mu[2]);
                proposed_DMI_front_E = (z == Nz - 1) ? 0 : D_rescaled[2] * (mu.array_4D[x][y][z + 1][0] * proposed_mu[1] - mu.array_4D[x][y][z + 1][1] * proposed_mu[0]);
                proposed_DMI_back_E = (z == 0) ? 0 :  - D_rescaled[2] * (mu.array_4D[x][y][z - 1][0] * proposed_mu[1] - mu.array_4D[x][y][z - 1][1] * proposed_mu[0]);

                // ... and find their sum
                double proposed_local_E = (
                    proposed_zeeman_E +
                    proposed_anisotropy_E +
                    2 * proposed_exchange_right_E + // Multiply by two since this energy is the same for the neighbouring atom
                    2 * proposed_exchange_left_E +
                    2 * proposed_exchange_up_E +
                    2 * proposed_exchange_down_E +
                    2 * proposed_exchange_front_E +
                    2 * proposed_exchange_back_E +
                    2 * proposed_DMI_right_E +
                    2 * proposed_DMI_left_E +
                    2 * proposed_DMI_up_E +
                    2 * proposed_DMI_down_E +
                    2 * proposed_DMI_front_E +
                    2 * proposed_DMI_back_E
                );

                #ifdef DO_TIME_PROFILING
                    if (id == 0)
                    {
                        time_local.stop();
                        time_atomdata.start();
                    }
                #endif

                // Accept or reject based on Botlzmann factor
                double local_delta_E = proposed_local_E - local_E;
                double r = rand_double(); 

                if ((local_delta_E < 0.0) || (r < exp(-local_delta_E / (kB * T)))) {
                    // If accepted, package proposed data in AtomData object
                    is_updated = true;
                    atom_data.atom_pos[0] = x;
                    atom_data.atom_pos[1] = y;
                    atom_data.atom_pos[2] = z;
                    atom_data.new_mu[0] = proposed_mu[0];
                    atom_data.new_mu[1] = proposed_mu[1];
                    atom_data.new_mu[2] = proposed_mu[2];
                    atom_data.local_delta_E = local_delta_E;
                    atom_data.zeeman_E = proposed_zeeman_E;
                    atom_data.anisotropy_E = proposed_anisotropy_E;
                    atom_data.exchange_right_E = proposed_exchange_right_E;
                    atom_data.exchange_left_E = proposed_exchange_left_E;
                    atom_data.exchange_up_E = proposed_exchange_up_E;
                    atom_data.exchange_down_E = proposed_exchange_down_E;
                    atom_data.exchange_front_E = proposed_exchange_front_E;
                    atom_data.exchange_back_E = proposed_exchange_back_E;
                    atom_data.DMI_right_E = proposed_DMI_right_E;
                    atom_data.DMI_left_E = proposed_DMI_left_E;
                    atom_data.DMI_up_E = proposed_DMI_up_E;
                    atom_data.DMI_down_E = proposed_DMI_down_E;
                    atom_data.DMI_front_E = proposed_DMI_front_E;
                    atom_data.DMI_back_E = proposed_DMI_back_E;
                } else {
                    // If rejected, package old data in AtomData object
                    is_updated = false;
                    atom_data.atom_pos[0] = x;
                    atom_data.atom_pos[1] = y;
                    atom_data.atom_pos[2] = z;
                    atom_data.new_mu[0] = mu.array_4D[x][y][z][0];
                    atom_data.new_mu[1] = mu.array_4D[x][y][z][1];
                    atom_data.new_mu[2] = mu.array_4D[x][y][z][2];
                    atom_data.local_delta_E = 0.0;
                    atom_data.zeeman_E = zeeman_E;
                    atom_data.anisotropy_E = anisotropy_E;
                    atom_data.exchange_right_E = exchange_right_E;
                    atom_data.exchange_left_E = exchange_left_E;
                    atom_data.exchange_up_E = exchange_up_E;
                    atom_data.exchange_down_E = exchange_down_E;
                    atom_data.exchange_front_E = exchange_front_E;
                    atom_data.exchange_back_E = exchange_back_E;
                    atom_data.DMI_right_E = DMI_right_E;
                    atom_data.DMI_left_E = DMI_left_E;
                    atom_data.DMI_up_E = DMI_up_E;
                    atom_data.DMI_down_E = DMI_down_E;
                    atom_data.DMI_front_E = DMI_front_E;
                    atom_data.DMI_back_E = DMI_back_E;
                }

                #ifdef DO_TIME_PROFILING
                    if (id == 0)
                    {
                        time_atomdata.stop();
                    }
                #endif
            }
        }

        #ifdef DO_TIME_PROFILING
            if (id == 0)
            {
                time_communication.start();
            }
        #endif

        // All gather to get all information on all processes 
        MPI_Allgather(&atom_data, 1, MPI_AtomData, all_data.data(), 1, MPI_AtomData, MPI_COMM_WORLD);

        #ifdef DO_TIME_PROFILING
            if (id == 0)
            {                            
                time_communication.stop();
                time_update.start();
            }
        #endif

        // Reset change in energy to zero for this iteration
        delta_E = 0.0; 

        // Loop over all selected atoms
        for (int i = 0; i < p; i++) // Non-split communication
        {            
            // Extract AtomData
            AtomData atom_data_update = all_data[i];
            
            // If no energy change, then skip to next
            if (atom_data_update.local_delta_E == 0)
                continue;

            // Extract atom coordinates
            int x = atom_data_update.atom_pos[0];
            int y = atom_data_update.atom_pos[1];
            int z = atom_data_update.atom_pos[2];

            // Update spin
            mu.array_4D[x][y][z][0] = atom_data_update.new_mu[0];
            mu.array_4D[x][y][z][1] = atom_data_update.new_mu[1];
            mu.array_4D[x][y][z][2] = atom_data_update.new_mu[2];

            // Update the energy arrays
            zeeman.array_4D[x][y][z][0] = atom_data_update.zeeman_E;
            anisotropy.array_4D[x][y][z][0] = atom_data_update.anisotropy_E;
            if(x != Nx - 1) exchange_x.array_4D[x + 1][y][z][0] = atom_data_update.exchange_right_E;
            if(x != 0) exchange_x.array_4D[x][y][z][0] = atom_data_update.exchange_left_E;
            if(y != Ny - 1) exchange_y.array_4D[x][y + 1][z][0] = atom_data_update.exchange_up_E;
            if(y != 0) exchange_y.array_4D[x][y][z][0] = atom_data_update.exchange_down_E;
            if(z != Nz - 1) exchange_z.array_4D[x][y][z + 1][0] = atom_data_update.exchange_front_E;
            if(z != 0) exchange_z.array_4D[x][y][z][0] = atom_data_update.exchange_back_E;
            if(x != Nx - 1) DMI_x.array_4D[x + 1][y][z][0] = atom_data_update.DMI_right_E;
            if(x != 0) DMI_x.array_4D[x][y][z][0] = atom_data_update.DMI_left_E;
            if(y != Ny - 1) DMI_y.array_4D[x][y + 1][z][0] = atom_data_update.DMI_up_E;
            if(y != 0) DMI_y.array_4D[x][y][z][0] = atom_data_update.DMI_down_E;
            if(z != Nz - 1) DMI_z.array_4D[x][y][z + 1][0] = atom_data_update.DMI_front_E;
            if(z != 0) DMI_z.array_4D[x][y][z][0] = atom_data_update.DMI_back_E;

            // Calculate total change in energy due to contribution from each update
            delta_E += atom_data_update.local_delta_E;
        }

        // Update the total system energy
        total_E += delta_E;

        #ifdef DO_TIME_PROFILING
            if (id == 0)
                time_update.stop();
        #endif

        // Barrier to synchronise all processes
        MPI_Barrier(MPI_COMM_WORLD);

        #ifdef STOPPING_CRITERION // Stopping criterion version
            // Stops if the change in energy relative to 10000 iterations ago is less than rtol of the original energy
            if (it > 10000 && fabs((energy_tracker.array_4D[it][0][0][0] - energy_tracker.array_4D[it - 10000][0][0][0]) / energy_tracker.array_4D[0][0][0][0]) < rtol)
            {
                #ifdef VERBOSE
                    if(id == 0)
                        cout << "Simulation stopped evolving at it = " << it << " < " << it_max << " = it_max reached at rtol = " << rtol << " after " << p * it << " updates ..." << endl;
                #endif
                break;
            }
        #endif

        // [Temperautre profiling logic]
        #ifdef TEMPERATURE_PROFILE
            if (id == 0)
            {
                double average_mu_x;
                for (int i = 0; i < Nx; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            average_mu_x += mu.array_4D[i][j][k][0];
                        }
                    }
                }
                average_mu_x /= Nx * Ny * Nz;
                average_mu_xs.array_4D[it][0][0][0] = average_mu_x;
            }
            T += 1e20; // Evolve T
        #endif

        // Decay alpha exponentially
        if (alpha_decay == true)
            alpha *= 1 - 1/it_max;

        // Increment the iteration
        it ++;

        // If final itetation, inform user if verbose 
        #ifdef VERBOSE
            if (id == 0)
                if (it == it_max)
                    cout << "Maximum number of iterations reached after " << p * it << " updates ..." << endl;
        #endif


    } // End of main while loop

    #ifdef DO_TIME_PROFILING
        if (id == 0)
        {
            time_custom.start();
        }
    #endif

    // Free MPI_AtomData
    MPI_Type_free(&MPI_AtomData);

    #ifdef DO_TIME_PROFILING
        if (id == 0)
        {
            time_custom.stop();
        }
    #endif

    // End of run time
    if (id == 0)
        time_run.stop();

    // Return key statistics
    if (id == 0)
    {
        #ifdef VERBOSE
            cout << "With " << p << " processes on a " << Nx << " x " << Ny << " x " << Nz << " system the simulation took " << time_run.elapsed_time() << "s to run ...";
        #else
            cout << p << "\t" << Nx << "\t" << Ny << "\t" << Nz << "\t" << it << "\t" << time_run.elapsed_time();
        #endif
        // ... and time profiling
        #ifdef DO_TIME_PROFILING
            cout  << "\t" << time_custom.elapsed_time() << "\t"
            << time_initial_E.elapsed_time() << "\t"
            << time_atom_selection.elapsed_time() << "\t"
            << time_perturbation.elapsed_time() << "\t"
            << time_local.elapsed_time() << "\t"
            << time_atomdata.elapsed_time() << "\t"
            << time_communication.elapsed_time() << "\t"
            << time_update.elapsed_time();
        #endif
    cout << endl;

    // Write final states to file
    #ifdef TO_FILE
        if (id == 0)
        {
            #ifdef VERBOSE
                cout << "Saving results to file ..." << endl;
            #endif
            
            mu.to_tab_file("mu_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            zeeman.to_tab_file("zeeman_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            anisotropy.to_tab_file("anisotropy_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            exchange_x.to_tab_file("exchange_x_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            exchange_y.to_tab_file("exchange_y_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            exchange_z.to_tab_file("exchange_z_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            DMI_x.to_tab_file("dmi_x_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            DMI_y.to_tab_file("dmi_y_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            DMI_z.to_tab_file("dmi_z_end", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            if (atomistic==true)
                energy_tracker.to_tab_file("energy_data", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            else // Rescale energies by dV; best to do this at the end for precision since dV is a small number 
            {
                double dV = dx * dy * dz;
                for (int i = 0; i < it; i ++)
                    energy_tracker.array_4D[i][0][0][0] = dV * energy_tracker.array_4D[i][0][0][0];
                energy_tracker.to_tab_file("energy_data", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
            } 
        }
    #endif

    #ifdef TEMPERATURE_PROFILE
        if (id == 0)
            average_mu_xs.to_tab_file("average_mu_x", Nx, Ny, Nz, Lx, Ly, Lz, Ms, T, it, H, K, u, J, D, p, atomistic);
    #endif

    // Simulation complete!
    #ifdef VERBOSE
        if (id == 0)
            cout << "Simulation complete!" << endl;
    #endif
    }
};
