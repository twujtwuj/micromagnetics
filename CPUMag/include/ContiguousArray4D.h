// Thomas William Unitt-Jones (ACSE-twu18)
/**
 * @file ContiguousArray4D.h
 * @brief Defines the ContiguousArray4D class for managing contiguous arrays with dimensions up to 4D.
 */

#ifndef CONTIGUOUS_ARRAY_4D_H
#define CONTIGUOUS_ARRAY_4D_H

#include <fstream>
#include <iostream>
#include <string>

/**
 * @class ContiguousArray4D
 * @brief Manages contiguous arrays with dimensions up to 4D.
 *
 * The ContiguousArray4D class provides a flexible structure for managing contiguous arrays
 * of different dimensions (1D, 2D, 3D, 4D) and allows subscripting to access elements conveniently.
 * It also provides a method to save these arrays to a file along with relevant metadata for readability
 * in Python or other languages.
 */
class ContiguousArray4D
{
public:
    double* array_1D; ///< Pointer to a 1D array
    double** array_2D; ///< Pointer to a 2D array
    double*** array_3D; ///< Pointer to a 3D array
    double**** array_4D; ///< Pointer to a 4D array
    int nx, ny, nz, nv; ///< Dimensions of the array

    /**
     * @brief Default constructor for ContiguousArray4D.
     */
    ContiguousArray4D();

    /**
     * @brief Constructor for ContiguousArray4D.
     * @param x Dimension along the x-axis.
     * @param y Dimension along the y-axis.
     * @param z Dimension along the z-axis.
     * @param v Dimension along the v-axis.
     */
    ContiguousArray4D(int x, int y, int z, int v);

    /**
     * @brief Destructor for ContiguousArray4D.
     */
    ~ContiguousArray4D();

    /**
     * @brief Save the array to a tab-delimited file with metadata.
     * @param file_name The name of the output file.
     * @param Nx Number of nodes along the x-axis of system (not necessarily the dimension of the array).
     * @param Ny Number of nodes along the y-axis of system (").
     * @param Nz Number of nodes along the z-axis of system (").
     * @param Lx Physical length along the x-axis of system (").
     * @param Ly Physical length along the y-axis of system (").
     * @param Lz Physical length along the z-axis of system (").
     * @param Ms Saturation magnetisation.
     * @param T Temperature.
     * @param it Iteration number.
     * @param H External magnetic field [Hx, Hy, Hz].
     * @param K Uniaxial anisotropy constant.
     * @param u Uniaxial anisotropy direction [ux, uy, uz].
     * @param J Exchange constant.
     * @param D DMI constant.
     * @param procs Number of processes.
     * @param atomistic Flag indicating interpretation of constants.
     */
    void to_tab_file(std::string file_name, int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double Ms, double T, int it, double H[3], double K, double u[3], double J, double D, int procs, bool atomistic);
};

#endif // CONTIGUOUS_ARRAY_4D_H
