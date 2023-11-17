// Thomas William Unitt-Jones (ACSE-twu18)
// ContiguousArray4D.cpp
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "ContiguousArray4D.h"

using namespace std;

/**
 * @brief Default constructor for the ContiguousArray4D class.
 * Initialises array pointers to nullptr.
 */
ContiguousArray4D::ContiguousArray4D()
    : array_1D(nullptr), array_2D(nullptr), array_3D(nullptr), array_4D(nullptr) {}

/**
 * @brief Constructor for the ContiguousArray4D class.
 * @param x Dimension along the x-axis.
 * @param y Dimension along the y-axis.
 * @param z Dimension along the z-axis.
 * @param v Dimension along the v-axis.
 */
ContiguousArray4D::ContiguousArray4D(int x, int y, int z, int v)
    : nx(x), ny(y), nz(z), nv(v)
{
    array_1D = new double[nx * ny * nz * nv];
    array_2D = new double*[nx * ny * nz];
    array_3D = new double**[nx * ny];
    array_4D = new double***[nx];

    for (int i = 0; i < nx; i++)
    {
        array_4D[i] = &array_3D[i * ny];
        for (int j = 0; j < ny; j++)
        {
            array_4D[i][j] = &array_2D[(i * ny + j) * nz];
            for (int k = 0; k < nz; k++)
            {
                array_4D[i][j][k] = &array_1D[((i * ny + j) * nz + k) * nv];
            }
        }
    }
}

/**
 * @brief Save the array to a tab-delimited file with metadata in ((x, y), (z, v)) order.
 * @param file_name The name of the output file.
 * @param Nx Number of nodes along the x-axis of the system (not necessarily the dimension of the array).
 * @param Ny Number of nodes along the y-axis of the system (").
 * @param Nz Number of nodes along the z-axis of the system (").
 * @param Lx Physical length along the x-axis of the system (").
 * @param Ly Physical length along the y-axis of the system (").
 * @param Lz Physical length along the z-axis of the system (").
 * @param Ms Saturation magnetization.
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
void ContiguousArray4D::to_tab_file(string file_name, int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double Ms, double T, int it, double H[3], double K, double u[3], double J, double D, int procs, bool atomistic) 
{
    string fname = "./out/" + file_name + ".dat";

    fstream fout;
    fout.open(fname, ios::out);

    if (fout.fail())
    {
        cout << "Error opening file!" << endl;
        cout.flush();
        //exit(0);
    }

    ostringstream out;
    out << setprecision(15);  // Sets precision to 15 decimal places

    // Processes
    out << procs;
    fout << out.str() << endl;
    out.str("");

    // Iteration
    out << it;
    fout << out.str() << endl;
    out.str("");

    // Atomistic or continuous interpretation of magnetic constants
    out << (atomistic ? 1 : 0);
    fout << out.str() << endl;
    out.str("");

    // Node dimension of system
    out << Nx << "\t" << Ny << "\t" << Nz;
    fout << out.str() << endl;
    out.str("");
    
    // Array dimensions (member variables)
    out << nx << "\t" << ny << "\t" << nz << "\t" << nv;
    fout << out.str() << endl;
    out.str("");

    // Physical dimensions of system
    out << Lx << "\t" << Ly << "\t" << Lz;
    fout << out.str() << endl;
    out.str("");

    // Magnetic saturation
    out << Ms;
    fout << out.str() << endl;
    out.str(""); 

    // Temperature
    out << T;
    fout << out.str() << endl;
    out.str(""); 

    // Zeeman
    out << H[0] << "\t" << H[1] << "\t" << H[2];
    fout << out.str() << endl;
    out.str("");

    // Anisotropy
    out << K;
    fout << out.str() << endl;
    out.str("");
    out << u[0] << "\t" << u[1] << "\t" << u[2];
    fout << out.str() << endl;
    out.str(""); 

    // Exchange
    out << J;
    fout << out.str() << endl;
    out.str("");

    // DMI
    out << D;
    fout << out.str() << endl;
    out.str("");

    // ((x, y), (z, v)) order
    for (int i = 0; i < nx; i++) 
        for (int j = 0; j < ny; j++) 
        {
            for (int k = 0; k < nz; k++) 
                for (int l = 0; l < nv; l++)
                {
                    out << array_4D[i][j][k][l] << "\t";
                    fout << out.str();
                    out.str("");
                }
            fout << endl; // New line for new (x, y) values
        }
    fout.close();
};

