// Thomas William Unitt-Jones (ACSE-twu18)
/**
 * @file AtomData.h
 * @brief Defines the AtomData struct and its members for encapsulating atom data.
 */

#ifndef ATOMDATA_H
#define ATOMDATA_H

/**
 * @struct AtomData
 * @brief Encapsulates data about an atom, including position, spin, energy change, and energy terms.
 *
 * This struct defines the AtomData structure used to hold information about an atom.
 * AtomData objects are designed to be sent as a custom MPI data type MPI_AtomData.
 */
struct AtomData {
    int atom_pos[3]; ///< Position of atom [x, y, z]
    double new_mu[3]; ///< Accepted magnetic moment of atom [mx, my, mz]
    double local_delta_E; ///< Energy change of this atom due to pertubation or lackthereof
    double zeeman_E; ///< Zeeman
    double anisotropy_E; ///< Anisotropy
    double exchange_right_E; ///< Exchange; right neighbour
    double exchange_left_E; ///< Exchange; left neighbour
    double exchange_up_E; ///< Exchange; upper neighbour
    double exchange_down_E; ///< Exchange; lower neighbour
    double exchange_front_E; ///< Exchange; front neighbour
    double exchange_back_E; ///< Exchange; back neighbour
    double DMI_right_E; ///< DMI; right neighbour
    double DMI_left_E; ///< DMI; left neighbour
    double DMI_up_E; ///< DMI; upper neighbour
    double DMI_down_E; ///< DMI; lower neighbour
    double DMI_front_E; ///< DMI; front neighbour
    double DMI_back_E; ///< DMI; back neighbour
};

#endif // ATOMDATA_H
