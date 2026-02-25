/* 2D heat equation

   Authors: Jussi Enkovaara, ...
   Copyright (C) 2014  CSC - IT Center for Science Ltd.

   Licensed under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   Copy of the GNU General Public License can be onbtained from
   see <http://www.gnu.org/licenses/>.
*/

/**
 * @file heat_mpi.h
 * @author Jussi Enkovaara
 * @brief Header file for 2D heat equation
 *
 * Defines a data structure for temperature field and for parallelization
 * information
 */

/**
 * @brief Datatype for temperature field
 */
typedef struct {
    int nx;          //!< local x-dimension of the field (excluding halo)
    int ny;          //!< local y-dimension of the field (excluding halo)
    int nx_full;     //!< global x-dimension of the field
    int ny_full;     //!< global y-dimension of the field
    double dx;       //!< Grid spacing in x-dimension
    double dy;       //!< Grid spacing in y-dimension
    double dx2;      //!< Square of grid spacing in x-dimension
    double dy2;      //!< Square of grid spacing in x-dimension
    /** @brief 2d data array
     *
     *  The actual data array. Array contains also the boundary layers
     *  so its dimensions are (nx+2) x (ny + x)
     */
    double **data; 
    MPI_Win rma_window; //!< RMA access window for one-sided communication
} field;

/**
 * @brief Datatype for parallelization information
 */
typedef struct {
    int size;        //!< Number of MPI tasks
    int rank;        //!< Rank of this MPI task
    int nup, ndown, nleft, nright;   //!< Ranks of neighbouring MPI tasks
    MPI_Comm comm;              //!< MPI communicator
    MPI_Datatype rowtype;       //!< MPI Datatype for communication of rows
    MPI_Datatype columntype;    //!< MPI Datatype for communication of columns
    MPI_Datatype subarraytype;  //!< MPI Datatype for communication of inner region
} parallel_data;

// We use here fixed grid spacing
#define DX 0.01
#define DY 0.01

/** @brief Utility function for allocating 2d array */
double **malloc_2d(int nx, int ny);

/** @brief Utility function for deallocating 2d array */
void free_2d(double **array);

void initialize_field_metadata(field * temperature, int nx, int ny,
                               parallel_data * parallel);

void parallel_initialize(parallel_data * parallel, int nx, int ny);

void initialize(field * temperature1, field * temperature2,
                parallel_data * parallel);

void evolve(field * curr, field * prev, double a, double dt);

void exchange(field * temperature, parallel_data * parallel);

void output(field * temperature, int iter, parallel_data * parallel);

void read_input(field * temperature1, field * temperature2, char *filename,
                parallel_data * parallel);

void copy_field(field * temperature1, field * temperature2);

void swap_fields(field * temperature1, field * temperature2);

void finalize(field * temperature1, field * temperature2,
              parallel_data * parallel);
