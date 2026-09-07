/******************************************************************************

  Toroidal Fourier preconditioner for the 3-D M3D-C1 PETSc solver.

  The preconditioner replaces a plane-varying matrix by its toroidal block-
  circulant average, diagonalizes that average with a discrete Fourier
  transform, and solves one coupled 2-D system per Fourier mode.

*******************************************************************************/
#ifdef M3DC1_PETSC
#ifndef M3DC1_TOROIDAL_FOURIER_H
#define M3DC1_TOROIDAL_FOURIER_H

#include "m3dc1_scorec.h"
#include "petscksp.h"

PetscErrorCode m3dc1_configure_toroidal_fourier_pc(
    KSP ksp, Mat A, PetscInt matrix_id);

#endif
#endif
