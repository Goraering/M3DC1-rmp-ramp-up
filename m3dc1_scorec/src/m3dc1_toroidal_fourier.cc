/******************************************************************************

  Toroidal Fourier preconditioner for a plane-major M3D-C1 matrix.

  For plane blocks A_{p,p+s}, form the toroidal average K_s and the symbol

      Ahat(k) = sum_s exp(+i 2*pi*k*s/nplane) K_s .

  Each physical plane process group is reused as a Fourier-mode process
  group: plane p owns mode k=p.  Processes at the same poloidal partition on
  every plane exchange vector data through column_comm.  Every unknown and
  field coupling in the selected matrix is retained in every mode matrix.

  The 3-D executable uses a real PETSc build.  A complex mode matrix H+iG is
  therefore represented by the real matrix [ H -G ; G H ], with real and
  imaginary components interleaved.  A native complex path is also provided
  for completeness.

*******************************************************************************/
#ifdef M3DC1_PETSC

#include "m3dc1_toroidal_fourier.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct TFKey {
  PetscInt row;
  PetscInt col;
  PetscInt shift;

  bool operator<(const TFKey &other) const {
    if (row != other.row)
      return row < other.row;
    if (col != other.col)
      return col < other.col;
    return shift < other.shift;
  }
};

struct TFEntry {
  TFKey key;
  PetscScalar value;
};

struct TFEntryLess {
  bool operator()(const TFEntry &a, const TFEntry &b) const {
    return a.key < b.key;
  }
};

static bool TFSameKey(const TFKey &a, const TFKey &b) {
  return a.row == b.row && a.col == b.col && a.shift == b.shift;
}

struct ToroidalFourierPC {
  ToroidalFourierPC()
      : A(NULL), mode_mat(NULL), mode_ksp(NULL), mode_rhs(NULL),
        mode_sol(NULL), column_comm(MPI_COMM_NULL),
        mode_comm(MPI_COMM_NULL), nplane(0), plane_id(0), npart(0),
        partition_id(0), local_dim(0), plane_dim(0), matrix_id(-1),
        matrix_state(0),
        rebuild(PETSC_FALSE), check_uniform(PETSC_TRUE), debug(PETSC_FALSE),
        timing(PETSC_FALSE), ready(PETSC_FALSE),
        vector_sizes_checked(PETSC_FALSE), apply_count(0) {}

  Mat A;
  Mat mode_mat;
  KSP mode_ksp;
  Vec mode_rhs;
  Vec mode_sol;
  MPI_Comm column_comm;
  MPI_Comm mode_comm;
  PetscInt nplane;
  PetscInt plane_id;
  PetscInt npart;
  PetscInt partition_id;
  PetscInt local_dim;
  PetscInt plane_dim;
  PetscInt matrix_id;
  PetscObjectState matrix_state;
  PetscBool rebuild;
  PetscBool check_uniform;
  PetscBool debug;
  PetscBool timing;
  PetscBool ready;
  PetscBool vector_sizes_checked;
  PetscInt apply_count;
  std::string options_prefix;
  std::vector<PetscScalar> plane_values;
  std::vector<PetscReal> phase_cos;
  std::vector<PetscReal> phase_sin;
};

static PetscErrorCode TFDestroyModeObjects(ToroidalFourierPC *ctx) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->mode_rhs));
  PetscCall(VecDestroy(&ctx->mode_sol));
  PetscCall(KSPDestroy(&ctx->mode_ksp));
  PetscCall(MatDestroy(&ctx->mode_mat));
  ctx->ready = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TFDestroy(PC pc) {
  ToroidalFourierPC *ctx = NULL;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  if (!ctx)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TFDestroyModeObjects(ctx));
  if (ctx->column_comm != MPI_COMM_NULL)
    PetscCallMPI(MPI_Comm_free(&ctx->column_comm));
  if (ctx->mode_comm != MPI_COMM_NULL)
    PetscCallMPI(MPI_Comm_free(&ctx->mode_comm));
  delete ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TFView(PC pc, PetscViewer viewer) {
  ToroidalFourierPC *ctx = NULL;
  PetscBool is_ascii = PETSC_FALSE;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII,
                                   &is_ascii));
  if (ctx && is_ascii) {
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "  toroidal Fourier block-circulant PC: matrix=%" PetscInt_FMT
        ", planes=%" PetscInt_FMT ", coupled dofs/mode=%" PetscInt_FMT
        ", all matrix couplings retained, options prefix=%s\n",
        ctx->matrix_id, ctx->nplane, ctx->plane_dim,
        ctx->options_prefix.c_str()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TFSetUp(PC pc) {
  ToroidalFourierPC *ctx = NULL;
  Mat A = NULL;
  PetscInt global_dim, global_cols, local_rows, local_cols;
  PetscInt rstart, rend, cstart, cend;
  PetscInt local_ok = 1, global_ok = 1;
  PetscMPIInt world_rank, world_size, column_rank, column_size, mpi_nentry;
  PetscBool assembled = PETSC_FALSE, has_get_row = PETSC_FALSE;
  int comm_comparison = MPI_UNEQUAL;
  int nplane_api = 0, plane_id_api = 0;
  double timing_setup_start = 0.0;
  double timing_extract_start = 0.0, timing_extract = 0.0;
  double timing_mode_matrix_start = 0.0, timing_mode_matrix = 0.0;
  double timing_mode_ksp_start = 0.0, timing_mode_ksp = 0.0;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_NULL,
             "Toroidal Fourier PC context is NULL");
  if (ctx->timing)
    timing_setup_start = MPI_Wtime();
  PetscCall(PCGetOperators(pc, &A, NULL));
  PetscCheck(A, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_NULL,
             "Toroidal Fourier PC has no operator");
  ctx->A = A;
  PetscCall(TFDestroyModeObjects(ctx));
  ctx->apply_count = 0;
  ctx->vector_sizes_checked = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)A),
                                PETSC_COMM_WORLD, &comm_comparison));
  PetscCheck(comm_comparison == MPI_IDENT || comm_comparison == MPI_CONGRUENT,
             PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP,
             "-fouriersolve matrix %" PetscInt_FMT
             " must use a communicator congruent with PETSC_COMM_WORLD",
             ctx->matrix_id);
  PetscCall(MatAssembled(A, &assembled));
  PetscCheck(assembled, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
             "-fouriersolve matrix %" PetscInt_FMT " is not assembled",
             ctx->matrix_id);
  PetscCall(MatHasOperation(A, MATOP_GET_ROW, &has_get_row));
  PetscCheck(has_get_row, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "-fouriersolve matrix %" PetscInt_FMT
             " does not support MatGetRow()",
             ctx->matrix_id);
  PetscCall(MatGetSize(A, &global_dim, &global_cols));
  PetscCall(MatGetLocalSize(A, &local_rows, &local_cols));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetOwnershipRangeColumn(A, &cstart, &cend));
  PetscCheck(global_dim == global_cols,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
             "-fouriersolve matrix %" PetscInt_FMT
             " must be square; global size=%" PetscInt_FMT "x%" PetscInt_FMT,
             ctx->matrix_id, global_dim, global_cols);
  local_ok = local_rows == local_cols && rstart == cstart && rend == cend ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
             "-fouriersolve matrix %" PetscInt_FMT
             " must have identical row/column ownership on every rank; this "
             "rank owns rows [%" PetscInt_FMT ",%" PetscInt_FMT
             ") and columns [%" PetscInt_FMT ",%" PetscInt_FMT ")",
             ctx->matrix_id, rstart, rend, cstart, cend);

  m3dc1_plane_getnum(&nplane_api);
  m3dc1_plane_getid(&plane_id_api);
  PetscCheck(nplane_api > 1, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "Toroidal Fourier PC requires at least two planes");
  ctx->nplane = (PetscInt)nplane_api;
  ctx->plane_id = (PetscInt)plane_id_api;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &world_size));
  PetscCheck(world_size % nplane_api == 0, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_SIZ,
             "MPI size %d is not divisible by nplane %d", (int)world_size,
             nplane_api);
  ctx->npart = world_size / nplane_api;
  ctx->partition_id = world_rank % ctx->npart;
  local_ok = ctx->plane_id == world_rank / ctx->npart ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONGSTATE,
             "At least one rank violates the plane-major MPI layout; on this "
             "rank %d maps to plane %d and m3dc1_plane_getid() returned %d",
             (int)world_rank, (int)(world_rank / ctx->npart), plane_id_api);

  if (ctx->column_comm == MPI_COMM_NULL) {
    PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, (int)ctx->partition_id,
                                (int)ctx->plane_id, &ctx->column_comm));
    PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, (int)ctx->plane_id,
                                (int)ctx->partition_id, &ctx->mode_comm));
  }
  PetscCallMPI(MPI_Comm_rank(ctx->column_comm, &column_rank));
  PetscCallMPI(MPI_Comm_size(ctx->column_comm, &column_size));
  local_ok = column_size == nplane_api && column_rank == plane_id_api ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONGSTATE,
             "A toroidal column communicator is inconsistent; on this rank "
             "size=%d (expected %d), column rank=%d, plane id=%d",
             (int)column_size, nplane_api, (int)column_rank, plane_id_api);

  if (ctx->check_uniform) {
    std::vector<double> phi((size_t)ctx->nplane);
    for (PetscInt p = 0; p < ctx->nplane; ++p) {
      int ip = (int)p;
      m3dc1_plane_getphi(&ip, &phi[(size_t)p]);
    }
    const double dphi = phi[1] - phi[0];
    const double phi_tol = 1.e-10 * std::max(1.0, std::abs(dphi));
    local_ok = std::abs(dphi) > phi_tol ? 1 : 0;
    for (PetscInt p = 2; p < ctx->nplane; ++p) {
      const double this_dphi = phi[(size_t)p] - phi[(size_t)p - 1];
      if (std::abs(this_dphi - dphi) > phi_tol)
        local_ok = 0;
    }
    PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                               PETSC_COMM_WORLD));
    PetscCheck(global_ok, PETSC_COMM_WORLD, PETSC_ERR_SUP,
               "-fouriersolve requires nondegenerate, uniformly spaced "
               "toroidal "
               "plane coordinates");
  }

  PetscCheck(global_dim % ctx->nplane == 0, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_SIZ,
             "Matrix dimension %" PetscInt_FMT
             " is not divisible by nplane %" PetscInt_FMT,
             global_dim, ctx->nplane);
  ctx->plane_dim = global_dim / ctx->nplane;
  ctx->local_dim = rend - rstart;
  local_ok = rstart >= ctx->plane_id * ctx->plane_dim &&
                     rend <= (ctx->plane_id + 1) * ctx->plane_dim
                 ? 1
                 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
             "At least one matrix row range is not plane-local; this rank "
             "owns [ %" PetscInt_FMT ", %" PetscInt_FMT
             " ) in plane %" PetscInt_FMT " range [ %" PetscInt_FMT
             ", %" PetscInt_FMT " )",
             rstart, rend, ctx->plane_id, ctx->plane_id * ctx->plane_dim,
             (ctx->plane_id + 1) * ctx->plane_dim);

  PetscInt min_local_dim = 0, max_local_dim = 0;
  PetscCallMPI(MPI_Allreduce(&ctx->local_dim, &min_local_dim, 1, MPIU_INT,
                             MPI_MIN, ctx->column_comm));
  PetscCallMPI(MPI_Allreduce(&ctx->local_dim, &max_local_dim, 1, MPIU_INT,
                             MPI_MAX, ctx->column_comm));
  local_ok = min_local_dim == max_local_dim ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_SIZ,
             "At least one toroidal partition column has different local "
             "sizes (this column min=%" PetscInt_FMT ", max=%" PetscInt_FMT
             ")",
             min_local_dim, max_local_dim);
  const PetscInt base_rstart = rstart - ctx->plane_id * ctx->plane_dim;
  PetscInt min_base_rstart = 0, max_base_rstart = 0;
  PetscCallMPI(MPI_Allreduce(&base_rstart, &min_base_rstart, 1, MPIU_INT,
                             MPI_MIN, ctx->column_comm));
  PetscCallMPI(MPI_Allreduce(&base_rstart, &max_base_rstart, 1, MPIU_INT,
                             MPI_MAX, ctx->column_comm));
  local_ok = min_base_rstart == max_base_rstart ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONGSTATE,
             "At least one toroidal partition column has different "
             "plane-local row starts (this column min=%" PetscInt_FMT
             ", max=%" PetscInt_FMT ")",
             min_base_rstart, max_base_rstart);

  /* Map this physical plane's rows to plane-local (row,col,shift) entries. */
  if (ctx->timing)
    timing_extract_start = MPI_Wtime();
  std::vector<TFEntry> entries;
  MatInfo ainfo;
  PetscCall(MatGetInfo(A, MAT_LOCAL, &ainfo));
  if (ainfo.nz_used > 0.0)
    entries.reserve((size_t)ainfo.nz_used);
  std::vector<PetscInt> local_shift_present((size_t)ctx->nplane, 0);
  std::vector<PetscInt> global_shift_present((size_t)ctx->nplane, 0);
  PetscReal local_anorm2 = 0.0;
  PetscInt local_columns_valid = 1;
  for (PetscInt row = rstart; row < rend; ++row) {
    const size_t row_entry_begin = entries.size();
    PetscInt ncols = 0;
    const PetscInt *cols = NULL;
    const PetscScalar *vals = NULL;
    PetscCall(MatGetRow(A, row, &ncols, &cols, &vals));
    for (PetscInt j = 0; j < ncols; ++j) {
      const PetscInt col_plane = cols[j] / ctx->plane_dim;
      if (cols[j] < 0 || col_plane < 0 || col_plane >= ctx->nplane) {
        local_columns_valid = 0;
        continue;
      }
      TFEntry entry;
      entry.key.row = row - ctx->plane_id * ctx->plane_dim;
      entry.key.col = cols[j] - col_plane * ctx->plane_dim;
      entry.key.shift =
          (col_plane - ctx->plane_id + ctx->nplane) % ctx->nplane;
      entry.value = vals[j];
      local_shift_present[(size_t)entry.key.shift] = 1;
      entries.push_back(entry);
      const PetscReal av = PetscAbsScalar(vals[j]);
      local_anorm2 += av * av;
    }
    PetscCall(MatRestoreRow(A, row, &ncols, &cols, &vals));
    /* Rows arrive in increasing order.  Sorting only the entries from this
       row produces the same global (row,col,shift) order as one very large
       sort, at substantially lower setup cost for sparse matrices. */
    std::sort(entries.begin() + row_entry_begin, entries.end(),
              TFEntryLess());
  }
  PetscCallMPI(MPI_Allreduce(&local_columns_valid, &global_ok, 1, MPIU_INT,
                             MPI_MIN, PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "At least one matrix column does not map to a toroidal plane");
  PetscMPIInt mpi_nplane = 0;
  PetscCall(PetscMPIIntCast(ctx->nplane, &mpi_nplane));
  PetscCallMPI(MPI_Allreduce(&local_shift_present[0],
                             &global_shift_present[0], mpi_nplane, MPIU_INT,
                             MPI_MAX, PETSC_COMM_WORLD));

  size_t ncombined = 0;
  for (size_t i = 0; i < entries.size(); ++i) {
    if (ncombined && TFSameKey(entries[ncombined - 1].key, entries[i].key)) {
      entries[ncombined - 1].value += entries[i].value;
    } else {
      if (ncombined != i)
        entries[ncombined] = entries[i];
      ++ncombined;
    }
  }
  entries.resize(ncombined);

  const PetscInt nentry = (PetscInt)entries.size();
  PetscInt min_nentry = 0, max_nentry = 0;
  PetscCallMPI(MPI_Allreduce(&nentry, &min_nentry, 1, MPIU_INT, MPI_MIN,
                             ctx->column_comm));
  PetscCallMPI(MPI_Allreduce(&nentry, &max_nentry, 1, MPIU_INT, MPI_MAX,
                             ctx->column_comm));
  local_ok = min_nentry == max_nentry ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONGSTATE,
             "Toroidal matrix sparsity differs between corresponding planes "
             "in at least one partition column (this column entry counts "
             "min=%" PetscInt_FMT ", max=%" PetscInt_FMT ")",
             min_nentry, max_nentry);
  PetscCall(PetscMPIIntCast(nentry, &mpi_nentry));

  std::vector<PetscScalar> local_values(nentry), average_values(nentry);
  for (PetscInt i = 0; i < nentry; ++i)
    local_values[i] = entries[(size_t)i].value;

  /* Exact, chunked pattern check avoids storing three extra nnz arrays. */
  PetscInt local_pattern_ok = 1, global_pattern_ok = 0;
  /* Keep the exact comparison, but use a multi-megabyte chunk so a matrix
     with millions of local nonzeros does not require hundreds of small,
     latency-bound broadcasts. */
  const PetscInt pattern_chunk = 262144;
  const PetscInt pattern_buffer_entries = std::min(pattern_chunk, nentry);
  std::vector<PetscInt> key_buffer(
      (size_t)(3 * pattern_buffer_entries));
  for (PetscInt begin = 0; begin < nentry; begin += pattern_chunk) {
    const PetscInt count = std::min(pattern_chunk, nentry - begin);
    PetscMPIInt mpi_key_count = 0;
    PetscCall(PetscMPIIntCast(3 * count, &mpi_key_count));
    if (column_rank == 0) {
      for (PetscInt i = 0; i < count; ++i) {
        const TFKey &key = entries[(size_t)(begin + i)].key;
        key_buffer[(size_t)(3 * i)] = key.row;
        key_buffer[(size_t)(3 * i + 1)] = key.col;
        key_buffer[(size_t)(3 * i + 2)] = key.shift;
      }
    }
    PetscCallMPI(MPI_Bcast(&key_buffer[0], mpi_key_count, MPIU_INT, 0,
                           ctx->column_comm));
    if (column_rank != 0) {
      for (PetscInt i = 0; i < count; ++i) {
        const TFKey &key = entries[(size_t)(begin + i)].key;
        if (key.row != key_buffer[(size_t)(3 * i)] ||
            key.col != key_buffer[(size_t)(3 * i + 1)] ||
            key.shift != key_buffer[(size_t)(3 * i + 2)]) {
          local_pattern_ok = 0;
          break;
        }
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_pattern_ok, &global_pattern_ok, 1,
                             MPIU_INT, MPI_MIN, PETSC_COMM_WORLD));
  PetscCheck(global_pattern_ok, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONGSTATE,
             "Toroidal Fourier averaging requires identical plane-local "
             "sparsity on corresponding partitions");

  if (mpi_nentry) {
    PetscCallMPI(MPI_Allreduce(&local_values[0], &average_values[0],
                               mpi_nentry, MPIU_SCALAR, MPI_SUM,
                               ctx->column_comm));
  }
  for (PetscInt i = 0; i < nentry; ++i)
    average_values[i] /= (PetscScalar)ctx->nplane;

  PetscReal local_diff2 = 0.0;
  for (PetscInt i = 0; i < nentry; ++i) {
    const PetscReal dv = PetscAbsScalar(local_values[i] - average_values[i]);
    local_diff2 += dv * dv;
  }
  PetscReal global_anorm2 = 0.0, global_diff2 = 0.0;
  PetscCallMPI(MPI_Allreduce(&local_anorm2, &global_anorm2, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_diff2, &global_diff2, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  const PetscReal variation =
      global_anorm2 > 0.0 ? PetscSqrtReal(global_diff2 / global_anorm2) : 0.0;
  if (ctx->timing)
    timing_extract = MPI_Wtime() - timing_extract_start;

  /* Build the symbol for mode k=plane_id on that plane's 2-D communicator. */
  if (ctx->timing)
    timing_mode_matrix_start = MPI_Wtime();
#ifdef PETSC_USE_COMPLEX
  const PetscInt mode_local_dim = ctx->local_dim;
  const PetscInt mode_global_dim = ctx->plane_dim;
#else
  const PetscInt mode_local_dim = 2 * ctx->local_dim;
  const PetscInt mode_global_dim = 2 * ctx->plane_dim;
#endif
  PetscCall(MatCreate(ctx->mode_comm, &ctx->mode_mat));
  PetscCall(MatSetSizes(ctx->mode_mat, mode_local_dim, mode_local_dim,
                        mode_global_dim, mode_global_dim));
  PetscCall(MatSetType(ctx->mode_mat, MATAIJ));
  PetscCall(MatSetOptionsPrefix(ctx->mode_mat,
                                ctx->options_prefix.c_str()));
  PetscCall(MatSetFromOptions(ctx->mode_mat));

  std::vector<PetscInt> dnnz((size_t)mode_local_dim, 0);
  std::vector<PetscInt> onnz((size_t)mode_local_dim, 0);
  PetscInt previous_row = -1, previous_col = -1;
  for (PetscInt i = 0; i < nentry; ++i) {
    const TFKey &key = entries[(size_t)i].key;
    if (key.row == previous_row && key.col == previous_col)
      continue;
    previous_row = key.row;
    previous_col = key.col;
    const bool diagonal =
        key.col >= base_rstart &&
        key.col < base_rstart + ctx->local_dim;
#ifdef PETSC_USE_COMPLEX
    const size_t local_row = (size_t)(key.row - base_rstart);
    if (diagonal)
      ++dnnz[local_row];
    else
      ++onnz[local_row];
#else
    const size_t local_row = (size_t)(2 * (key.row - base_rstart));
    if (diagonal) {
      dnnz[local_row] += 2;
      dnnz[local_row + 1] += 2;
    } else {
      onnz[local_row] += 2;
      onnz[local_row + 1] += 2;
    }
#endif
  }
  PetscCall(MatXAIJSetPreallocation(
      ctx->mode_mat, 1, dnnz.empty() ? NULL : &dnnz[0],
      onnz.empty() ? NULL : &onnz[0], NULL, NULL));
#ifndef PETSC_USE_COMPLEX
  /* The real representation stores each complex scalar as the exact
     interleaved (Re,Im) 2x2 block [H -G; G H].  MATAIJ still uses scalar
     storage, but advertising this logical block size lets factor packages
     such as MUMPS compress their input in block form without changing the
     matrix or the solve.  MatSetBlockSize may be called after scalar AIJ
     preallocation as long as every local size is compatible; the ownership
     checks below require precisely that even interleaved layout.  A user may
     override -mat_type, so do not apply this late metadata change to BAIJ or
     another format whose block size is fixed by preallocation. */
  PetscBool is_seqaij = PETSC_FALSE, is_mpiaij = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->mode_mat, MATSEQAIJ,
                                   &is_seqaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->mode_mat, MATMPIAIJ,
                                   &is_mpiaij));
  if (is_seqaij || is_mpiaij)
    PetscCall(MatSetBlockSize(ctx->mode_mat, 2));
#endif
  PetscCall(MatSetOption(ctx->mode_mat, MAT_NEW_NONZERO_ALLOCATION_ERR,
                         PETSC_TRUE));
  PetscCall(MatSetUp(ctx->mode_mat));

  PetscInt mode_rstart = 0, mode_rend = 0;
  PetscCall(MatGetOwnershipRange(ctx->mode_mat, &mode_rstart, &mode_rend));
#ifdef PETSC_USE_COMPLEX
  local_ok = mode_rstart == base_rstart &&
                     mode_rend == base_rstart + ctx->local_dim
                 ? 1
                 : 0;
#else
  local_ok = mode_rstart == 2 * base_rstart &&
                     mode_rend == 2 * (base_rstart + ctx->local_dim)
                 ? 1
                 : 0;
#endif
  PetscCallMPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPIU_INT, MPI_MIN,
                             PETSC_COMM_WORLD));
  PetscCheck(global_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
             "At least one Fourier-mode matrix has an unexpected ownership "
             "range");

  const PetscReal theta =
      2.0 * PETSC_PI * (PetscReal)ctx->plane_id / (PetscReal)ctx->nplane;
  ctx->phase_cos.resize((size_t)ctx->nplane);
  ctx->phase_sin.resize((size_t)ctx->nplane);
  for (PetscInt p = 0; p < ctx->nplane; ++p) {
    const PetscReal angle = theta * (PetscReal)p;
    ctx->phase_cos[(size_t)p] = PetscCosReal(angle);
    ctx->phase_sin[(size_t)p] = PetscSinReal(angle);
  }
  /* Entries are sorted by (row,col,shift).  Sum all toroidal shifts for a
     logical matrix entry first, then insert a complete logical row at once.
     Besides avoiding trigonometry per nonzero, this changes millions of
     tiny MatSetValues calls on large systems into roughly local_dim calls. */
  {
    const size_t row_capacity =
        ctx->local_dim > 0
            ? entries.size() / (size_t)ctx->local_dim + (size_t)1
            : (size_t)0;
    std::vector<PetscInt> symbol_cols;
    symbol_cols.reserve(row_capacity);
#ifdef PETSC_USE_COMPLEX
    std::vector<PetscScalar> symbol_values;
    symbol_values.reserve(row_capacity);
#else
    std::vector<PetscScalar> symbol_real;
    std::vector<PetscScalar> symbol_imag;
    std::vector<PetscInt> block_cols;
    std::vector<PetscScalar> block_values;
    symbol_real.reserve(row_capacity);
    symbol_imag.reserve(row_capacity);
    block_cols.reserve(2u * row_capacity);
    block_values.reserve(4u * row_capacity);
#endif
    size_t begin = 0;
    while (begin < entries.size()) {
      const PetscInt logical_row = entries[begin].key.row;
      symbol_cols.clear();
#ifdef PETSC_USE_COMPLEX
      symbol_values.clear();
#else
      symbol_real.clear();
      symbol_imag.clear();
#endif

      size_t end = begin;
      while (end < entries.size() && entries[end].key.row == logical_row) {
        const PetscInt logical_col = entries[end].key.col;
#ifdef PETSC_USE_COMPLEX
        PetscScalar symbol_value = 0.0;
#else
        PetscScalar symbol_h = 0.0, symbol_g = 0.0;
#endif
        do {
          const TFKey &key = entries[end].key;
          const PetscReal c = ctx->phase_cos[(size_t)key.shift];
          const PetscReal s = ctx->phase_sin[(size_t)key.shift];
#ifdef PETSC_USE_COMPLEX
          symbol_value += average_values[end] * PetscCMPLX(c, s);
#else
          const PetscReal ar = PetscRealPart(average_values[end]);
          const PetscReal ai = PetscImaginaryPart(average_values[end]);
          symbol_h += (PetscScalar)(ar * c - ai * s);
          symbol_g += (PetscScalar)(ar * s + ai * c);
#endif
          ++end;
        } while (end < entries.size() &&
                 entries[end].key.row == logical_row &&
                 entries[end].key.col == logical_col);

        symbol_cols.push_back(logical_col);
#ifdef PETSC_USE_COMPLEX
        symbol_values.push_back(symbol_value);
#else
        symbol_real.push_back(symbol_h);
        symbol_imag.push_back(symbol_g);
#endif
      }

#ifdef PETSC_USE_COMPLEX
      const PetscInt ncols = (PetscInt)symbol_cols.size();
      PetscCall(MatSetValues(ctx->mode_mat, 1, &logical_row, ncols,
                             symbol_cols.empty() ? NULL : &symbol_cols[0],
                             symbol_values.empty() ? NULL : &symbol_values[0],
                             INSERT_VALUES));
#else
      const PetscInt logical_ncols = (PetscInt)symbol_cols.size();
      block_cols.resize((size_t)(2 * logical_ncols));
      block_values.resize((size_t)(4 * logical_ncols));
      for (PetscInt j = 0; j < logical_ncols; ++j) {
        block_cols[(size_t)(2 * j)] = 2 * symbol_cols[(size_t)j];
        block_cols[(size_t)(2 * j + 1)] =
            2 * symbol_cols[(size_t)j] + 1;
        block_values[(size_t)(2 * j)] = symbol_real[(size_t)j];
        block_values[(size_t)(2 * j + 1)] = -symbol_imag[(size_t)j];
        block_values[(size_t)(2 * logical_ncols + 2 * j)] =
            symbol_imag[(size_t)j];
        block_values[(size_t)(2 * logical_ncols + 2 * j + 1)] =
            symbol_real[(size_t)j];
      }
      const PetscInt block_rows[2] = {2 * logical_row,
                                      2 * logical_row + 1};
      PetscCall(MatSetValues(
          ctx->mode_mat, 2, block_rows, 2 * logical_ncols,
          block_cols.empty() ? NULL : &block_cols[0],
          block_values.empty() ? NULL : &block_values[0], INSERT_VALUES));
#endif
      begin = end;
    }
  }
  PetscCall(MatAssemblyBegin(ctx->mode_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->mode_mat, MAT_FINAL_ASSEMBLY));
  if (ctx->timing)
    timing_mode_matrix = MPI_Wtime() - timing_mode_matrix_start;

  PetscCall(KSPCreate(ctx->mode_comm, &ctx->mode_ksp));
  PetscCall(KSPSetOptionsPrefix(ctx->mode_ksp,
                                ctx->options_prefix.c_str()));
  PetscCall(KSPSetOperators(ctx->mode_ksp, ctx->mode_mat, ctx->mode_mat));
  PetscCall(KSPSetType(ctx->mode_ksp, KSPPREONLY));
  PC mode_pc = NULL;
  PetscCall(KSPGetPC(ctx->mode_ksp, &mode_pc));
  PetscCall(PCSetType(mode_pc, PCLU));
  PetscCall(KSPSetFromOptions(ctx->mode_ksp));

  if (world_rank == 0) {
    std::cout << "[M3DC1 FOURIER INFO] matrix " << ctx->matrix_id << ": "
              << ctx->nplane << " concurrent modes, " << ctx->npart
              << " MPI ranks/mode, " << ctx->plane_dim
              << " coupled dofs/mode, toroidal variation=" << variation
              << ", options prefix=" << ctx->options_prefix
              << ", block shifts={";
    bool first_shift = true;
    for (PetscInt shift = 0; shift < ctx->nplane; ++shift) {
      if (!global_shift_present[(size_t)shift])
        continue;
      if (!first_shift)
        std::cout << ",";
      std::cout << shift;
      first_shift = false;
    }
    std::cout << "}" << std::endl;
  }
  if (ctx->debug && ctx->partition_id == 0) {
    KSPType ksp_type = NULL;
    PCType pc_type = NULL;
    PetscCall(KSPGetType(ctx->mode_ksp, &ksp_type));
    PetscCall(KSPGetPC(ctx->mode_ksp, &mode_pc));
    PetscCall(PCGetType(mode_pc, &pc_type));
    std::cout << "[M3DC1 FOURIER DEBUG rank " << world_rank << "] matrix "
              << ctx->matrix_id << " mode " << ctx->plane_id
              << ": KSP=" << ksp_type << " PC=" << pc_type
              << " local rows=" << mode_local_dim << std::endl;
  }

  /* Do not retain setup scratch arrays while the direct factorization is
     allocating its (usually much larger) symbolic and numeric factors. */
  std::vector<TFEntry>().swap(entries);
  std::vector<PetscScalar>().swap(local_values);
  std::vector<PetscScalar>().swap(average_values);
  std::vector<PetscInt>().swap(dnnz);
  std::vector<PetscInt>().swap(onnz);
  std::vector<PetscInt>().swap(key_buffer);
  std::vector<PetscInt>().swap(local_shift_present);
  std::vector<PetscInt>().swap(global_shift_present);

  /* KSPSetUp is collective only on one mode communicator.  Delay error
     propagation until every mode has reached a world collective; otherwise
     one failed factorization can leave the other modes waiting forever. */
  if (ctx->timing)
    timing_mode_ksp_start = MPI_Wtime();
  PetscErrorCode mode_setup_ierr = KSPSetUp(ctx->mode_ksp);
  if (ctx->timing)
    timing_mode_ksp = MPI_Wtime() - timing_mode_ksp_start;
  if (mode_setup_ierr && ctx->partition_id == 0)
    std::cout << "[M3DC1 FOURIER ERROR] matrix " << ctx->matrix_id
              << " mode " << ctx->plane_id
              << " KSP setup failed with PetscErrorCode "
              << (int)mode_setup_ierr << std::endl;
  PetscInt local_setup_failed = mode_setup_ierr ? 1 : 0;
  PetscInt any_setup_failed = 0;
  PetscCallMPI(MPI_Allreduce(&local_setup_failed, &any_setup_failed, 1,
                             MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCheck(!any_setup_failed, PETSC_COMM_WORLD, PETSC_ERR_LIB,
             "One or more toroidal Fourier mode KSP setups failed");
  PetscCall(MatCreateVecs(ctx->mode_mat, &ctx->mode_sol, &ctx->mode_rhs));

  ctx->plane_values.resize((size_t)ctx->nplane * (size_t)ctx->local_dim);
  PetscCall(MatGetState(A, &ctx->matrix_state));
  ctx->ready = PETSC_TRUE;
  PetscCall(PCSetFailedReason(pc, PC_NOERROR));
  if (ctx->timing) {
    const double timing_local[4] = {
        MPI_Wtime() - timing_setup_start, timing_extract,
        timing_mode_matrix, timing_mode_ksp};
    double timing_max[4] = {0.0, 0.0, 0.0, 0.0};
    PetscCallMPI(MPI_Reduce(timing_local, timing_max, 4, MPI_DOUBLE, MPI_MAX,
                             0, PETSC_COMM_WORLD));
    if (world_rank == 0) {
      std::cout << "[M3DC1 FOURIER TIMING] matrix " << ctx->matrix_id
                << " setup max: total=" << timing_max[0]
                << " extract/pattern/average=" << timing_max[1]
                << " mode_matrix=" << timing_max[2]
                << " mode_ksp_setup=" << timing_max[3] << " seconds"
                << std::endl;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TFApply(PC pc, Vec x, Vec y) {
  ToroidalFourierPC *ctx = NULL;
  const PetscScalar *xarray = NULL;
  const PetscScalar *zarray = NULL;
  PetscScalar *barray = NULL, *yarray = NULL;
  PetscMPIInt mpi_local_dim;
  double timing_apply_start = 0.0;
  double timing_forward_start = 0.0, timing_forward = 0.0;
  double timing_mode_solve_start = 0.0, timing_mode_solve = 0.0;
  double timing_failure_sync_start = 0.0, timing_failure_sync = 0.0;
  double timing_inverse_start = 0.0, timing_inverse = 0.0;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_NULL,
             "Toroidal Fourier PC context is NULL");
  if (ctx->timing)
    timing_apply_start = MPI_Wtime();

  if (ctx->rebuild && ctx->ready) {
    PetscObjectState current_state = 0;
    PetscCall(MatGetState(ctx->A, &current_state));
    PetscInt local_changed = current_state != ctx->matrix_state ? 1 : 0;
    PetscInt any_changed = 0;
    PetscCallMPI(MPI_Allreduce(&local_changed, &any_changed, 1, MPIU_INT,
                               MPI_MAX, PETSC_COMM_WORLD));
    if (any_changed)
      PetscCall(TFSetUp(pc));
  }
  PetscCheck(ctx->ready, PetscObjectComm((PetscObject)pc),
             PETSC_ERR_ARG_WRONGSTATE,
             "Toroidal Fourier PC was applied before setup");

  /* KSP work-vector layouts are invariant for a fixed operator.  Validate
     them collectively once per setup instead of paying for a world
     synchronization on every preconditioner application. */
  if (!ctx->vector_sizes_checked) {
    PetscInt xlocal = 0, ylocal = 0;
    PetscCall(VecGetLocalSize(x, &xlocal));
    PetscCall(VecGetLocalSize(y, &ylocal));
    PetscInt local_size_ok =
        xlocal == ctx->local_dim && ylocal == ctx->local_dim ? 1 : 0;
    PetscInt global_size_ok = 0;
    PetscCallMPI(MPI_Allreduce(&local_size_ok, &global_size_ok, 1, MPIU_INT,
                               MPI_MIN, PETSC_COMM_WORLD));
    PetscCheck(global_size_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
               "At least one toroidal Fourier PC vector local size does not "
               "match its matrix row ownership");
    ctx->vector_sizes_checked = PETSC_TRUE;
  }
  PetscCall(PetscMPIIntCast(ctx->local_dim, &mpi_local_dim));

  if (ctx->timing) {
    timing_forward_start = MPI_Wtime();
    ++ctx->apply_count;
  }
  PetscCall(VecGetArrayRead(x, &xarray));
  PetscCallMPI(MPI_Allgather(xarray, mpi_local_dim, MPIU_SCALAR,
                             ctx->plane_values.empty()
                                 ? NULL
                                 : &ctx->plane_values[0],
                             mpi_local_dim, MPIU_SCALAR,
                             ctx->column_comm));
  PetscCall(VecRestoreArrayRead(x, &xarray));

  PetscCall(VecGetArray(ctx->mode_rhs, &barray));
#ifdef PETSC_USE_COMPLEX
  std::fill(barray, barray + ctx->local_dim, (PetscScalar)0.0);
  for (PetscInt p = 0; p < ctx->nplane; ++p) {
    const PetscScalar phase =
        PetscCMPLX(ctx->phase_cos[(size_t)p], -ctx->phase_sin[(size_t)p]);
    const PetscScalar *plane =
        &ctx->plane_values[(size_t)p * (size_t)ctx->local_dim];
    for (PetscInt j = 0; j < ctx->local_dim; ++j) {
      barray[j] += phase * plane[j];
    }
  }
#else
  std::fill(barray, barray + 2 * ctx->local_dim, (PetscScalar)0.0);
  for (PetscInt p = 0; p < ctx->nplane; ++p) {
    const PetscReal c = ctx->phase_cos[(size_t)p];
    const PetscReal s = ctx->phase_sin[(size_t)p];
    const PetscScalar *plane =
        &ctx->plane_values[(size_t)p * (size_t)ctx->local_dim];
    for (PetscInt j = 0; j < ctx->local_dim; ++j) {
      const PetscScalar value = plane[j];
      barray[2 * j] += (PetscScalar)(c * value);
      barray[2 * j + 1] -= (PetscScalar)(s * value);
    }
  }
#endif
  PetscCall(VecRestoreArray(ctx->mode_rhs, &barray));
  if (ctx->timing)
    timing_forward = MPI_Wtime() - timing_forward_start;
  /* A mode solve is collective only on mode_comm.  Convert subgroup errors
     and negative convergence reasons into one collective outer-PC failure. */
  if (ctx->timing)
    timing_mode_solve_start = MPI_Wtime();
  PetscErrorCode mode_solve_ierr =
      KSPSolve(ctx->mode_ksp, ctx->mode_rhs, ctx->mode_sol);
  if (ctx->timing) {
    timing_mode_solve = MPI_Wtime() - timing_mode_solve_start;
    timing_failure_sync_start = MPI_Wtime();
  }
  KSPConvergedReason local_reason = KSP_CONVERGED_ITERATING;
  if (!mode_solve_ierr)
    mode_solve_ierr = KSPGetConvergedReason(ctx->mode_ksp, &local_reason);
  PetscInt local_failed =
      mode_solve_ierr || local_reason < 0 ? 1 : 0, any_failed = 0;
  PetscCallMPI(MPI_Allreduce(&local_failed, &any_failed, 1, MPIU_INT, MPI_MAX,
                             PETSC_COMM_WORLD));
  if (ctx->timing)
    timing_failure_sync = MPI_Wtime() - timing_failure_sync_start;
  if (any_failed) {
    PetscCall(VecSet(y, 0.0));
    PetscCall(PCSetFailedReason(pc, PC_SUBPC_ERROR));
    if (ctx->debug && local_failed && ctx->partition_id == 0)
      std::cout << "[M3DC1 FOURIER DEBUG] matrix " << ctx->matrix_id
                << " mode " << ctx->plane_id
                << " failed with PetscErrorCode " << (int)mode_solve_ierr
                << " and KSP reason " << (int)local_reason
                << std::endl;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Form this mode's contribution to every physical plane, then sum and
     scatter one plane back to each rank.  In a real PETSc build this moves
     nplane*local_dim scalars instead of allgathering the interleaved complex
     mode vectors (2*nplane*local_dim scalars), and it avoids retaining a
     second transform-sized work buffer. */
  if (ctx->timing)
    timing_inverse_start = MPI_Wtime();
  PetscCall(VecGetArrayRead(ctx->mode_sol, &zarray));
#ifdef PETSC_USE_COMPLEX
  for (PetscInt p = 0; p < ctx->nplane; ++p) {
    const PetscScalar phase =
        PetscCMPLX(ctx->phase_cos[(size_t)p], ctx->phase_sin[(size_t)p]);
    PetscScalar *contribution =
        &ctx->plane_values[(size_t)p * (size_t)ctx->local_dim];
    for (PetscInt j = 0; j < ctx->local_dim; ++j) {
      contribution[j] = phase * zarray[j];
    }
  }
#else
  for (PetscInt p = 0; p < ctx->nplane; ++p) {
    const PetscReal c = ctx->phase_cos[(size_t)p];
    const PetscReal s = ctx->phase_sin[(size_t)p];
    PetscScalar *contribution =
        &ctx->plane_values[(size_t)p * (size_t)ctx->local_dim];
    for (PetscInt j = 0; j < ctx->local_dim; ++j) {
      const PetscScalar zr = zarray[2 * j];
      const PetscScalar zi = zarray[2 * j + 1];
      contribution[j] = (PetscScalar)(c * zr - s * zi);
    }
  }
#endif
  PetscCall(VecRestoreArrayRead(ctx->mode_sol, &zarray));

  PetscCall(VecGetArray(y, &yarray));
  PetscCallMPI(MPI_Reduce_scatter_block(
      ctx->plane_values.empty() ? NULL : &ctx->plane_values[0], yarray,
      mpi_local_dim, MPIU_SCALAR, MPI_SUM, ctx->column_comm));
  const PetscScalar inverse_nplane =
      (PetscScalar)1.0 / (PetscScalar)ctx->nplane;
  for (PetscInt j = 0; j < ctx->local_dim; ++j)
    yarray[j] *= inverse_nplane;
  PetscCall(VecRestoreArray(y, &yarray));
  if (ctx->timing)
    timing_inverse = MPI_Wtime() - timing_inverse_start;
  PetscCall(PCSetFailedReason(pc, PC_NOERROR));
  if (ctx->timing) {
    const double timing_local[5] = {
        MPI_Wtime() - timing_apply_start, timing_forward,
        timing_mode_solve, timing_failure_sync, timing_inverse};
    double timing_max[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    PetscMPIInt world_rank = 0;
    PetscCallMPI(MPI_Reduce(timing_local, timing_max, 5, MPI_DOUBLE, MPI_MAX,
                             0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank));
    if (world_rank == 0) {
      std::cout << "[M3DC1 FOURIER TIMING] matrix " << ctx->matrix_id
                << " apply " << ctx->apply_count
                << " max: total=" << timing_max[0]
                << " forward=" << timing_max[1]
                << " mode_solve=" << timing_max[2]
                << " failure_sync=" << timing_max[3]
                << " inverse=" << timing_max[4] << " seconds"
                << std::endl;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

static PetscErrorCode TFConfigurePC(KSP ksp, Mat A, PetscInt matrix_id) {
  PC pc = NULL;
  const char *outer_prefix = NULL;
  PetscFunctionBeginUser;

  ToroidalFourierPC *ctx = new ToroidalFourierPC();
  ctx->A = A;
  ctx->matrix_id = matrix_id;
  PetscCall(KSPGetOptionsPrefix(ksp, &outer_prefix));
  ctx->options_prefix = outer_prefix ? outer_prefix : "";
  ctx->options_prefix += "fourier_";
  PetscCall(PetscOptionsGetBool(NULL, ctx->options_prefix.c_str(), "-rebuild",
                                &ctx->rebuild, NULL));
  PetscCall(PetscOptionsGetBool(NULL, ctx->options_prefix.c_str(),
                                "-check_uniform",
                                &ctx->check_uniform, NULL));
  PetscCall(PetscOptionsGetBool(NULL, ctx->options_prefix.c_str(), "-debug",
                                &ctx->debug, NULL));
  PetscCall(PetscOptionsGetBool(NULL, ctx->options_prefix.c_str(), "-timing",
                                &ctx->timing, NULL));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetSetUp(pc, TFSetUp));
  PetscCall(PCShellSetApply(pc, TFApply));
  PetscCall(PCShellSetDestroy(pc, TFDestroy));
  PetscCall(PCShellSetView(pc, TFView));
  PetscCall(PCShellSetName(pc, "coupled toroidal Fourier modes"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode m3dc1_configure_toroidal_fourier_pc(
    KSP ksp, Mat A, PetscInt matrix_id) {
  PetscFunctionBeginUser;
  PetscCall(TFConfigurePC(ksp, A, matrix_id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
