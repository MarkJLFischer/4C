// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_iterative.hpp"

#include "4C_comm_utils.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linear_solver_amgnxn_preconditioner.hpp"
#include "4C_linear_solver_epetra_utils.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_linear_solver_method_projector.hpp"
#include "4C_linear_solver_preconditioner_ifpack.hpp"
#include "4C_linear_solver_preconditioner_muelu.hpp"
#include "4C_linear_solver_preconditioner_projection.hpp"
#include "4C_linear_solver_preconditioner_teko.hpp"
#include "4C_utils_exceptions.hpp"

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <Xpetra_TpetraMultiVector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

using BelosVectorType = Epetra_MultiVector;
using BelosMatrixType = Epetra_Operator;

namespace
{
  using TpetraScalar = double;
  using TpetraLocalOrdinal = int;
  using TpetraGlobalOrdinal = int;
  using TpetraNode = Tpetra::Map<>::node_type;
  using TpetraMap = Tpetra::Map<TpetraLocalOrdinal, TpetraGlobalOrdinal, TpetraNode>;
  using TpetraMatrix =
      Tpetra::CrsMatrix<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal, TpetraNode>;
  using TpetraMultiVector =
      Tpetra::MultiVector<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal, TpetraNode>;
  using TpetraOperator =
      Tpetra::Operator<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal, TpetraNode>;
  using XpetraMatrix = Xpetra::Matrix<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal,
      TpetraNode>;
  using XpetraMultiVector = Xpetra::MultiVector<TpetraScalar, TpetraLocalOrdinal,
      TpetraGlobalOrdinal, TpetraNode>;
  using MueLuHierarchy = MueLu::Hierarchy<TpetraScalar, TpetraLocalOrdinal,
      TpetraGlobalOrdinal, TpetraNode>;

  struct StructuredPermutation
  {
    std::vector<int> old_dof_gid_for_new_dof_gid;
    std::vector<int> old_node_gid_for_new_node_gid;
    std::unordered_map<int, int> new_dof_gid_for_old_dof_gid;
    Teuchos::Array<int> local_nodes_per_dimension;
  };

  std::vector<double> unique_sorted_coordinates(
      const Epetra_MultiVector& coordinates, const int component)
  {
    std::vector<double> values(coordinates.MyLength());
    for (int lid = 0; lid < coordinates.MyLength(); ++lid) values[lid] = coordinates[component][lid];
    std::sort(values.begin(), values.end());

    const double scale = values.empty() ? 1.0 : std::max(1.0, std::abs(values.back() - values.front()));
    const double tolerance = 1.0e-10 * scale;
    std::vector<double> unique;
    for (const double value : values)
      if (unique.empty() || std::abs(value - unique.back()) > tolerance) unique.push_back(value);
    return unique;
  }

  int coordinate_index(const std::vector<double>& values, const double value)
  {
    const auto upper = std::lower_bound(values.begin(), values.end(), value);
    auto closest = upper;
    if (closest == values.end() ||
        (closest != values.begin() && std::abs(value - *(closest - 1)) < std::abs(value - *closest)))
      --closest;

    const double scale = values.empty() ? 1.0 : std::max(1.0, std::abs(values.back() - values.front()));
    FOUR_C_ASSERT_ALWAYS(closest != values.end() && std::abs(value - *closest) <= 1.0e-10 * scale,
        "Coordinate {} does not lie on the inferred tensor-product grid.", value);
    return static_cast<int>(std::distance(values.begin(), closest));
  }

  Teuchos::Array<int> get_structured_dimensions(const Teuchos::ParameterList& muelu_parameters)
  {
    FOUR_C_ASSERT_ALWAYS(muelu_parameters.isSublist("user data"),
        "The Tpetra structured backend requires a MueLu 'user data' sublist.");
    const auto& user_data = muelu_parameters.sublist("user data");
    constexpr const char* typed_name = "Array<LO> lNodesPerDim";
    if (user_data.isType<Teuchos::Array<int>>(typed_name))
      return user_data.get<Teuchos::Array<int>>(typed_name);
    if (user_data.isType<Teuchos::Array<int>>("lNodesPerDim"))
      return user_data.get<Teuchos::Array<int>>("lNodesPerDim");
    FOUR_C_THROW(
        "The Tpetra structured backend requires 'Array<LO> lNodesPerDim' in MueLu user data.");
  }

  StructuredPermutation build_structured_permutation(const Epetra_CrsMatrix& matrix,
      const Epetra_MultiVector& coordinates, const int dofs_per_node,
      const Teuchos::Array<int>& local_nodes_per_dimension)
  {
    FOUR_C_ASSERT_ALWAYS(local_nodes_per_dimension.size() == 3,
        "The current Tpetra structured backend requires exactly three local mesh dimensions.");
    FOUR_C_ASSERT_ALWAYS(dofs_per_node > 0, "The number of degrees of freedom per node is invalid.");
    FOUR_C_ASSERT_ALWAYS(matrix.NumMyRows() == coordinates.MyLength() * dofs_per_node,
        "The matrix has {} local rows, but the coordinate map and block size describe {} rows.",
        matrix.NumMyRows(), coordinates.MyLength() * dofs_per_node);

    const long long expected_nodes = static_cast<long long>(local_nodes_per_dimension[0]) *
                                     local_nodes_per_dimension[1] * local_nodes_per_dimension[2];
    FOUR_C_ASSERT_ALWAYS(expected_nodes == coordinates.MyLength(),
        "lNodesPerDim describes {} nodes, but the coordinate vector contains {} nodes.",
        expected_nodes, coordinates.MyLength());

    std::array<std::vector<double>, 3> axis_values;
    for (int axis = 0; axis < 3; ++axis)
      axis_values[axis] = unique_sorted_coordinates(coordinates, axis);

    // Match logical structured dimensions to physical coordinate axes. This supports the tutorial
    // convention {56,12,12}, where logical x is the beam's physical z direction.
    std::array<int, 3> physical_axis_for_logical_axis{-1, -1, -1};
    std::array<bool, 3> physical_axis_used{false, false, false};
    for (int logical_axis = 0; logical_axis < 3; ++logical_axis)
      for (int physical_axis = 0; physical_axis < 3; ++physical_axis)
        if (!physical_axis_used[physical_axis] &&
            static_cast<int>(axis_values[physical_axis].size()) ==
                local_nodes_per_dimension[logical_axis])
        {
          physical_axis_for_logical_axis[logical_axis] = physical_axis;
          physical_axis_used[physical_axis] = true;
          break;
        }

    FOUR_C_ASSERT_ALWAYS(std::none_of(physical_axis_for_logical_axis.begin(),
                             physical_axis_for_logical_axis.end(), [](const int axis) { return axis < 0; }),
        "Could not match lNodesPerDim {{{}, {}, {}}} to the coordinate-grid dimensions "
        "{{{}, {}, {}}}.",
        local_nodes_per_dimension[0], local_nodes_per_dimension[1],
        local_nodes_per_dimension[2], axis_values[0].size(), axis_values[1].size(),
        axis_values[2].size());

    StructuredPermutation permutation;
    permutation.local_nodes_per_dimension = local_nodes_per_dimension;
    permutation.old_node_gid_for_new_node_gid.assign(coordinates.MyLength(), -1);

    for (int old_node_lid = 0; old_node_lid < coordinates.MyLength(); ++old_node_lid)
    {
      std::array<int, 3> logical_index{};
      for (int logical_axis = 0; logical_axis < 3; ++logical_axis)
      {
        const int physical_axis = physical_axis_for_logical_axis[logical_axis];
        logical_index[logical_axis] =
            coordinate_index(axis_values[physical_axis], coordinates[physical_axis][old_node_lid]);
      }
      const int new_node_gid = logical_index[0] + local_nodes_per_dimension[0] *
                                                     (logical_index[1] +
                                                         local_nodes_per_dimension[1] * logical_index[2]);
      FOUR_C_ASSERT_ALWAYS(permutation.old_node_gid_for_new_node_gid[new_node_gid] == -1,
          "Two 4C nodes map to structured node {}. The coordinate grid is not one-to-one.",
          new_node_gid);
      permutation.old_node_gid_for_new_node_gid[new_node_gid] = coordinates.Map().GID(old_node_lid);
    }

    const int num_dofs = matrix.NumMyRows();
    permutation.old_dof_gid_for_new_dof_gid.assign(num_dofs, -1);
    permutation.new_dof_gid_for_old_dof_gid.reserve(num_dofs);
    const int old_dof_index_base = matrix.RowMap().IndexBase();
    const int old_node_index_base = coordinates.Map().IndexBase();
    std::unordered_map<int, int> new_node_gid_for_old_node_gid;
    new_node_gid_for_old_node_gid.reserve(coordinates.MyLength());
    for (int new_node_gid = 0; new_node_gid < coordinates.MyLength(); ++new_node_gid)
      new_node_gid_for_old_node_gid.emplace(
          permutation.old_node_gid_for_new_node_gid[new_node_gid], new_node_gid);

    for (int old_dof_lid = 0; old_dof_lid < matrix.NumMyRows(); ++old_dof_lid)
    {
      const int old_dof_gid = matrix.RowMap().GID(old_dof_lid);
      const int zero_based_dof = old_dof_gid - old_dof_index_base;
      const int component = zero_based_dof % dofs_per_node;
      const int old_node_gid = zero_based_dof / dofs_per_node + old_node_index_base;
      const auto node = new_node_gid_for_old_node_gid.find(old_node_gid);
      FOUR_C_ASSERT_ALWAYS(node != new_node_gid_for_old_node_gid.end(),
          "Matrix DOF {} refers to node {}, which is absent from the coordinate map.", old_dof_gid,
          old_node_gid);
      const int new_dof_gid = node->second * dofs_per_node + component;
      permutation.old_dof_gid_for_new_dof_gid[new_dof_gid] = old_dof_gid;
      permutation.new_dof_gid_for_old_dof_gid.emplace(old_dof_gid, new_dof_gid);
    }
    FOUR_C_ASSERT_ALWAYS(std::none_of(permutation.old_dof_gid_for_new_dof_gid.begin(),
                             permutation.old_dof_gid_for_new_dof_gid.end(),
                             [](const int gid) { return gid < 0; }),
        "The structured DOF permutation is incomplete.");
    return permutation;
  }

  Teuchos::RCP<TpetraMultiVector> permute_dof_multivector_to_tpetra(
      const Epetra_MultiVector& source, const Teuchos::RCP<const TpetraMap>& map,
      const StructuredPermutation& permutation)
  {
    FOUR_C_ASSERT_ALWAYS(source.MyLength() == static_cast<int>(map->getLocalNumElements()),
        "DOF multivector length does not match the Tpetra map.");
    auto target = Teuchos::rcp(new TpetraMultiVector(map, source.NumVectors(), false));
    auto target_view = target->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (int new_gid = 0; new_gid < source.MyLength(); ++new_gid)
    {
      const int old_lid = source.Map().LID(permutation.old_dof_gid_for_new_dof_gid[new_gid]);
      FOUR_C_ASSERT_ALWAYS(old_lid >= 0, "Source multivector is missing a permuted DOF.");
      for (int vector = 0; vector < source.NumVectors(); ++vector)
        target_view(new_gid, vector) = source[vector][old_lid];
    }
    return target;
  }

  Teuchos::RCP<TpetraMultiVector> permute_node_multivector_to_tpetra(
      const Epetra_MultiVector& source, const Teuchos::RCP<const TpetraMap>& map,
      const StructuredPermutation& permutation)
  {
    FOUR_C_ASSERT_ALWAYS(source.MyLength() == static_cast<int>(map->getLocalNumElements()),
        "Node multivector length does not match the Tpetra map.");
    auto target = Teuchos::rcp(new TpetraMultiVector(map, source.NumVectors(), false));
    auto target_view = target->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (int new_gid = 0; new_gid < source.MyLength(); ++new_gid)
    {
      const int old_lid = source.Map().LID(permutation.old_node_gid_for_new_node_gid[new_gid]);
      FOUR_C_ASSERT_ALWAYS(old_lid >= 0, "Source coordinate vector is missing a permuted node.");
      for (int vector = 0; vector < source.NumVectors(); ++vector)
        target_view(new_gid, vector) = source[vector][old_lid];
    }
    return target;
  }
}  // namespace

struct Core::LinearSolver::IterativeSolver::TpetraSolverData
{
  StructuredPermutation permutation;
  Teuchos::RCP<const TpetraMap> dof_map;
  Teuchos::RCP<const TpetraMap> node_map;
  Teuchos::RCP<TpetraMatrix> matrix;
  Teuchos::RCP<TpetraMultiVector> rhs;
  Teuchos::RCP<MueLuHierarchy> hierarchy;
  Teuchos::RCP<TpetraOperator> preconditioner;
};

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::IterativeSolver::IterativeSolver(MPI_Comm comm, Teuchos::ParameterList& params)
    : comm_(comm), params_(params)
{
}

Core::LinearSolver::IterativeSolver::~IterativeSolver() = default;

bool Core::LinearSolver::IterativeSolver::uses_tpetra_backend() const
{
  const std::string backend = params().get("backend", std::string("Epetra"));
  FOUR_C_ASSERT_ALWAYS(backend == "Epetra" || backend == "Tpetra",
      "Unknown Belos backend '{}'. Supported values are Epetra and Tpetra.", backend);
  return backend == "Tpetra";
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::IterativeSolver::reset()
{
  preconditioner_ = nullptr;
  tpetra_data_.reset();
  active_dof_map_ = nullptr;
  a_ = nullptr;
  b_ = nullptr;
  ncall_ = 0;
  numiters_ = -1;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::IterativeSolver::setup(std::shared_ptr<Core::LinAlg::SparseOperator> A,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const bool refactor,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
{
  if (!params().isSublist("Belos Parameters")) FOUR_C_THROW("Do not have belos parameter list");
  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");

  a_ = A;
  b_ = b;

  if (uses_tpetra_backend())
  {
    FOUR_C_ASSERT_ALWAYS(projector == nullptr,
        "The experimental Tpetra backend does not yet support a linear-system projector.");
    setup_tpetra();
    return;
  }

  if (Core::Communication::my_mpi_rank(comm_) == 0)
    std::cout << "*******************************************************" << std::endl;

  const int reuse = belist.get("reuse", 0);
  const bool create = !allow_reuse_preconditioner(reuse);
  if (create)
  {
    ncall_ = 0;
    preconditioner_ = create_preconditioner(belist, projector);
  }

  if (create)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      std::cout << "Compute preconditioner " << std::endl;
      std::cout << "*******************************************************" << std::endl;
    }

    preconditioner_->setup(*a_, *b_);
  }
  else
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      std::cout << "Reuse preconditioner (reused for " << ncall() << " steps)" << std::endl;
      std::cout << "*******************************************************" << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Core::LinearSolver::IterativeSolver::solve(Core::LinAlg::MultiVector<double>& x)
{
  if (uses_tpetra_backend()) return solve_tpetra(x);

  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");

  Teuchos::RCP<Epetra_Operator> a =
      Teuchos::make_rcp<Core::LinearSolver::EpetraOperatorAdapter>(a_);

  auto problem = Teuchos::make_rcp<Belos::LinearProblem<double, BelosVectorType, BelosMatrixType>>(
      a, Teuchos::rcpFromRef(x.get_epetra_multi_vector()),
      Teuchos::rcpFromRef(b_->get_epetra_multi_vector()));

  if (preconditioner_ != nullptr)
  {
    auto belosPrec =
        Teuchos::make_rcp<Belos::EpetraPrecOp>(Teuchos::rcp(preconditioner_->prec_operator()));
    problem->setRightPrec(belosPrec);
  }

  const bool set = problem->setProblem();
  if (set == false)
    FOUR_C_THROW("Core::LinearSolver::BelosSolver: Iterative solver failed to set up correctly.");

  std::shared_ptr<Belos::SolverManager<double, BelosVectorType, BelosMatrixType>> newSolver;

  if (belist.isParameter("SOLVER_XML_FILE"))
  {
    const std::string xmlFileName = belist.get<std::string>("SOLVER_XML_FILE");
    Teuchos::ParameterList belosParams;
    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName,
        Teuchos::Ptr<Teuchos::ParameterList>(&belosParams),
        *Core::Communication::to_teuchos_comm<int>(comm_));

    if (belosParams.isSublist("GMRES"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("GMRES"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver =
          std::make_shared<Belos::PseudoBlockGmresSolMgr<double, BelosVectorType, BelosMatrixType>>(
              problem, belosSolverList);
    }
    else if (belosParams.isSublist("CG"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("CG"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver =
          std::make_shared<Belos::PseudoBlockCGSolMgr<double, BelosVectorType, BelosMatrixType>>(
              problem, belosSolverList);
    }
    else if (belosParams.isSublist("BiCGSTAB"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("BiCGSTAB"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver = std::make_shared<Belos::BiCGStabSolMgr<double, BelosVectorType, BelosMatrixType>>(
          problem, belosSolverList);
    }
    else
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Unknown iterative solver solver type chosen.");
  }
  else
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "WARNING: The linear solver input parameters from the input file will be "
                   "removed soon. Switch to an appropriate xml-file version."
                << std::endl;

    std::string solverType = belist.get<std::string>("Solver Type");
    if (solverType == "GMRES")
      newSolver =
          std::make_shared<Belos::BlockGmresSolMgr<double, BelosVectorType, BelosMatrixType>>(
              problem, Teuchos::rcpFromRef(belist));
    else if (solverType == "CG")
      newSolver = std::make_shared<Belos::BlockCGSolMgr<double, BelosVectorType, BelosMatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else if (solverType == "BiCGSTAB")
      newSolver = std::make_shared<Belos::BiCGStabSolMgr<double, BelosVectorType, BelosMatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Unknown iterative solver solver type chosen.");
  }

  Belos::ReturnType ret = newSolver->solve();

  numiters_ = newSolver->getNumIters();
  ncall_ += 1;

  // Check and communicate failed convergence to user
  {
    int my_error = 0;
    if (ret != Belos::Converged) my_error = 1;
    int glob_error = 0;
    glob_error = Core::Communication::sum_all(my_error, comm_);

    if (glob_error > 0)
    {
      if (belist.get<bool>("THROW_IF_UNCONVERGED"))
      {
        FOUR_C_THROW("Core::LinearSolver::BelosSolver: Iterative solver did not converge.");
      }
      else
      {
        if (Core::Communication::my_mpi_rank(comm_) == 0)
        {
          std::cout
              << "Core::LinearSolver::BelosSolver: WARNING: Iterative solver did not converge."
              << std::endl;
        }
      }
    }
  }

  return 0;
}

void Core::LinearSolver::IterativeSolver::setup_tpetra()
{
  FOUR_C_ASSERT_ALWAYS(Core::Communication::num_mpi_ranks(comm_) == 1,
      "The structured Tpetra conversion currently supports one MPI rank. A multi-rank run needs "
      "a Cartesian repartition in addition to the lexicographic permutation.");
  FOUR_C_ASSERT_ALWAYS(params().isSublist("MueLu Parameters"),
      "The Tpetra backend currently requires the MueLu preconditioner.");

  auto sparse_matrix =
      Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(Teuchos::rcpFromRef(*a_));
  FOUR_C_ASSERT_ALWAYS(!sparse_matrix.is_null(),
      "The Tpetra backend currently supports only Core::LinAlg::SparseMatrix, not block matrices.");

  auto& muelu_solver_parameters = params().sublist("MueLu Parameters");
  FOUR_C_ASSERT_ALWAYS(muelu_solver_parameters.isParameter("PRECONDITIONER_XML_FILE"),
      "PRECONDITIONER_XML_FILE is required for the Tpetra MueLu backend.");
  const std::string xml_file =
      muelu_solver_parameters.get<std::string>("PRECONDITIONER_XML_FILE");

  Teuchos::ParameterList muelu_parameters;
  auto comm = Core::Communication::to_teuchos_comm<int>(comm_);
  auto tpetra_comm = Teuchos::rcp(comm);
  Teuchos::updateParametersFromXmlFileAndBroadcast(
      xml_file, Teuchos::ptr(&muelu_parameters), *comm);
  validate_coarse_solver(muelu_parameters);

  const int dofs_per_node = muelu_solver_parameters.get<int>("PDE equations");
  FOUR_C_ASSERT_ALWAYS(dofs_per_node == 3,
      "The current structured Tpetra permutation is implemented for three displacement DOFs per "
      "node, but the solver reports {}.",
      dofs_per_node);

  if (muelu_solver_parameters.isParameter("Coordinates"))
    Core::LinearSolver::Parameters::fix_coordinates(
        "MueLu/Tpetra", sparse_matrix->row_map(), muelu_solver_parameters);

  const auto coordinates =
      muelu_solver_parameters
          .get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", nullptr);
  const auto nullspace =
      muelu_solver_parameters
          .get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullptr);
  FOUR_C_ASSERT_ALWAYS(coordinates != nullptr, "MueLu coordinates are unavailable.");
  FOUR_C_ASSERT_ALWAYS(nullspace != nullptr, "MueLu nullspace is unavailable.");

  const auto dimensions = get_structured_dimensions(muelu_parameters);
  const Epetra_CrsMatrix& epetra_matrix = sparse_matrix->epetra_matrix();

  auto data = std::make_unique<TpetraSolverData>();
  {
    TEUCHOS_FUNC_TIME_MONITOR(
        "Core::LinAlg::Solver: Tpetra structured permutation construction");
    data->permutation = build_structured_permutation(epetra_matrix,
        coordinates->get_epetra_multi_vector(), dofs_per_node, dimensions);
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver: Epetra to Tpetra matrix conversion");
    data->dof_map = Teuchos::rcp(new TpetraMap(
        static_cast<Tpetra::global_size_t>(epetra_matrix.NumGlobalRows()), 0, tpetra_comm));
    data->node_map = Teuchos::rcp(new TpetraMap(
        static_cast<Tpetra::global_size_t>(coordinates->global_length()), 0, tpetra_comm));
    data->matrix = Teuchos::rcp(new TpetraMatrix(data->dof_map, epetra_matrix.MaxNumEntries()));

    std::vector<double> old_values(epetra_matrix.MaxNumEntries());
    std::vector<int> old_columns(epetra_matrix.MaxNumEntries());
    Teuchos::Array<double> new_values(epetra_matrix.MaxNumEntries());
    Teuchos::Array<int> new_columns(epetra_matrix.MaxNumEntries());
    for (int new_row_gid = 0; new_row_gid < epetra_matrix.NumMyRows(); ++new_row_gid)
    {
      const int old_row_gid = data->permutation.old_dof_gid_for_new_dof_gid[new_row_gid];
      int num_entries = 0;
      const int error = epetra_matrix.ExtractGlobalRowCopy(old_row_gid,
          epetra_matrix.MaxNumEntries(), num_entries, old_values.data(), old_columns.data());
      FOUR_C_ASSERT_ALWAYS(error == 0,
          "Epetra failed to extract global matrix row {} (error {}).", old_row_gid, error);
      new_values.resize(num_entries);
      new_columns.resize(num_entries);
      for (int entry = 0; entry < num_entries; ++entry)
      {
        const auto new_column =
            data->permutation.new_dof_gid_for_old_dof_gid.find(old_columns[entry]);
        FOUR_C_ASSERT_ALWAYS(new_column != data->permutation.new_dof_gid_for_old_dof_gid.end(),
            "Matrix column DOF {} is absent from the structured permutation.", old_columns[entry]);
        new_columns[entry] = new_column->second;
        new_values[entry] = old_values[entry];
      }
      data->matrix->insertGlobalValues(new_row_gid, new_columns(), new_values());
    }
    data->matrix->fillComplete(data->dof_map, data->dof_map);
  }

  Teuchos::RCP<Xpetra::CrsMatrix<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal,
      TpetraNode>> xpetra_crs =
      Teuchos::rcp(new Xpetra::TpetraCrsMatrix<TpetraScalar, TpetraLocalOrdinal,
          TpetraGlobalOrdinal, TpetraNode>(data->matrix));
  Teuchos::RCP<XpetraMatrix> xpetra_matrix = Teuchos::rcp(
      new Xpetra::CrsMatrixWrap<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal,
          TpetraNode>(xpetra_crs));
  xpetra_matrix->SetFixedBlockSize(dofs_per_node);

  Teuchos::RCP<TpetraMultiVector> tpetra_coordinates;
  Teuchos::RCP<TpetraMultiVector> tpetra_nullspace;
  {
    TEUCHOS_FUNC_TIME_MONITOR(
        "Core::LinAlg::Solver: Epetra to Tpetra multivector conversion");
    tpetra_coordinates = permute_node_multivector_to_tpetra(
        coordinates->get_epetra_multi_vector(), data->node_map, data->permutation);
    tpetra_nullspace = permute_dof_multivector_to_tpetra(
        nullspace->get_epetra_multi_vector(), data->dof_map, data->permutation);
    data->rhs = permute_dof_multivector_to_tpetra(
        b_->get_epetra_multi_vector(), data->dof_map, data->permutation);
  }

  muelu_parameters.set("number of equations", dofs_per_node);
  auto& user_data = muelu_parameters.sublist("user data");
  user_data.set<Teuchos::RCP<XpetraMultiVector>>(
      "Coordinates", Xpetra::toXpetra(tpetra_coordinates));
  user_data.set<Teuchos::RCP<XpetraMultiVector>>(
      "Nullspace", Xpetra::toXpetra(tpetra_nullspace));

  {
    TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver: Tpetra MueLu hierarchy construction");
    data->hierarchy = MueLu::CreateXpetraPreconditioner<TpetraScalar, TpetraLocalOrdinal,
        TpetraGlobalOrdinal, TpetraNode>(xpetra_matrix, muelu_parameters);
    data->hierarchy->IsPreconditioner(true);
    data->preconditioner = Teuchos::rcp(
        new MueLu::TpetraOperator<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal,
            TpetraNode>(data->hierarchy));
  }

  tpetra_data_ = std::move(data);
  ncall_ = 0;
}

int Core::LinearSolver::IterativeSolver::solve_tpetra(
    Core::LinAlg::MultiVector<double>& x)
{
  FOUR_C_ASSERT_ALWAYS(tpetra_data_ != nullptr, "The Tpetra solver has not been set up.");
  auto tpetra_x = permute_dof_multivector_to_tpetra(
      x.get_epetra_multi_vector(), tpetra_data_->dof_map, tpetra_data_->permutation);

  auto problem = Teuchos::rcp(
      new Belos::LinearProblem<TpetraScalar, TpetraMultiVector, TpetraOperator>(
          tpetra_data_->matrix, tpetra_x, tpetra_data_->rhs));
  problem->setRightPrec(tpetra_data_->preconditioner);
  FOUR_C_ASSERT_ALWAYS(problem->setProblem(),
      "Core::LinearSolver::BelosSolver: Tpetra problem setup failed.");

  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");
  Teuchos::ParameterList belos_parameters;
  if (belist.isParameter("SOLVER_XML_FILE"))
  {
    Teuchos::updateParametersFromXmlFileAndBroadcast(belist.get<std::string>("SOLVER_XML_FILE"),
        Teuchos::ptr(&belos_parameters),
        *Core::Communication::to_teuchos_comm<int>(comm_));
  }
  else
  {
    belos_parameters.setParameters(belist);
  }

  std::shared_ptr<Belos::SolverManager<TpetraScalar, TpetraMultiVector, TpetraOperator>> solver;
  if (belos_parameters.isSublist("GMRES"))
  {
    auto solver_parameters = Teuchos::rcpFromRef(belos_parameters.sublist("GMRES"));
    if (belist.isParameter("Convergence Tolerance"))
      solver_parameters->set(
          "Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
    solver = std::make_shared<Belos::PseudoBlockGmresSolMgr<TpetraScalar, TpetraMultiVector,
        TpetraOperator>>(problem, solver_parameters);
  }
  else if (belos_parameters.isSublist("CG"))
  {
    auto solver_parameters = Teuchos::rcpFromRef(belos_parameters.sublist("CG"));
    if (belist.isParameter("Convergence Tolerance"))
      solver_parameters->set(
          "Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
    solver = std::make_shared<Belos::PseudoBlockCGSolMgr<TpetraScalar, TpetraMultiVector,
        TpetraOperator>>(problem, solver_parameters);
  }
  else if (belos_parameters.isSublist("BiCGSTAB"))
  {
    auto solver_parameters = Teuchos::rcpFromRef(belos_parameters.sublist("BiCGSTAB"));
    if (belist.isParameter("Convergence Tolerance"))
      solver_parameters->set(
          "Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
    solver = std::make_shared<Belos::BiCGStabSolMgr<TpetraScalar, TpetraMultiVector,
        TpetraOperator>>(problem, solver_parameters);
  }
  else
  {
    FOUR_C_THROW("Unknown Tpetra Belos solver type in solver XML file.");
  }

  const Belos::ReturnType result = solver->solve();
  numiters_ = solver->getNumIters();
  ++ncall_;

  {
    TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver: Tpetra to Epetra solution conversion");
    auto tpetra_x_view = tpetra_x->getLocalViewHost(Tpetra::Access::ReadOnly);
    Epetra_MultiVector& epetra_x = x.get_epetra_multi_vector();
    for (int new_gid = 0; new_gid < epetra_x.MyLength(); ++new_gid)
    {
      const int old_lid =
          epetra_x.Map().LID(tpetra_data_->permutation.old_dof_gid_for_new_dof_gid[new_gid]);
      FOUR_C_ASSERT_ALWAYS(old_lid >= 0, "Solution vector is missing a permuted DOF.");
      for (int vector = 0; vector < epetra_x.NumVectors(); ++vector)
        epetra_x[vector][old_lid] = tpetra_x_view(new_gid, vector);
    }
  }

  if (result != Belos::Converged)
  {
    if (belist.get<bool>("THROW_IF_UNCONVERGED"))
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Tpetra iterative solver did not converge.");
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "Core::LinearSolver::BelosSolver: WARNING: Tpetra iterative solver did not "
                   "converge."
                << std::endl;
  }
  return 0;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
bool Core::LinearSolver::IterativeSolver::allow_reuse_preconditioner(const int reuse)
{
  // first, check some parameters with information that has to be updated
  Teuchos::ParameterList& linSysParams = params().sublist("Belos Parameters");

  bool bAllowReuse = linSysParams.get<bool>("reuse preconditioner", true);
  int max_stall_iter = linSysParams.get<int>("max linear iterations for stall");

  // 1. check if we are allowed to reuse the preconditioner over several nonlinear solves
  if (not ncall() or not reuse or ncall() % reuse == 0)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "Recomputation due to reaching " << ncall() << " nonlinear steps." << std::endl;

    bAllowReuse = false;
  }

  // 2. check if the number of iterations exceeds the given one for stalling of the solver
  if (bAllowReuse and get_num_iters() > max_stall_iter)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "Recomputation due to linear solver stalling." << std::endl;

    bAllowReuse = false;
  }

  // here, each processor has its own local decision made
  // bAllowReuse = true -> preconditioner can be reused
  // bAllowReuse = false -> preconditioner has to be recomputed
  // If one or more processors decide that the preconditioner has to be recomputed
  // all of the processors have to recompute it

  // synchronize results of all processors
  // all processors have to do the same (either recompute preconditioner or allow reusing it)
  int nProc = Core::Communication::num_mpi_ranks(comm_);
  int lAllowReuse = bAllowReuse == true ? 1 : 0;
  int gAllowReuse = 0;
  gAllowReuse = Core::Communication::sum_all(lAllowReuse, comm_);

  return gAllowReuse == nProc;
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase>
Core::LinearSolver::IterativeSolver::create_preconditioner(Teuchos::ParameterList& solverlist,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  1.1)   create_preconditioner");

  std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase> preconditioner;

  if (params().isSublist("IFPACK Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::IFPACKPreconditioner>(params());
  }
  else if (params().isSublist("MueLu Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::MueLuPreconditioner>(params());
  }
  else if (params().isSublist("Teko Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::TekoPreconditioner>(params());
  }
  else if (params().isSublist("AMGnxn Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::AmGnxnPreconditioner>(params());
  }
  else
    FOUR_C_THROW("Unknown preconditioner chosen for iterative linear solver.");

  if (projector != nullptr)
  {
    preconditioner =
        std::make_shared<Core::LinearSolver::ProjectionPreconditioner>(preconditioner, projector);
  }

  return preconditioner;
}

FOUR_C_NAMESPACE_CLOSE
