#ifndef DUNE_HPDG_BUILDINGBLOCKS_SOLVE
#define DUNE_HPDG_BUILDINGBLOCKS_SOLVE

#include <dune/common/shared_ptr.hh>
#include <dune/common/stringutility.hh>
#include <dune/hpdg/buildingblocks/details.hh>
#include <dune/hpdg/iterationsteps/solversetup.hh>
#include <dune/solvers/common/defaultbitvector.hh>
#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/tnnmg/functionals/bcqfconstrainedlinearization.hh>
#include <dune/tnnmg/functionals/boxconstrainedquadraticfunctional.hh>
#include <dune/tnnmg/iterationsteps/nonlineargsstep.hh>
#include <dune/tnnmg/iterationsteps/tnnmgstep.hh>
#include <dune/tnnmg/localsolvers/scalarobstaclesolver.hh>
#include <dune/tnnmg/projections/obstacledefectprojection.hh>
#include <memory>

namespace Dune::HPDG::BuildingBlocks {

/**
 * @brief Solves the quadratic obstacle problem using TNNMG up to given
 * accuracy.
 *
 * @tparam Basis
 * @tparam Matrix
 * @tparam Vector
 * @param basis
 * @param matrix
 * @param rhs
 * @param x Serves as initial iterate
 * @param lower
 * @param upper
 * @param tol
 * @param maxIterationSteps
 * @return auto
 */
template<typename Basis, typename Matrix, typename Vector>
auto
solveObstacle(const Basis& basis,
              const Matrix& matrix,
              const Vector& rhs,
              Vector x,
              const Vector& lower,
              const Vector& upper,
              double tol = 1e-8,
              int maxIterationSteps = 100,
              const Vector* referenceSolution = nullptr)
{
  using namespace Dune;
  auto mgw = HPDG::MultigridSetup::multigridSolver<Vector>(
    basis.gridView().grid(), Detail::maxDegree(basis), matrix);
  mgw->setProblem(matrix, x, rhs);
  mgw->preprocess();

  auto norm =
    std::make_shared<Dune::Solvers::EnergyNorm<Matrix, Vector>>(matrix);

  const auto dummyBits = HPDG::BuildingBlocks::Detail::allFalseBitVector(x);
  using BitVector = std::decay_t<decltype(dummyBits)>;

  // TNNMG
  // 1. functional
  using Functional = Dune::TNNMG::
    BoxConstrainedQuadraticFunctional<Matrix, Vector, Vector, Vector, double>;
  auto J = Functional(matrix, rhs, lower, upper);

  // 2. nonlinear Gauss-Seidel
  auto localSolver = gaussSeidelLocalSolver(
    gaussSeidelLocalSolver(Dune::TNNMG::ScalarObstacleSolver()));

  using NonlinearSmoother =
    Dune::TNNMG::NonlinearGSStep<Functional, decltype(localSolver), BitVector>;
  auto nonlinearSmoother =
    std::make_shared<NonlinearSmoother>(J, x, localSolver);

  // 3. Linearization
  using Linearization =
    Dune::TNNMG::BoxConstrainedQuadraticFunctionalConstrainedLinearization<
      Functional,
      BitVector>;
  // 4. Projection
  using DefectProjection = Dune::TNNMG::ObstacleDefectProjection;
  // 5. Line search
  using LineSearchSolver = Dune::TNNMG::ScalarObstacleSolver;

  // pack everything into an TNNMG step
  using Step = Dune::TNNMG::TNNMGStep<Functional,
                                      BitVector,
                                      Linearization,
                                      DefectProjection,
                                      LineSearchSolver>;
  int mu = 1; // #multigrid steps in Newton step
  auto step = std::make_shared<Step>(
    J, x, nonlinearSmoother, mgw, mu, DefectProjection(), LineSearchSolver());
  step->setIgnore(dummyBits);

  using Solver = Dune::Solvers::LoopSolver<Vector, BitVector>;

  auto solver = Solver(step, maxIterationSteps, tol, norm, Solver::FULL, true, referenceSolution);
  solver.addCriterion(
    [&]() {
      return Dune::formatString("   % 12.5e", step->lastDampingFactor());
    },
    "   damping     ");

  solver.addCriterion(
    [&]() {
      auto count = 0;
      const auto& tr = step->linearization().truncated();
      for (const auto& b : tr) {
        for (const auto& e : b)
          count += e.all(); // This is a little hack: We use that the Bit"set"
                            // has size one here.
      }
      return Dune::formatString("   % 12d", count);
    },
    "   truncated   ");

  solver.preprocess();
  solver.solve();
  // return x;
  return std::make_pair(x, solver.iterationCount());
}

/**
 * @brief Solves the quadratic min. problem up to a given accuracy
 *
 * @tparam Basis
 * @tparam Matrix
 * @tparam Vector
 * @param basis
 * @param matrix
 * @param rhs
 * @param x Serves as initial iterate
 * @param tol
 * @param maxIterationSteps
 * @return auto
 */
template<typename Basis, typename Matrix, typename Vector>
auto
solveLinear(const Basis& basis,
              const Matrix& matrix,
              const Vector& rhs,
              Vector x,
              double tol = 1e-8,
              int maxIterationSteps = 100,
              const Vector* referenceSolution = nullptr)
{
  using namespace Dune;
  auto mgw = HPDG::MultigridSetup::multigridSolver<Vector>(
    basis.gridView().grid(), Detail::maxDegree(basis), matrix);
  mgw->setProblem(matrix, x, rhs);
  mgw->preprocess();

  auto norm =
    std::make_shared<Dune::Solvers::EnergyNorm<Matrix, Vector>>(matrix);

  const auto dummyBits = HPDG::BuildingBlocks::Detail::allFalseBitVector(x);
  using BitVector = std::decay_t<decltype(dummyBits)>;

  using Solver = Dune::Solvers::LoopSolver<Vector, BitVector>;

  auto solver = Solver(mgw, maxIterationSteps, tol, norm, Solver::FULL, true, referenceSolution);

  solver.preprocess();
  solver.solve();
  return std::make_pair(x, solver.iterationCount());
}
}
#endif