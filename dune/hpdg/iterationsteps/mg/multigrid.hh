#ifndef DUNE_HPDG_ITERATIONSTEPS_MULTIGRID_HH
#define DUNE_HPDG_ITERATIONSTEPS_MULTIGRID_HH

#include <functional>
#include <vector>
#include <memory>


// This is just Ansgar Burchardt's parallel MG code w/o the parallel parts
namespace Dune {
namespace HPDG {

template<typename Vector>
using Smoother = typename std::function<void(Vector&, const Vector&)>;

template<typename Vector>
using TransferOperator = typename std::function<void(Vector&, const Vector&)>;

template<typename Vector>
using CoarseSolver = typename std::function<void(Vector&, const Vector&)>;

template<typename Vector>
using Operator = typename std::function<void(Vector&, const Vector&)>;

template<typename Vector>
class Multigrid
{
public:
  struct LevelOperations
  {
    Operator<Vector> apply;

    // smoother
    Smoother<Vector> preSmoother;
    Smoother<Vector> postSmoother;

    // grid transfer
    /**
     * Restriction from this level to the next coarser level.
     *
     * Not used on the coarsest level.
     */
    TransferOperator<Vector> restrict;
    /**
     * Prolongation from the next coarser level to this level.
     *
     * Not used on the coarsest level.
     */
    TransferOperator<Vector> prolong;

    unsigned int preSmootherSteps = 5;
    unsigned int postSmootherSteps = 5;
  };

  void apply(Vector& x, Vector& b) const;

  void levelOperations(const std::vector<LevelOperations>& levelOperations);
  void coarseSolver(const CoarseSolver<Vector>& coarseSolver);

private:
  struct State {
    std::vector<Vector> x;
    std::vector<Vector> r;
    Vector tmp1, tmp2;
  };

  unsigned int mu_ = 1;
  CoarseSolver<Vector> coarseSolver_;
  std::vector<LevelOperations> levelOperations_;

  void applySmoother(State& state,
                     const LevelOperations& op,
                     const Smoother<Vector>& smoother,
                     unsigned steps,
                     Vector& x,
                     Vector& r) const;

  void applyLevel(State& state, unsigned level) const;
};

template<typename Vector, typename IterationStep>
Smoother<Vector>
smootherFromIterationStep(std::shared_ptr<IterationStep> iterationStep,
                          const typename Vector::field_type& dampening = typename Vector::field_type(1))
{
  return [=](auto& x, const auto& b) mutable {
    //x.resize(b.size());
    x=b;
    x = 0;
    iterationStep->apply(x, b);
    x *= dampening;
  };
}
template<typename Vector, typename IterationStep>
Smoother<Vector>
smootherFromIterationStep2(std::shared_ptr<IterationStep> iterationStep,
                          const typename Vector::field_type& dampening = typename Vector::field_type(1))
{
  return [=](auto& x, const auto& b) mutable {
    iterationStep->setProblem(x);
    iterationStep->rhs_=&b;
    iterationStep->preprocess();
    iterationStep->iterate();
    //iterationStep->apply(x, b);
    x *= dampening;
  };
}
template<typename Vector, typename MultigridTransfer>
TransferOperator<Vector>
restrictFromMultigridTransfer(std::shared_ptr<MultigridTransfer> multigridTransfer)
{
  return [=](auto& y, const auto& x) {
    multigridTransfer->restrict(x, y);
  };
}

template<typename Vector, typename MultigridTransfer>
TransferOperator<Vector>
prolongFromMultigridTransfer(std::shared_ptr<MultigridTransfer> multigridTransfer)
{
  return [=](auto& y, const auto& x) {
    multigridTransfer->prolong(x, y);
  };
}

template<typename Solver>
CoarseSolver<typename Solver::domain_type>
coarseSolverFromSolver(Solver& solver)
{
  return [&](auto& x, const auto& b) {
    //InverseOperatorResult statistics;
    //auto tmp = b;
    //solver.apply(x, tmp, statistics);
  };
}

template<typename Vector, typename Matrix>
Operator<Vector>
operatorFromMatrix(std::shared_ptr<Matrix> matrix)
{
  return [=](Vector& y, const Vector& x) {
    matrix->mv(x, y);
  };
}

template<typename Vector, typename Matrix>
Operator<Vector>
operatorFromMatrix(const Matrix& matrix)
{
  return [&](Vector& y, const Vector& x) {
    matrix.mv(x, y);
  };
}

} /* namespace ParMG */
} /* namespace Dune */

#include "multigrid_impl.hh"

#endif
