#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

#include <dune/common/timer.hh>

#include <dune/solvers/common/resize.hh>
#include "dune/solvers/iterationsteps/iterationstep.hh"
#include "dune/solvers/iterationsteps/lineariterationstep.hh"
#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/solvers/linearsolver.hh>

#include <dune/tnnmg/iterationsteps/linearcorrection.hh>

namespace Dune {
namespace TNNMG { // leave the namespace as it its, o/w you have to change it everywhere

/**
 * \brief One iteration of the TNNMG method
 *
 * \tparam F Functional to minimize
 * \tparam BV Bit-vector type for marking ignored components
 */
template<class F, class BV, class Linearization,
                                  class DefectProjection,
                                  class LineSearchSolver>
class ParTNNMGStep :
  public IterationStep<typename F::Vector, BV>
{
  using Base = IterationStep<typename F::Vector, BV>;

public:

  using Vector = typename F::Vector;
  using ConstrainedVector = typename Linearization::ConstrainedVector;
  using ConstrainedMatrix = typename Linearization::ConstrainedMatrix;
  using BitVector = typename Base::BitVector;
  using ConstrainedBitVector = typename Linearization::ConstrainedBitVector;
  using Functional = F;
  using IterativeSolver = Solvers::IterativeSolver< ConstrainedVector, Solvers::DefaultBitVector_t<ConstrainedVector> >;
  using LinearSolver = Solvers::LinearSolver< ConstrainedMatrix,  ConstrainedVector >;

  /** \brief Constructor with an iterative solver object for the linear correction
   * \param iterativeSolver This is a callback used to solve the constrained linearized system
   * \param projection This is a callback used to compute a projection into a defect-admissible set
   * \param lineSolver This is a callback used to minimize a directional restriction of the functional
   *        for computing a damping parameter
   */
#if 0
  ParTNNMGStep(const Functional& f,
            Vector& x,
            const std::function<void(Vector)>& collect,
            std::shared_ptr<IterationStep<Vector,BitVector> > nonlinearSmoother,
            std::shared_ptr<IterativeSolver> iterativeSolver,
            const DefectProjection& projection,
            const LineSearchSolver& lineSolver)
  : Base(x),
    f_(&f),
    collect_(collect),
    nonlinearSmoother_(nonlinearSmoother),
    linearCorrection_(makeLinearCorrection<ConstrainedMatrix>(iterativeSolver)),
    projection_(projection),
    lineSolver_(lineSolver)
  {}

  /** \brief Constructor with a linear solver object for the linear correction
   * \param linearSolver This is a callback used to solve the constrained linearized system
   * \param projection This is a callback used to compute a projection into a defect-admissible set
   * \param lineSolver This is a callback used to minimize a directional restriction of the functional
   *        for computing a damping parameter
   */
  ParTNNMGStep(const Functional& f,
            Vector& x,
            const std::function<void(Vector)>& collect,
            std::shared_ptr<IterationStep<Vector,BitVector> > nonlinearSmoother,
            std::shared_ptr<LinearSolver> linearSolver,
            const DefectProjection& projection,
            const LineSearchSolver& lineSolver)
  : Base(x),
    f_(&f),
    collect_(collect),
    nonlinearSmoother_(nonlinearSmoother),
    linearCorrection_(makeLinearCorrection(linearSolver)),
    projection_(projection),
    lineSolver_(lineSolver)
  {}

#endif
  /** \brief Constructor with a LinearIterationStep object for the linear correction
   * \param linearIterationStep This is a callback used to solve the constrained linearized system
   * \param projection This is a callback used to compute a projection into a defect-admissible set
   * \param lineSolver This is a callback used to minimize a directional restriction of the functional
   *        for computing a damping parameter
   */
  ParTNNMGStep(const Functional& f,
            Vector& x,
            const std::function<void(Vector&)>& collect,
            const std::function<void(Vector&)>& restrict,
            std::shared_ptr<Solvers::IterationStep<Vector,BitVector> > nonlinearSmoother,
            std::shared_ptr<Solvers::LinearIterationStep<ConstrainedMatrix,ConstrainedVector> > linearIterationStep,
            unsigned int noOfLinearIterationSteps,
            const DefectProjection& projection,
            const LineSearchSolver& lineSolver)
  : Base(x),
    f_(&f),
    collect_(collect),
    restrict_(restrict),
    nonlinearSmoother_(nonlinearSmoother),
    linearCorrection_(makeLinearCorrection(linearIterationStep, noOfLinearIterationSteps)),
    projection_(projection),
    lineSolver_(lineSolver)
  {}

  using Base::getIterate;

  void preprocess() override
  {
    nonlinearSmoother_->setIgnore(this->ignore());
    nonlinearSmoother_->preprocess();
  }

  void setPreSmoothingSteps(std::size_t i)
  {
    preSmoothingSteps_ = i;
  }

  /**
   * \brief Do one TNNMG step
   */
  void iterate() override
  {

    const auto& f = *f_;
    const auto& ignore = (*this->ignoreNodes_);
    auto& x = *getIterate();

    // Nonlinear presmoothing
    for (std::size_t i=0; i<preSmoothingSteps_; ++i) {
      nonlinearSmoother_->iterate();
      collect_(x); // TODO: This might actually be too much communication. Maybe, it's sufficient to do this once after iterating a few times.
    }


    // Compute constraint/truncated linearization
    if (not linearization_)
      linearization_ = std::make_shared<Linearization>(f, ignore);

    linearization_->bind(x);

    auto&& A = linearization_->hessian();
    // we copy the gradient, such that we're allowed to modify it
    auto r = linearization_->negativeGradient();

    collect_(r);

    // Compute inexact solution of the linearized problem
    Solvers::resizeInitializeZero(correction_, x);
    Solvers::resizeInitializeZero(constrainedCorrection_, r);


    linearCorrection_(A, constrainedCorrection_, r);
    linearization_->extendCorrection(constrainedCorrection_, correction_);

    // Project onto admissible set
    projection_(f, x, correction_);

    collect_(correction_);

    // Line search
    //auto fv = directionalRestriction(f, x, correction_);
    //dampingFactor_ = 0;
    //lineSolver_(dampingFactor_, fv, false);

    // TODO: This is an interface change. But I dont want to change
    // every single functional such that it supports communication.
    // Thus, we just apply a different line search
    //
    // (also, note that x is still in the proper state
    // since it was not changed since the last application of
    // the nonlinear smoother )
    dampingFactor_ = lineSolver_(x, correction_);

    if (std::isnan(dampingFactor_)) {
      printf("Found a NaN\n");
      dampingFactor_ = 0; // TODO This maybe should be broadcasted
    }

    correction_ *= dampingFactor_;
    // correction is still in valid state, no communication needed

    x += correction_;
    // same holds for x
  }

  /**
   * \brief Export the last computed damping factor
   */
  double lastDampingFactor() const
  {
    return dampingFactor_;
  }

  /**
   * \brief Export the last used linearization
   */
  const Linearization& linearization() const
  {
    return *linearization_;
  }

private:

  const Functional* f_;
  const std::function<void(Vector&)>& collect_;
  const std::function<void(Vector&)>& restrict_;

  std::shared_ptr<IterationStep<Vector,BitVector> > nonlinearSmoother_;
  std::size_t preSmoothingSteps_ = 1;

  std::shared_ptr<Linearization> linearization_;

  //! \brief linear correction
  LinearCorrection<ConstrainedMatrix, ConstrainedVector> linearCorrection_;

  typename Linearization::ConstrainedVector constrainedCorrection_;
  Vector correction_;
  DefectProjection projection_;
  LineSearchSolver lineSolver_;
  double dampingFactor_;
};


} // end namespace TNNMG
} // end namespace Dune

