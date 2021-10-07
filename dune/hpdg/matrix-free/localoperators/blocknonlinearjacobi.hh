#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_BLOCK_NONLINEAR_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_BLOCK_NONLINEAR_JACOBI_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/functions/backends/istlvectorbackend.hh>

#include <dune/tnnmg/iterationsteps/nonlineargsstep.hh>

#include <dune/istl/matrix.hh>

#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

template<class V,
         class GV,
         class Basis,
         class LocalSolver,
         class FunctionalFactory,
         class MatrixCreator>
class BlockNonlinearJacobi : public LocalOperator<V, GV>
{
  using Base = LocalOperator<V, GV>;
  using LV = typename Basis::LocalView;
  using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
  using Field = typename V::field_type;
  using LocalMatrix =
    Dune::Matrix<Dune::FieldMatrix<Field, 1, 1>>; // TODO: This is not
                                                  // necessarily 1x1

  static constexpr int dim = GV::dimension;

public:
  template<class LS, class FF>
  BlockNonlinearJacobi(const Basis& b,
                           LS&& localSolver,
                           FF&& functionalFactory,
                           MatrixCreator mc)
    : basis_(b)
    , localView_(basis_.localView())
    , localSolver_(std::forward<LS>(localSolver))
    , functionalFactory_(std::forward<FF>(functionalFactory))
    , matrixCreator_(std::move(mc))
  {}

  void setIterations(int i) {
    iter_ = i;
  }

  void bind(const typename Base::Entity& e)
  {
    localView_.bind(e);

    localVector_.resize(localView_.size());
    for (auto& e : localVector_)
      e = 0;

    matrixCreator_.bind(e);
  }

  // compute matrix diagonal block
  // and solve block for input part
  void compute()
  {
    auto idx = localView_.index(0)[0];

    localMatrix_ = matrixCreator_.matrix();

    assert(localMatrix_.N() == localVector_.size() &&
           localMatrix_.M() == localVector_.size());

    // now that the localMatrix is set up, we can apply the localsolver:
    // auto insideCoeffs = std::vector<Field>(fe.localBasis().size());
    // auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
    // for (size_t i = 0; i < localVector_.size(); i++) {
    //   // insideCoeffs[i] = inputBackend(localView_.index(i));
    //   localVector_[i] = inputBackend(localView_.index(i));
    // }
    auto ignore = std::vector<bool>(localVector_.size(), false);

    // auto f = functionalFactory_(matrixCreator_.matrix(), idx);
    auto f = functionalFactory_(localMatrix_, idx);

    auto gs = TNNMG::NonlinearGSStep<decltype(f), LocalSolver, decltype(ignore)>(f, localVector_, localSolver_);
    gs.setIgnore(ignore);

    for (int i = 0; i < iter_; ++i)
      gs.iterate();
    //   // TNNMG::gaussSeidelLoop(localVector_, f, ignore, localSolver_);

    // std::cout << "Solving local defect problem with nonlinear GS\n";
    // using Solver = Solvers::LoopSolver<decltype(localVector_), decltype(ignore)>;
    // auto norm = Solvers::EnergyNorm<decltype(localMatrix_), decltype(localVector_)>(localMatrix_);
    // auto s = Solver(gs, 100, 1e-10, norm, Solver::QUIET);
    // s.preprocess();
    // s.solve();

    // okay now that was fun, we do the thing with TNNMG oO
    // maybe not.
  }

  void write(double factor)
  {

    if (factor != 1.0)
      for (auto& entry : localVector_)
        entry *= factor;
    if (factor == 0.0)
      return;

    auto outputBackend = Functions::istlVectorBackend(*(this->output_));
    auto* rowEntry = &(outputBackend[localView_.index(0)]);
    for (size_t localRow = 0; localRow < localView_.size(); ++localRow) {
      rowEntry[localRow] += localVector_[localRow];
    }
  }

private:
  const Basis& basis_;
  LV localView_;
  BlockVector<FieldVector<typename V::field_type, 1>> localVector_; // contiguous memory buffer
  LocalSolver localSolver_;
  FunctionalFactory functionalFactory_;
  MatrixCreator matrixCreator_; // Creates the diagonal blocks
  LocalMatrix localMatrix_;
  int iter_ = 1;
};
}
}
}
#endif
