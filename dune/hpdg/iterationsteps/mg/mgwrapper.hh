#ifndef DUNE_HPDG_ITERATIONSTEPS_MG_WRAPPER_HH
#define DUNE_HPDG_ITERATIONSTEPS_MG_WRAPPER_HH

#include <dune/solvers/iterationsteps/lineariterationstep.hh>

#include <dune/hpdg/iterationsteps/mg/multigrid.hh>

#include <functional>

namespace Dune {
namespace HPDG {
  template<class M, class V, class BV>
  class MultigridWrapper : public Dune::Solvers::LinearIterationStep<M, V, BV> {

    public:
    Multigrid<V>& mg_;
    std::function<void(std::shared_ptr<const M>)> preprocessFunction_;

    MultigridWrapper(Multigrid<V>& mg) :
      mg_(mg) {}

    virtual void iterate() {
      auto b = *(this->rhs_); // the MG impl. modifies the rhs, hence we make a copy here
      auto& x = *(this->x_);

      mg_.apply(x,b);
    }

    virtual void preprocess() {
      preprocessFunction_(this->mat_);
    }
  };
}
}
#endif//DUNE_HPDG_ITERATIONSTEPS_MG_WRAPPER_HH
