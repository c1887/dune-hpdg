#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_IPDG_BLOCK_PROJECTED_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_IPDG_BLOCK_PROJECTED_JACOBI_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/functions/backends/istlvectorbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/istl/matrix.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  template<class V, class GV, class Basis, class LocalSolver, class MatrixCreator>
  class IPDGBlockProjectedJacobi : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<Field, 1,1>>; // TODO: This is not necessarily 1x1

    static constexpr int dim = GV::dimension;

    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    public:

      template<class LS>
      IPDGBlockProjectedJacobi(const Basis& b, LS&& localSolver, MatrixCreator mc, double penalty=2.0) :
        basis_(b),
        penalty_(penalty),
        localView_(basis_.localView()),
        localSolver_(std::forward<LS>(localSolver)),
        matrixCreator_(std::move(mc)) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for(auto& e: localVector_)
          e=0;

        matrixCreator_.bind(e);
      }

      // compute matrix diagonal block
      // and solve block for input part
      void compute() {

        const auto& fe = localView_.tree().finiteElement();

        localMatrix_ = matrixCreator_.matrix();

        assert(localMatrix_.N() == localVector_.size() && localMatrix_.M() == localVector_.size());

        // now that the localMatrix is set up, we can apply the localsolver:
        auto insideCoeffs = std::vector<Field>(fe.localBasis().size());
        auto lowerC = std::vector<Field>(fe.localBasis().size());
        auto upperC = std::vector<Field>(fe.localBasis().size());
        auto inputBackend = Functions::istlVectorBackend(*(this->input_));
        auto lowerBE = Functions::istlVectorBackend(*lower_);
        auto upperBE = Functions::istlVectorBackend(*upper_);
        for (size_t i = 0; i < insideCoeffs.size(); i++) {
          insideCoeffs[i] = inputBackend[localView_.index(i)];
          lowerC[i] = lowerBE(localView_.index(i));
          upperC[i] = upperBE(localView_.index(i));
        }

        localSolver_(localMatrix_, insideCoeffs, localVector_, lowerC, upperC);
      }

      void write(double factor) {

        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;
        if (factor == 0.0)
          return;

        auto outputBackend = Functions::istlVectorBackend(*(this->output_));
        auto* rowEntry = &(outputBackend[localView_.index(0)]);
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          rowEntry[localRow] += localVector_[localRow];
        }
      }

      void setObstacles(const V* lower, const V* upper) {
        lower_=lower;
        upper_=upper;
      }


    private:

      const Basis& basis_;
      double penalty_;
      LV localView_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      LocalSolver localSolver_;
      MatrixCreator matrixCreator_; // Creates the diagonal blocks
      LocalMatrix localMatrix_;
      const V* lower_;
      const V* upper_;
  };
}
}
}
#endif
