#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_LAPLACE_DIAGONAL_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_LAPLACE_DIAGONAL_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** computes (∇φ_j, ∇u) for all j on a given element */
  template<class V, class GV, class Basis>
  class LaplaceDiagonal : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using LIS = typename Basis::LocalIndexSet;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;

    static constexpr int dim = GV::dimension;

    public:

      LaplaceDiagonal(const Basis& b) :
        basis_(b),
        localView_(basis_.localView()),
        localIndexSet_(basis_.localIndexSet()) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);
        localIndexSet_.bind(localView_);

        localVector_.resize(localView_.maxSize());
        for (auto& entry: localVector_)
          entry=0;
      }

      void compute() {

        using JacobianType = typename FE::Traits::LocalBasisType::Traits::JacobianType;
        using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;

        const auto& fe = localView_.tree().finiteElement();
        QuadratureRuleKey feQuad(fe);
        QuadratureRuleKey quadKey = feQuad.derivative().square();
        const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

        auto nPoints = quad.size();

        std::vector<JacobianType> referenceGradients(fe.localBasis().size());

        const auto& geometry = localView_.element().geometry();

        for (size_t q = 0; q < nPoints; q++) {

          const auto& quadPos = quad[q].position();
          const auto& invJacobian = geometry.jacobianInverseTransposed(quadPos);

          fe.localBasis().evaluateJacobian(quadPos, referenceGradients);

          auto z = geometry.integrationElement(quadPos)*quad[q].weight();

          for (size_t i = 0; i < referenceGradients.size(); i++) {
            FVdimworld gradient;
            invJacobian.mv(referenceGradients[i][0], gradient);
            // sum  into localVector:
            localVector_[i] += z*(gradient*gradient);
          }
        }
      }

      void write(double factor) {

        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;
        if (factor == 0.0)
          return;

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<localIndexSet_.size(); ++localRow)
        {
          auto& rowEntry = outputBackend(localIndexSet_.index(localRow));
          rowEntry += localVector_[localRow];
        }
      }

    private:
      const Basis& basis_;
      LV localView_;
      LIS localIndexSet_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
  };
}
}
}
#endif
