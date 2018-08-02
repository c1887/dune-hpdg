#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_LAPLACE_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_LAPLACE_OPERATOR_HH
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
  class LaplaceOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using LIS = typename Basis::LocalIndexSet;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;

    static constexpr int dim = GV::dimension;

    public:

      LaplaceOperator(const Basis& b) :
        basis_(b),
        localView_(basis_.localView()),
        localIndexSet_(basis_.localIndexSet()) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);
        localIndexSet_.bind(localView_);

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        localVector_= &(outputBackend(localIndexSet_.index(0)));
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

        // we need the coefficients at every quadrature point. We extract and order them once:
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        auto coeffs = std::vector<Field>(fe.localBasis().size());
        for (size_t i = 0; i < coeffs.size(); i++) {
          coeffs[i] = inputBackend(localIndexSet_.index(i));
        }

        for (size_t q = 0; q < nPoints; q++) {

          FVdimworld duq(0);

          const auto& quadPos = quad[q].position();
          const auto& invJacobian = geometry.jacobianInverseTransposed(quadPos);

          fe.localBasis().evaluateJacobian(quadPos, referenceGradients);

          // multiply reference gradients with coefficient and sum them up
          for (size_t i = 0; i < fe.localBasis().size(); i++)
            duq.axpy(coeffs[i], referenceGradients[i][0]);

          // apply inverse Jacobian once, multiply with quadrature weight and integration element
          {
            auto tmp = duq;
            invJacobian.mv(tmp, duq);

            auto z = geometry.integrationElement(quadPos)*quad[q].weight();
            duq*=z;
          }

          for (size_t i = 0; i < referenceGradients.size(); i++) {
            FVdimworld gradient;
            invJacobian.mv(referenceGradients[i][0], gradient);
            // sum  into localVector:
            localVector_[i] += gradient*duq;
          }
        }
      }

      void write(double) {

        //if (factor!=1.0)
          //for (auto& entry: localVector_)
            //entry*=factor;
        //if (factor == 0.0)
          //return;

        //auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        //for (size_t localRow=0; localRow<localIndexSet_.size(); ++localRow)
        //{
          //auto& rowEntry = outputBackend(localIndexSet_.index(localRow));
          //rowEntry += localVector_[localRow];
        //}
      }

    private:
      const Basis& basis_;
      LV localView_;
      LIS localIndexSet_;
      typename V::field_type* localVector_; // in DG, the memory should be already contiguous. Don't blame me if you shoot yourself in the foot!
  };
}
}
}
#endif
