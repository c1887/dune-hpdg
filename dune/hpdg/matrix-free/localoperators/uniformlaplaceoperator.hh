// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_UNIFORM_LAPLACE_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_UNIFORM_LAPLACE_OPERATOR_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** computes (∇φ_j, ∇u) for all j on a given element 
   *
   * In contrast to the more general LocalLaplaceOperator,
   * this implementation assumes and exploits that the geometry 
   * and the local finite element will be the same for every element.
   *
   */
  template<class V, class GV, class Basis>
  class UniformLaplaceOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;
    using JacobianType = typename FE::Traits::LocalBasisType::Traits::JacobianType;
    using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;

    static constexpr int dim = GV::dimension;

    public:

      UniformLaplaceOperator(const Basis& b) :
        basis_(b),
        localView_(basis_.localView()) {

          // precompute some things:

          // bind to the first element in gridview
          localView_.bind(*(basis_.gridView().template begin<0>()));
        
          localVector_.resize(localView_.maxSize());

          const auto& fe = localView_.tree().finiteElement();
          QuadratureRuleKey feQuad(fe);
          QuadratureRuleKey quadKey = feQuad.derivative().square();
          quad_ = QuadratureRuleCache<double, dim>::rule(quadKey);
        
          referenceGradients_.resize(quad_.size());
          gradients_.resize(quad_.size());

          feSize_ = fe.localBasis().size();

          const auto& geometry = localView_.element().geometry();

          for (size_t q = 0; q < quad_.size(); q++) {
            gradients_[q].resize(feSize_);
            referenceGradients_[q].resize(feSize_);

            const auto& quadPos = quad_[q].position();
            const auto& invJacobian = geometry.jacobianInverseTransposed(quadPos);

            fe.localBasis().evaluateJacobian(quadPos, referenceGradients_[q]);

            for (size_t i = 0; i < feSize_; i++) {
              invJacobian.mv(referenceGradients_[q][i][0], gradients_[q][i]);

              auto z = geometry.integrationElement(quadPos)*quad_[q].weight();
              gradients_[q][i]*=z;
            }
          }
        }

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        // resize was already done in constructor
        for (auto& entry: localVector_)
            entry=0;
      }

      void compute() {


        const auto& geometry = localView_.element().geometry();

        // we need the coefficients at every quadrature point. We extract and order them once:
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        auto coeffs = std::vector<Field>(feSize_);
        for (size_t i = 0; i < feSize_; i++) {
          coeffs[i] = inputBackend(localView_.index(i));
        }

        for (size_t q = 0; q < quad_.size(); q++) {

          const auto& invJacobian = geometry.jacobianInverseTransposed(quad_[q].position());

          FVdimworld duq(0);

          // multiply reference gradients with coefficient and sum them up
          for (size_t i = 0; i < feSize_; i++)
            duq.axpy(coeffs[i], referenceGradients_[q][i][0]);

          // apply inverse Jacobian once
          {
            auto tmp = duq;
            invJacobian.mv(tmp, duq);
          }

          for (size_t i = 0; i < feSize_; i++) {
            // sum  into localVector:
            localVector_[i] += gradients_[q][i]*duq;
          }
        }
      }

      void write(double factor) {

        if (factor == 0.0)
          return;
        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<feSize_; ++localRow)
        {
          auto& rowEntry = outputBackend(localView_.index(localRow));
          rowEntry += localVector_[localRow];
        }
      }

    private:
      const Basis& basis_;
      LV localView_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      QuadratureRule<double, dim> quad_;
      std::vector<std::vector<JacobianType>> referenceGradients_;
      std::vector<std::vector<FVdimworld>> gradients_;
      size_t feSize_;
  };
}
}
}
#endif
