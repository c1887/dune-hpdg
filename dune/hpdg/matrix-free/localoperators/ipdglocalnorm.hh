// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_HPDG_MATRIX_FREE_LOCAL_IPDG_LOCAL_NORM_HH
#define DUNE_HPDG_MATRIX_FREE_LOCAL_IPDG_LOCAL_NORM_HH
#include <dune/common/fmatrix.hh>
#include <dune/common/math.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** 
   * This local operator expects an output vector of the same length
   * as the input vector, but the vector blocks should have size one.
   *
   * Then, each component will contain the value of the DG norm of the input
   * function on the corresponding element (the _local_ norm):
   *
   * IPDGLocalNorm(x)_i = (Dx, Dx)_E(i) + sum_(e in E(i)) sigma/(2|e|) \int [x]^2 dS
   *
   * (and similar for Dirichlet edges).
   *
   * TODO: Use cached local finite elements
   * */
  template<class V, class GV, class Basis>
  class IPDGLocalNorm : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;
    using JacobianType = typename FE::Traits::LocalBasisType::Traits::JacobianType;
    using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;
    using Range = typename FE::Traits::LocalBasisType::Traits::RangeType;


    static constexpr int dim = GV::dimension;

    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    public:

      IPDGLocalNorm(const Basis& b, double penalty=2.0, bool dirichlet=false) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        localView_(basis_.localView()),
        outerLocalView_(basis_.localView()),
        mapper_(basis_.gridView(), mcmgElementLayout()) {
        }

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for(auto& e: localVector_)
          e=0;
      }

      void compute() {
        computeBulk();
        writeBulk();
        computeFace();
      }

      void write(double) {
        /** Nothing */
      }

    private:

      /** Write the bulk parts and set localVector to zero afterwards */
      void writeBulk() {
        auto elementIdx = localView_.index(0)[0];
        auto& out = (*(this->output_))[elementIdx][0];
        for(std::size_t i = 0; i < localVector_.size(); i++) {
          out += localVector_[i] * (*(this->input_))[elementIdx][i];
        }

        for(auto& l: localVector_)
          l=0.;
      }

      void computeFace() {
        const auto& gv = basis_.gridView();

        const auto& insideFE = localView_.tree().finiteElement();
        // for dg, we know that the coeffs. will be continous in memory
        //auto insideCoeffs = std::vector<Field>(insideFE.localBasis().size());
        const Field* insideCoeffs;
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        insideCoeffs = &(inputBackend(localView_.index(0))); // assuming 0 maps to the lowest index

        for (const auto& is: intersections(gv, localView_.element())) {
          if (!is.neighbor()) // this is also for processor boundaries valid
          {
            if(dirichlet_ && is.boundary()) {
              computeDirichletBoundaryEdge(is, insideCoeffs);
              continue;
            }
            else
              continue;
          }

          // Skip handled edges
          if(mapper_.index(is.inside()) < mapper_.index(is.outside()))
            continue;

          // bind outer view
          outerLocalView_.bind(is.outside());

          outerLocalVector_.resize(outerLocalView_.size());
          for(auto& e: outerLocalVector_)
            e=0;

          auto insideGeometry = is.geometryInInside();
          auto outsideGeometry = is.geometryInOutside();

          const auto& outsideFE = outerLocalView_.tree().finiteElement();

          //auto outsideCoeffs = std::vector<Field>(outsideFE.localBasis().size());
          const Field* outsideCoeffs = &(inputBackend(outerLocalView_.index(0)));

          auto maxOrder = std::max(insideFE.localBasis().order(), outsideFE.localBasis().order());

          auto penalty = penalty_ * power(maxOrder, 2);

          auto quadKey = QuadratureRuleKey(is.type(), maxOrder).square();
          auto quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

          auto edgeLength = is.geometry().volume();

          std::vector<Range> insideValues(insideFE.localBasis().size());
          std::vector<Range> outsideValues(insideFE.localBasis().size());

          auto nPoints = quad.size();

          for (size_t q = 0; q < nPoints; q++) {
            Range insideU(0);
            Range outsideU(0);

            const auto& quadPos = quad[q].position();

            insideFE.localBasis().evaluateFunction(insideGeometry.global(quadPos), insideValues);
            outsideFE.localBasis().evaluateFunction(outsideGeometry.global(quadPos), outsideValues);


            for (size_t i = 0; i < insideValues.size(); i++) {
              insideU.axpy(insideCoeffs[i], insideValues[i]);
            }
            for (size_t i = 0; i < outsideValues.size(); i++) {
              outsideU.axpy(outsideCoeffs[i], outsideValues[i]);
            }

            auto weight = is.geometry().integrationElement(quadPos)*quad[q].weight();

            // The {dphi}[u] and {du}[phi] terms are dropped for the norm

            // 2. + sigma/|e| [phi_i][u]
            {
              // compute jump term
              Range jump(0);
              for (const auto& val: insideU)
                jump += val;
              for (const auto& val: outsideU)
                jump -= val;

              jump*= (weight*penalty/(2.*edgeLength)); // The 2 is because we're visiting from both sides

              for (size_t i = 0; i < insideValues.size(); i++)
                localVector_[i] += jump * insideValues[i];
              for (size_t i = 0; i < outsideValues.size(); i++)
                outerLocalVector_[i] -= jump * outsideValues[i];
            }
          }

          // hier müssen dann die terme direkt geschrieben werden, weil wir noch andere Kanten besuchen werden
          auto sum=0.;
          for(std::size_t i = 0; i < localVector_.size(); i++) {
            sum+=localVector_[i]*insideCoeffs[i];
            localVector_[i]=0;
          }
          for(std::size_t i = 0; i < outerLocalVector_.size(); i++) {
            sum+=outerLocalVector_[i]*outsideCoeffs[i];
          }

          auto& out = *(this->output_);
          out[localView_.index(0)[0]][0] += sum;
          out[outerLocalView_.index(0)[0]][0] += sum;

        }
      }

      void computeBulk() {

        const auto& fe = localView_.tree().finiteElement();
        QuadratureRuleKey feQuad(fe);
        QuadratureRuleKey quadKey = feQuad.derivative().square();
        const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

        auto nPoints = quad.size();

        std::vector<JacobianType> referenceGradients(fe.localBasis().size());

        const auto& geometry = localView_.element().geometry();

        // we need the coefficients at every quadrature point. We extract and order them once:
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        const Field* coeffs = &(inputBackend(localView_.index(0)));
        //auto coeffs = std::vector<Field>(fe.localBasis().size());
        //for (size_t i = 0; i < coeffs.size(); i++) {
          //coeffs[i] = inputBackend(localView_.index(i));
        //}

        for (size_t q = 0; q < nPoints; q++) {

          FVdimworld duq(0);

          const auto& quadPos = quad[q].position();
          const auto& invJacobian = geometry.jacobianInverseTransposed(quadPos);

          fe.localBasis().evaluateJacobian(quadPos, referenceGradients);

          // multiply reference gradients with coefficient and sum them up
          for (size_t i = 0; i < referenceGradients.size(); i++)
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

      // Unfortunately, this is very much the same code as for the non-boundary edges.
      // However, guarding everything in if/else clauses there is even uglier
      template<class IS, class C>
      void computeDirichletBoundaryEdge(const IS& is, const C& insideCoeffs) {
        const auto& insideFE = localView_.tree().finiteElement();
        auto penalty = penalty_ * std::pow(insideFE.localBasis().order(), 2.0);

        auto insideGeometry = is.geometryInInside();


        auto maxOrder = insideFE.localBasis().order();

        auto quadKey = QuadratureRuleKey(is.type(), maxOrder).square();
        auto quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

        auto edgeLength = is.geometry().volume();

        std::vector<Range> insideValues(insideFE.localBasis().size());

        auto nPoints = quad.size();

        for (size_t q = 0; q < nPoints; q++) {
          Range insideU(0);

          const auto& quadPos = quad[q].position();

          insideFE.localBasis().evaluateFunction(insideGeometry.global(quadPos), insideValues);


          // multiply reference values and gradients with coefficient and sum them up
          for (size_t i = 0; i < insideValues.size(); i++) {
            insideU.axpy(insideCoeffs[i], insideValues[i]);
          }

          auto weight = is.geometry().integrationElement(quadPos)*quad[q].weight();

          // now, we have to compute every part of the IPDG scheme:
          // 2. + sigma/|e| [phi_i][u]
          {
            // compute jump term
            Range jump(0);
            for (const auto& val: insideU)
              jump += val;

            jump*= (weight*penalty/edgeLength);

            for (size_t i = 0; i < insideValues.size(); i++)
              localVector_[i] += jump * insideValues[i];
          }
        }

        auto& out = (*(this->output_))[localView_.index(0)[0]][0];
        for(std::size_t i = 0; i < localVector_.size(); i++) {
          out += localVector_[i]*insideCoeffs[i];
          localVector_[i]=0;
        }
      }

      const Basis& basis_;
      double penalty_;
      bool dirichlet_;
      LV localView_;
      LV outerLocalView_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      std::vector<typename V::field_type> outerLocalVector_; // contiguous memory buffer
      Dune::MultipleCodimMultipleGeomTypeMapper<typename Basis::GridView> mapper_;
  };
}
}
}
#endif
