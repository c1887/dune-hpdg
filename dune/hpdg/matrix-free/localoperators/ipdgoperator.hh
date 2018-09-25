// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_IPDG_FACE_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_IPDG_FACE_OPERATOR_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** computes (∇φ_j, ∇u) for all j on a given element */
  template<class V, class GV, class Basis>
  class IPDGOperator : public LocalOperator<V, GV> {
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

      IPDGOperator(const Basis& b, double penalty=2.0, bool dirichlet=false) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        localView_(basis_.localView()),
        outerLocalView_(basis_.localView()),
        mapper_(basis_.gridView(), mcmgElementLayout()) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for(auto& e: localVector_)
          e=0;
      }

      void compute() {
        computeBulk();
        computeFace();
      }

      void write(double factor) {

        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;
        if (factor == 0.0)
          return;

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          auto& rowEntry = outputBackend(localView_.index(localRow));
          rowEntry += localVector_[localRow];
        }
      }

    private:

      void computeFace() {
        // lambda that applies linear operator to vector
        auto applyOp= [](const auto& op, auto& vec) {
          auto tmp = vec;
          op.mv(tmp, vec);
        };

        const auto& gv = basis_.gridView();

        const auto& insideFE = localView_.tree().finiteElement();
        auto insideCoeffs = std::vector<Field>(insideFE.localBasis().size());
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        for (size_t i = 0; i < insideCoeffs.size(); i++) {
              insideCoeffs[i] = inputBackend(localView_.index(i)); }

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

          auto outsideCoeffs = std::vector<Field>(outsideFE.localBasis().size());
          for (size_t i = 0; i < outsideCoeffs.size(); i++) {
            outsideCoeffs[i] = inputBackend(outerLocalView_.index(i));
          }

          auto maxOrder = std::max(insideFE.localBasis().order(), outsideFE.localBasis().order());

          auto penalty = penalty_ * std::pow(maxOrder, 2.0);

          auto quadKey = QuadratureRuleKey(is.type(), maxOrder).square();
          auto quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

          auto outerNormal = is.centerUnitOuterNormal();
          auto edgeLength = is.geometry().volume();

          std::vector<JacobianType> insideRefGradients(insideFE.localBasis().size());
          std::vector<JacobianType> outsideRefGradients(outsideFE.localBasis().size());
          std::vector<Range> insideValues(insideFE.localBasis().size());
          std::vector<Range> outsideValues(insideFE.localBasis().size());

          auto nPoints = quad.size();

          for (size_t q = 0; q < nPoints; q++) {
            FVdimworld insideDuq(0);
            FVdimworld outsideDuq(0);
            Range insideU(0);
            Range outsideU(0);

            const auto& quadPos = quad[q].position();

            insideFE.localBasis().evaluateFunction(insideGeometry.global(quadPos), insideValues);
            outsideFE.localBasis().evaluateFunction(outsideGeometry.global(quadPos), outsideValues);
            insideFE.localBasis().evaluateJacobian(insideGeometry.global(quadPos), insideRefGradients);
            outsideFE.localBasis().evaluateJacobian(outsideGeometry.global(quadPos), outsideRefGradients);


            // multiply reference values and gradients with coefficient and sum them up
            for (size_t i = 0; i < insideFE.localBasis().size(); i++) {
              insideDuq.axpy(insideCoeffs[i], insideRefGradients[i][0]);
              insideU.axpy(insideCoeffs[i], insideValues[i]);
            }
            for (size_t i = 0; i < outsideFE.localBasis().size(); i++) {
              outsideDuq.axpy(outsideCoeffs[i], outsideRefGradients[i][0]);
              outsideU.axpy(outsideCoeffs[i], outsideValues[i]);
            }

            auto insideInvJac = is.inside().geometry().jacobianInverseTransposed(insideGeometry.global(quadPos));
            auto outsideInvJac = is.outside().geometry().jacobianInverseTransposed(outsideGeometry.global(quadPos));

            auto weight = is.geometry().integrationElement(quadPos)*quad[q].weight();

            // apply jacobians and integration weights
            applyOp(insideInvJac, insideDuq);
            applyOp(outsideInvJac, outsideDuq);

            // now, we have to compute every part of the IPDG scheme:
            // 0. - {du/dn}[phi_i]
            {
              // compute average of normal derivative:
              auto grad = insideDuq;
              grad+=outsideDuq;
              auto avg = grad*outerNormal;
              avg*=0.5*weight;

              for (size_t i = 0; i < insideValues.size(); i++)
                localVector_[i] -= avg * insideValues[i];
              for (size_t i = 0; i < outsideValues.size(); i++)
                outerLocalVector_[i] += avg * outsideValues[i]; // the minus in front and the one from the jump cancel out, hence +=
            }
            // 1. - {dphi_i/dn}[u] // TODO: This is currently only for symmetric IPDG. One has to assign the right sign for this term otherwise
            {
              // compute jump term
              Range jump(0);
              for (const auto& val: insideU)
                jump += val;
              for (const auto& val: outsideU)
                jump -= val;
              jump*=weight;

              for (size_t i = 0; i < insideRefGradients.size(); i++) {
                auto avg = insideRefGradients[i][0];
                applyOp(insideInvJac, avg);
                avg*= 0.5;
                localVector_[i] -= jump*(avg*outerNormal);
              }
              for (size_t i = 0; i < outsideRefGradients.size(); i++) {
                auto avg = outsideRefGradients[i][0];
                applyOp(outsideInvJac, avg);
                avg*= 0.5;
                outerLocalVector_[i] -= jump*(avg*outerNormal);
              }
            }
            // 2. + sigma/|e| [phi_i][u]
            {
              // compute jump term
              Range jump(0);
              for (const auto& val: insideU)
                jump += val;
              for (const auto& val: outsideU)
                jump -= val;

              jump*= (weight*penalty/edgeLength);

              for (size_t i = 0; i < insideValues.size(); i++)
                localVector_[i] += jump * insideValues[i];
              for (size_t i = 0; i < outsideValues.size(); i++)
                outerLocalVector_[i] -= jump * outsideValues[i];
            }
          }
          // TODO: not thread-safe here
          if (this->factor_!=1.0)
            for (auto& entry: outerLocalVector_)
              entry*=this->factor_;

          auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
          for (size_t localRow=0; localRow<outerLocalView_.size(); ++localRow)
          {
            auto& rowEntry = outputBackend(outerLocalView_.index(localRow));
            rowEntry += outerLocalVector_[localRow];
          }
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
        auto coeffs = std::vector<Field>(fe.localBasis().size());
        for (size_t i = 0; i < coeffs.size(); i++) {
          coeffs[i] = inputBackend(localView_.index(i));
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

      // Unfortunately, this is very much the same code as for the non-boundary edges.
      // However, guarding everything in if/else clauses there is even uglier
      template<class IS, class C>
      void computeDirichletBoundaryEdge(const IS& is, const C& insideCoeffs) {
        auto applyOp= [](const auto& op, auto& vec) {
          auto tmp = vec;
          op.mv(tmp, vec);
        };

        const auto& insideFE = localView_.tree().finiteElement();
        auto penalty = penalty_ * std::pow(insideFE.localBasis().order(), 2.0);

        auto insideGeometry = is.geometryInInside();


        auto maxOrder = insideFE.localBasis().order();

        auto quadKey = QuadratureRuleKey(is.type(), maxOrder).square();
        auto quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

        auto outerNormal = is.centerUnitOuterNormal();
        auto edgeLength = is.geometry().volume();

        std::vector<JacobianType> insideRefGradients(insideFE.localBasis().size());
        std::vector<Range> insideValues(insideFE.localBasis().size());

        auto nPoints = quad.size();

        for (size_t q = 0; q < nPoints; q++) {
          FVdimworld insideDuq(0);
          Range insideU(0);

          const auto& quadPos = quad[q].position();

          insideFE.localBasis().evaluateFunction(insideGeometry.global(quadPos), insideValues);
          insideFE.localBasis().evaluateJacobian(insideGeometry.global(quadPos), insideRefGradients);


          // multiply reference values and gradients with coefficient and sum them up
          for (size_t i = 0; i < insideFE.localBasis().size(); i++) {
            insideDuq.axpy(insideCoeffs[i], insideRefGradients[i][0]);
            insideU.axpy(insideCoeffs[i], insideValues[i]);
          }

          auto insideInvJac = is.inside().geometry().jacobianInverseTransposed(insideGeometry.global(quadPos));

          auto weight = is.geometry().integrationElement(quadPos)*quad[q].weight();

          // apply jacobians and integration weights
          applyOp(insideInvJac, insideDuq);

          // now, we have to compute every part of the IPDG scheme:
          // 0. - {du/dn}[phi_i]
          {
            // compute average of normal derivative:
            auto grad = insideDuq;
            auto avg = grad*outerNormal;
            avg*=weight; // no 0.5 here as the average is defined differently for boundary edges

            for (size_t i = 0; i < insideValues.size(); i++)
              localVector_[i] -= avg * insideValues[i];
          }
          // 1. - {dphi_i/dn}[u] // TODO: This is currently only for symmetric IPDG. One has to assign the right sign for this term otherwise
          {
            // compute jump term
            Range jump(0);
            for (const auto& val: insideU)
              jump += val;
            jump*=weight;

            for (size_t i = 0; i < insideRefGradients.size(); i++) {
              auto avg = insideRefGradients[i][0];
              applyOp(insideInvJac, avg);
              localVector_[i] -= jump*(avg*outerNormal);
            }
          }
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
