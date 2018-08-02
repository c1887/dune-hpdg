#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_IPDG_BLOCK_PROJECTED_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_IPDG_BLOCK_PROJECTED_JACOBI_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/istl/matrix.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  template<class V, class GV, class Basis, class LocalSolver>
  class IPDGBlockProjectedJacobi : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;
    //using LocalMatrix = Dune::Matrix<FieldBlock>;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<Field, 1,1>>; // TODO: This is not necessarily 1x1

    static constexpr int dim = GV::dimension;

    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    public:

      template<class LS>
      IPDGBlockProjectedJacobi(const Basis& b, LS&& localSolver, double penalty=2.0, bool dirichlet=false) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        localView_(basis_.localView()),
        localSolver_(std::forward<LS>(localSolver)) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for(auto& e: localVector_)
          e=0;
        localMatrix_.setSize(localVector_.size(), localVector_.size());
        localMatrix_ = 0.0;
      }

      // compute matrix diagonal block
      // and solve block for input part
      void compute() {

        const auto& fe = localView_.tree().finiteElement();

        // bulk term: standard laplace:
        {
          auto laplace_localAssembler = LaplaceAssembler<typename Basis::GridView::Grid, FE, FE>();
          laplace_localAssembler.assemble(localView_.element(), localMatrix_, fe, fe);
        }
        // now the jump terms involving only the inner basis functions:
        for (const auto& edge: intersections(basis_.gridView(), localView_.element())) {
          double avg_factor = 0.5;
          //double avg_factor = 1.0;
          if (edge.boundary()) {
            if(!dirichlet_)
              continue;
            else
              avg_factor = 1.0; // We "average" only along the inner function as there is no outer function for a boundary edge
          }

          // Even though we do not use the outer edge, we still have to find modify the
          // penalty parameter accordingly to the higher degree to reassemble the same matrix
          // blocks as for the actual assembled matrix:
          auto order = fe.localBasis().order();
          if (edge.neighbor()) {
            auto lv = localView_;
            lv.bind(edge.outside());
            order = std::max(order, lv.tree().finiteElement().localBasis().order());
          }
          auto penalty = penalty_ * std::pow(order, 2.0);

          using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;

          using JacobianType = typename FE::Traits::LocalBasisType::Traits::JacobianType;
          using RangeType = typename FE::Traits::LocalBasisType::Traits::RangeType;

          // get geometry and store it
          const auto edgeGeometry = edge.geometry();
          const auto insideGeometry = edge.inside().geometry();

          const auto edgeLength = edgeGeometry.volume();

          // get quadrature rule
          QuadratureRuleKey tFEquad(edge.type(), order);
          QuadratureRuleKey quadKey = tFEquad.derivative().square();

          const auto& quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

          // store gradients
          std::vector<JacobianType> insideReferenceGradients(fe.localBasis().size());
          std::vector<FVdimworld> insideGradients(fe.localBasis().size());

          // store values of shape functions
          std::vector<RangeType> tFEinsideValues(fe.localBasis().size());

          const auto outerNormal = edge.centerUnitOuterNormal();

          // loop over quadrature points
          for (size_t pt=0; pt < quad.size(); ++pt)
          {
            // get quadrature point
            const auto& quadPos = quad[pt].position();

            // get transposed inverse of Jacobian of transformation
            const auto& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));

            // get integration factor
            const auto integrationElement = edgeGeometry.integrationElement(quadPos);

            // get gradients of shape functions on both the inside and outside element
            fe.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), insideReferenceGradients);

            // transform gradients
            for (size_t i=0; i<insideGradients.size(); ++i)
              invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);

            // evaluate basis functions
            fe.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEinsideValues);

            // compute matrix entries
            auto z = quad[pt].weight() * integrationElement;

            // Basis functions from inside as test functions
            for (size_t i=0; i<fe.localBasis().size(); ++i)
            {
              // Basis functions from inside as ansatz functions
              for (size_t j=0; j<fe.localBasis().size(); ++j)
              {
                auto zij = -avg_factor*z*tFEinsideValues[i]*(insideGradients[j]*outerNormal)
                  - avg_factor*z*tFEinsideValues[j]*(insideGradients[i]*outerNormal) // TODO: Just SIPG by now
                  + penalty*z/edgeLength*tFEinsideValues[i]*tFEinsideValues[j];
                  localMatrix_[i][j]+=zij;
              }
            }
          }
        }

        // now that the localMatrix is set up, we can apply the localsolver:
        auto insideCoeffs = std::vector<Field>(fe.localBasis().size());
        auto lowerC = std::vector<Field>(fe.localBasis().size());
        auto upperC = std::vector<Field>(fe.localBasis().size());
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        auto lowerBE = Fufem::istlVectorBackend<const Field>(lower_);
        auto upperBE = Fufem::istlVectorBackend<const Field>(upper_);
        for (size_t i = 0; i < insideCoeffs.size(); i++) {
          insideCoeffs[i] = inputBackend(localView_.index(i));
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

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          auto& rowEntry = outputBackend(localView_.index(localRow));
          rowEntry += localVector_[localRow];
        }
      }

      void setObstacles(const V& lower, const V& upper) {
        lower_=lower;
        upper_=upper;
      }


    private:

      const Basis& basis_;
      double penalty_;
      bool dirichlet_;
      LV localView_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      LocalSolver localSolver_;
      LocalMatrix localMatrix_;
      V lower_;
      V upper_;
  };
}
}
}
#endif
