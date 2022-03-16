// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_SLOW_IPDG_BLOCK_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_SLOW_IPDG_BLOCK_JACOBI_HH
#include <dune/common/fmatrix.hh>
#include <dune/common/math.hh>

#include <dune/istl/matrix.hh>

#include <dune/functions/backends/istlvectorbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/hpdg/assemblers/localassemblers/variableipdg.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/istl/matrix.hh>

#include <dune/hpdg/common/indexedcache.hh>
#include <dune/hpdg/common/mutablequadraturenodes.hh>
#include <dune/hpdg/localfunctions/assemblycache.hh>
#include <dune/hpdg/matrix-free/localoperators/gausslobattomatrices.hh>

/** This is a factory for the diagonal block of an IPDG Laplace discretization.
 * After binding to an element via bind(), you can get the corresponding
 * diagonal matrix block of the stiffness matrix via matrix().
 *
 * This is particulary useful for block Jacobi algorithms.
 */
namespace Dune {
namespace HPDG {

  template<class Basis>
  class SlowIPDGDiagonalBlock {
    using GV = typename Basis::GridView;
    static constexpr int dim = GV::dimension;
    using LV = typename Basis::LocalView;
    using Field = double;
    using FV = Dune::FieldVector<Field, dim>;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<Field, 1,1>>; // TODO: This is not necessarily 1x1
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;

    public:

      SlowIPDGDiagonalBlock(const Basis& b, double penalty=2.0, bool dirichlet=false) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        localView_(basis_.localView()),
        outerView_(basis_.localView())
      {}

      template<class Entity>
      void bind(const Entity& e)
      {
        localView_.bind(e);

        localMatrix_.setSize(localView_.size(), localView_.size());
        localMatrix_=0.;
      }

      /** Get the matrix for the bound element */
      const auto& matrix() {
        compute();
        return localMatrix_;
      }

    private:
      // compute matrix diagonal block
      // and solve block for input part
      void compute() {
        computeBulk();
        computeFace();
      }

      void computeFace() {
        const auto& fe = localView_.tree().finiteElement();
        insideCache_.bind(&fe);
        auto localAssembler = VInteriorPenaltyDGAssembler<typename GV::Grid, HPDG::AssemblyCache<FE>, HPDG::AssemblyCache<FE>>(penalty_, dirichlet_);

        // this is of course way too much work :(
        // TODO: We only need the inner x inner terms but we do compute all of the stuff
        // with the slow assembler. This is a big no.
        using MatrixContainer = Dune::Matrix<LocalMatrix>;
        auto mc = MatrixContainer(2,2);


        for (const auto& is: intersections(basis_.gridView(), localView_.element())) {
          if (not is.neighbor()) {
            // TODO Das funktioniert hier irgendwie nicht. Tun wir erstmal so, als sei das Neumann
            //continue; // TODO <-- Die Zeile dann löschen
            auto tmp =localMatrix_;

            localAssembler.assemble(is, tmp, insideCache_, insideCache_);
            //localMatrix_+=tmp;
            for(std::size_t i = 0; i < fe.size(); i++) {
              auto row = localView_.index(i)[1];
              for(std::size_t j = 0; j < fe.size(); j++) {
                auto col = localView_.index(j)[1];
                localMatrix_[row][col]+=tmp[i][j];
              }
            }
          }
          else {
            outerView_.bind(is.outside());
            const auto& ofe = outerView_.tree().finiteElement();


            auto maxOrder = std::max(ofe.localBasis().order(), fe.localBasis().order());
            auto penalty = penalty_ * maxOrder*maxOrder; // TODO: ^2 scaling is wrong for dim==1

            using FVdimworld = typename Dune::template FieldVector<double,dim>;

            using JacobianType = typename FE::Traits::LocalBasisType::Traits::JacobianType;
            using RangeType = typename FE::Traits::LocalBasisType::Traits::RangeType;

            // get geometry and store it
            const auto edgeGeometry = is.geometry();
            const auto insideGeometry = is.inside().geometry();

            const auto edgeLength = edgeGeometry.volume();

            // get quadrature rule
            QuadratureRuleKey tFEquad(is.type(), maxOrder);
            QuadratureRuleKey quadKey = tFEquad.square();

            const auto& quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

            // store gradients for both the inner and outer elements
            std::vector<JacobianType> insideReferenceGradients(fe.localBasis().size());
            std::vector<FVdimworld> insideGradients(fe.localBasis().size());

            // store values of shape functions
            std::vector<RangeType> tFEinsideValues(fe.localBasis().size());

            const auto outerNormal = is.centerUnitOuterNormal();

            // loop over quadrature points
            for (size_t pt=0; pt < quad.size(); ++pt)
            {
              // get quadrature point
              const auto& quadPos = quad[pt].position();

              // get transposed inverse of Jacobian of transformation
              const auto& invJacobian = insideGeometry.jacobianInverseTransposed(is.geometryInInside().global(quadPos));

              // get integration factor
              const auto integrationElement = edgeGeometry.integrationElement(quadPos);

              // get gradients of shape functions on both the inside and outside element
              insideCache_.localBasis().evaluateJacobian(is.geometryInInside().global(quadPos), insideReferenceGradients);

              // transform gradients
              for (size_t i=0; i<insideGradients.size(); ++i)
                invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);

              // evaluate basis functions
              insideCache_.localBasis().evaluateFunction(is.geometryInInside().global(quadPos), tFEinsideValues);

              // compute matrix entries
              auto z = quad[pt].weight() * integrationElement;

              // Basis functions from inside as test functions
              for (size_t i=0; i<fe.localBasis().size(); ++i)
              {
                // Basis functions from inside as ansatz functions
                for (size_t j=0; j<fe.localBasis().size(); ++j)
                {
                  // M11, see Riviere, p. 54f
                  auto zij = -0.5*z*tFEinsideValues[i]*(insideGradients[j]*outerNormal);
                  zij += -0.5*z*tFEinsideValues[j]*(insideGradients[i]*outerNormal);
                  zij += penalty*z/edgeLength*tFEinsideValues[i]*tFEinsideValues[j];
                  localMatrix_[i][j]+=zij;
                }
              }
            }

            /*
            for(std::size_t i = 0; i < fe.size(); i++) {
              auto row = localView_.index(i)[1];
              for(std::size_t j = 0; j < fe.size(); j++) {
                auto col = localView_.index(j)[1];
                localMatrix_[row][col]+=tmp[i][j];
              }
            }
            */
          }
        }
      }

      void computeBulk() {
        const auto& fe = localView_.tree().finiteElement();
        insideCache_.bind(&fe);
        //using FE = std::decay_t<decltype(fe)>;
        auto localAssembler = LaplaceAssembler<typename GV::Grid, HPDG::AssemblyCache<FE>, HPDG::AssemblyCache<FE>>();
        auto tmp = localMatrix_;
        localAssembler.assemble(localView_.element(), tmp, insideCache_, insideCache_);
        //localMatrix_+=tmp;

        for(std::size_t i = 0; i < fe.size(); i++) {
          auto row = localView_.index(i)[1];
          for(std::size_t j = 0; j < fe.size(); j++) {
            auto col = localView_.index(j)[1];
            localMatrix_[row][col]+=tmp[i][j];
          }
        }
      }



      // members:
      const Basis& basis_;
      double penalty_;
      bool dirichlet_;
      LV localView_;
      LV outerView_;
      LocalMatrix localMatrix_;
      HPDG::AssemblyCache<FE> insideCache_;
  };
}
}
#endif
