// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_HEAT_BLOCK_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_HEAT_BLOCK_JACOBI_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/istl/matrix.hh>

#include <dune/hpdg/common/mappedcache.hh>

/** This is a factory for the diagonal block of an IPDG Laplace discretization.
 * After binding to an element via bind(), you can get the corresponding
 * diagonal matrix block of the stiffness matrix via matrix().
 *
 * This is particulary useful for block Jacobi algorithms.
 */
namespace Dune {
namespace HPDG {

  template<class Basis>
  class HeatDiagonalBlock {
    using GV = typename Basis::GridView;
    static constexpr int dim = GV::dimension;
    using LV = typename Basis::LocalView;
    using Field = double;
    using FV = Dune::FieldVector<Field, dim>;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<Field, 1,1>>; // TODO: This is not necessarily 1x1

    static_assert(dim==2, "Sumfactorized IPDG only for dim=2 currently");

    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    public:

      HeatDiagonalBlock(const Basis& b, double penalty=2.0, double massFactor=1., double laplaceFactor=1.) :
        basis_(b),
        penalty_(penalty),
        localView_(basis_.localView()),
        cache_(std::bind(&HeatDiagonalBlock::matrixGenerator, this, std::placeholders::_1)),
        rules_(std::bind(&HeatDiagonalBlock::ruleGenerator, this, std::placeholders::_1))
        {
          factors_.mass = massFactor;
          factors_.laplace = laplaceFactor;
        }

      template<class Entity>
      void bind(const Entity& e)
      {
        localView_.bind(e);

        localMatrix_.setSize(localView_.size(), localView_.size());
        localMatrix_ = 0.0;

        // get order:
        localDegree_ = basis_.preBasis().degree(e);

        matrixPair_ = &cache_.value(localDegree_);
        rule_ = &rules_.value(localDegree_);
      }

      /** Get the matrix for the bound element */
      const auto& matrix() {
        compute();
        return localMatrix_;
      }

      void setDirichlet(bool dirichletBoundary) {
        dirichlet_ = dirichletBoundary;
      }

    private:
      // compute matrix diagonal block
      // and solve block for input part
      void compute() {

        // bulk term: standard laplace:
        {
          computeBulk();
          computeFace();
        }

      }

      void computeFace() {
        const auto& fe = localView_.tree().finiteElement();
        // now the jump terms involving only the inner basis functions:
        for (const auto& edge: intersections(basis_.gridView(), localView_.element())) {
          double avg_factor = 0.5;
          if (edge.boundary()) {
            if(!dirichlet_)
              continue;
            else
              avg_factor = 1.0; // We "average" only along the inner function as there is no outer function for a boundary edge
          }

          auto order = localDegree_;

          // if the edge is nonconforming, we can not use the precalculated values and need to re-evaluate
          decltype(nonConformingMatrices(edge)) nc_matrices;
          auto* edgeMatrices = matrixPair_;

          // set the rule if needed:
          if (edge.neighbor()) {

            // set order
            auto outerDegree =basis_.preBasis().degree(edge.outside());
            order = std::max(localDegree_, outerDegree);

            if (outerDegree > localDegree_) {
              rule_ = &(rules_[outerDegree]);
            }

            if (!edge.conforming() or outerDegree > localDegree_) {
              nc_matrices = nonConformingMatrices(edge);
              edgeMatrices = nc_matrices.get();
            }
          }



          // Even though we do not use the outer edge, we still have to find modify the
          // penalty parameter accordingly to the higher degree to reassemble the same matrix
          // blocks as for the actual assembled matrix:
          auto penalty = penalty_ * std::pow(order, 2.0);

          using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;

          // get geometry and store it
          const auto edgeGeometry = edge.geometry();
          const auto insideGeometry = edge.inside().geometry();

          const auto edgeLength = edgeGeometry.volume();

          // store gradients
          std::vector<FVdimworld> insideGradients(fe.localBasis().size());
          std::vector<Dune::FieldVector<double, 1>> insideValues(fe.localBasis().size());

          const auto outerNormal = edge.centerUnitOuterNormal();

          // loop over quadrature points
          for (size_t pt=0; pt < rule_->size(); ++pt)
          {
            // get quadrature point
            const auto& quadPos = (*rule_)[pt].position();

            // get transposed inverse of Jacobian of transformation
            const auto& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));

            // get integration factor
            const auto integrationElement = edgeGeometry.integrationElement(quadPos);

            // get gradients of shape functions on both the inside and outside element
            auto insideReferenceGradients = gradientsOnEdge(pt, *edgeMatrices, edge.indexInInside());

            // transform gradients
            for (size_t i=0; i<insideGradients.size(); ++i)
              invJacobian.mv(insideReferenceGradients[i], insideGradients[i]);

            // evaluate basis functions
            auto insideValues = valuesOnEdge(pt, *edgeMatrices, edge.indexInInside());

            // compute matrix entries
            auto z = (*rule_)[pt].weight() * integrationElement;

            // Basis functions from inside as test functions
            for (size_t i=0; i<localView_.size(); ++i)
            {
              // Basis functions from inside as ansatz functions
              for (size_t j=0; j<localView_.size(); ++j)
              {
                auto zij = -avg_factor*z*insideValues[i]*(insideGradients[j]*outerNormal)
                  - avg_factor*z*insideValues[j]*(insideGradients[i]*outerNormal) // TODO: Just SIPG by now
                  + penalty*z/edgeLength*insideValues[i]*insideValues[j];
                  localMatrix_[i][j]+=factors_.laplace*zij;
              }
            }
          }
          rule_ = &(rules_[localDegree_]);
        }
      }

      void computeBulk() {
        auto geometry = localView_.element().geometry();

        auto size = (localDegree_+1)*(localDegree_+1);

        // store gradients of shape functions and base functions
        using Gradient = FieldVector<double, dim>;
        std::vector<Gradient> gradients(size);

        // loop over quadrature points
        for (size_t q0=0; q0 < rule_->size(); ++q0) {
          FV quadPos = {{(*rule_)[q0].position(), 0}};
          for (size_t q1=0; q1 < rule_->size(); ++q1)
          {
            // get quadrature point
            quadPos[1]=(*rule_)[q1].position();

            // get transposed inverse of Jacobian of transformation
            const auto& invJacobian = geometry.jacobianInverseTransposed(quadPos);

            // get integration factor
            const auto integrationElement = geometry.integrationElement(quadPos);

            // get gradients of shape functions
            auto referenceGradients = computeGradients(q0, q1);

            // auto values:
            auto vals = computeElementValues(q0, q1);

            // transform gradients
            for (size_t i=0; i<gradients.size(); ++i)
              invJacobian.mv(referenceGradients[i], gradients[i]);

            // compute matrix entries
            auto z = (*rule_)[q0].weight() * (*rule_)[q1].weight() * integrationElement;
            auto z_laplace = factors_.laplace * z;
            auto z_mass = factors_.mass *z;
            for (size_t i=0; i<localMatrix_.N(); ++i)
            {
              for (size_t j=i+1; j<localMatrix_.M(); ++j)
              {
                double zij_l = (gradients[i] * gradients[j]) * z_laplace;
                double zij_m = (vals[i] * vals[j]) * z_mass;
                localMatrix_[i][j]+=zij_l + zij_m;
                localMatrix_[j][i]+=zij_l + zij_m;
              }
              localMatrix_[i][i]+= (gradients[i] * gradients[i]) * z_laplace + vals[i]*vals[i]*z_mass;
            }
          }
        }
      }

      auto computeGradients(size_t q0, size_t q1) const {
        using Gradient = FieldVector<double, dim>;
        std::vector<Gradient> referenceGradients(localView_.size());

        for(size_t i = 0; i < localDegree_+1; i++) {
          for(size_t j = 0; j < localDegree_+1; j++) {
            referenceGradients[flatIndex(i,j, localDegree_)] = {{(*matrixPair_)[1][i][q0]*(*matrixPair_)[0][j][q1], (*matrixPair_)[0][i][q0]*(*matrixPair_)[1][j][q1]}};
          }
        }

        return referenceGradients;
      }

      auto computeElementValues(size_t q0, size_t q1) const {
        using Value = FieldVector<double, 1>;
        std::vector<Value> vals(localView_.size());

        for(size_t i = 0; i < localDegree_+1; i++) {
          for(size_t j = 0; j < localDegree_+1; j++) {
            vals[flatIndex(i,j, localDegree_)] = (*matrixPair_)[0][i][q0]*(*matrixPair_)[0][j][q1];
          }
        }

        return vals;
      }

      template<class MP>
      auto valuesOnEdge(size_t quad_nr, const MP& edgeMatrices, int  edgeNumber) const {
        using Value = FieldVector<double, 1>;
        std::vector<Value> vals(localView_.size());

        auto lastIdx = (*matrixPair_)[0].M()-1;
        for(size_t i = 0; i < localDegree_+1; i++) {
          for(size_t j = 0; j < localDegree_+1; j++) {
            switch(edgeNumber) {
              case 0:
                vals[flatIndex(i,j, localDegree_)] = {{(*matrixPair_)[0][i][0]*edgeMatrices[0][j][quad_nr]}};
                break;
              case 1:
                vals[flatIndex(i,j, localDegree_)] = {{(*matrixPair_)[0][i][lastIdx]*edgeMatrices[0][j][quad_nr]}};
                break;
              case 2:
                vals[flatIndex(i,j, localDegree_)] = {{edgeMatrices[0][i][quad_nr]*(*matrixPair_)[0][j][0]}};
                break;
              case 3:
                vals[flatIndex(i,j, localDegree_)] = {{edgeMatrices[0][i][quad_nr]*(*matrixPair_)[0][j][lastIdx]}};
                break;
              default:
                DUNE_THROW(Dune::Exception, "Illegal edge number provided");
            }
          }
        }

        return vals;
      }

      template<class MP>
      auto gradientsOnEdge(size_t quad_nr, const MP& edgeMatrices, int edgeNumber) const {
        using Gradient = FieldVector<double, dim>;
        std::vector<Gradient> referenceGradients(localView_.size());

        auto lastIdx = (*matrixPair_)[0].M()-1;
        for (size_t i = 0; i< localDegree_ +1; i++) {
          for(size_t j = 0; j < localDegree_+1; j++) {
            switch(edgeNumber) {
              case 0:
                referenceGradients[flatIndex(i,j, localDegree_)] = {{(*matrixPair_)[1][i][0]*edgeMatrices[0][j][quad_nr],  (*matrixPair_)[0][i][0]*edgeMatrices[1][j][quad_nr]}};
                break;
              case 1:
                referenceGradients[flatIndex(i,j, localDegree_)] = {{(*matrixPair_)[1][i][lastIdx]*edgeMatrices[0][j][quad_nr], (*matrixPair_)[0][i][lastIdx]*edgeMatrices[1][j][quad_nr]}};
                break;
              case 2:
                referenceGradients[flatIndex(i,j, localDegree_)] = {{edgeMatrices[1][i][quad_nr]*(*matrixPair_)[0][j][0], edgeMatrices[0][i][quad_nr]*(*matrixPair_)[1][j][0]}};
                break;
              case 3:
                referenceGradients[flatIndex(i,j, localDegree_)] = {{edgeMatrices[1][i][quad_nr]*(*matrixPair_)[0][j][lastIdx], edgeMatrices[0][i][quad_nr]*(*matrixPair_)[1][j][lastIdx]}};
                break;
              default:
                DUNE_THROW(Dune::Exception, "Illegal edge number provided");
            }
          }
        }
        return referenceGradients;
      }

      template<class IS>
      auto nonConformingMatrices(const IS& is)
      {
        // basis nodes:
        auto gl_i = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, 2*localDegree_-1, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        assert(gl_i.size() == localDegree_+1);

        const auto& rule = *rule_;

        // sort node points (they're also ordered for the basis)
        std::sort(gl_i.begin(), gl_i.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        // compute inner and outer matrix pair for the noncorming edge.
        auto inner_quad = std::vector<typename GV::Grid::ctype>(rule.size());
        for (size_t i = 0; i < rule.size(); i++) {
          inner_quad[i] = is.geometryInInside().global(rule[i].position())[is.indexInInside() < 2]; // we only take the coordinate we need, which will be the first for edge number 2 and 3, and the second for 0 and 1
        }
        auto ret = std::make_unique<std::array<LocalMatrix, 2>>();

        auto& innerPair = (*ret);

        // resize:
        innerPair[0].setSize(gl_i.size(), inner_quad.size());
        innerPair[1].setSize(gl_i.size(), inner_quad.size());

        // fill cache
        for (std::size_t j = 0; j < rule.size(); j++) {
          for (std::size_t i = 0; i < gl_i.size(); i++) {
            innerPair[1][i][j]=lagrangePrime(inner_quad[j],i, gl_i);
            innerPair[0][i][j]=lagrange(inner_quad[j],i, gl_i);
          }
        }
        return ret;
      }

      auto flatIndex(unsigned int i0, unsigned int i1, int k) const {
        return i0 + i1*(k+1);
      }

      template<class X, class Q>
      inline double lagrangePrime(const X& x, size_t i, const Q& quad) const {
        double result = 0.;

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            double prod= 1.0/(quad[i].position()-quad[j].position());
            for (size_t l=0; l<quad.size(); l++)
              if (l!=i && l!=j)
                prod *= (x-quad[l].position())/(quad[i].position()-quad[l].position());
            result += prod;
          }
        return result;
      }

      template<class X, class Q>
      inline double lagrange(const X& x, size_t i, const Q& quad) const {
        double result = 1.;

        auto xi= quad[i].position();

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            result*= (x-quad[j].position())/(xi-quad[j].position());
          }
        return result;
      }

      auto matrixGenerator(int degree) {
        int order = 2*degree - 1;
        // get quadrature rule:
        auto gauss_lobatto = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes

        const auto& rule = rules_[degree];

        // sort quad points (they're also ordered for the basis)
        std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        // save all derivatives of 1-d basis functions at the quad points, i.e.
        // matrixPair_ij = l_i' (xi_j)
        LocalMatrix lp(localDegree_+1, rule.size());
        // same for the function values
        LocalMatrix l(localDegree_+1, rule.size());
        for (std::size_t i = 0; i < lp.N(); i++) {
          for (std::size_t j = 0; j < lp.M(); j++) {
            lp[i][j]=lagrangePrime(rule[j].position(),i, gauss_lobatto);
            l[i][j]=lagrange(rule[j].position(),i, gauss_lobatto);
          }
        }
        return std::array<LocalMatrix, 2>{{std::move(l), std::move(lp)}};
      }

      auto ruleGenerator(int degree) {
        int order = 2*degree -1;
        auto rule = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule
          (Dune::GeometryType::cube, order+1, Dune::QuadratureType::GaussLobatto); // TODO Welche Ordnung braucht man wirklich?

        std::sort(rule.begin(), rule.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        return rule;
      }


      // members:
      const Basis& basis_;
      double penalty_;
      bool dirichlet_ = false;
      LV localView_;
      struct {
        double mass;
        double laplace;
      } factors_;
      MappedCache<std::array<LocalMatrix, 2>, int> cache_; // contains all lagrange Polynomials at all quadrature points and all derivatives of said polynomials at all quad points
      MappedCache<Dune::QuadratureRule<typename GV::Grid::ctype, 1>, int> rules_;
      const typename decltype(cache_)::mapped_type* matrixPair_; // Current matrix pair
      Dune::QuadratureRule<typename GV::Grid::ctype,1>* rule_;
      int localDegree_;
      LocalMatrix localMatrix_;
  };
}
}
#endif
