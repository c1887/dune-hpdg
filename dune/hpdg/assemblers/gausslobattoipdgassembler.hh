// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <map>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/matrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/hpdg/common/mappedcache.hh>
#include <dune/hpdg/matrix-free/localoperators/gausslobattomatrices.hh>

namespace Dune {
namespace HPDG {

  /** Assembler for 2d problems using a
   * Gauss-Lobatto Qk local basis as used in Dune::HPDG.
   *
   * This implementation replaces the (slow) evaluation through
   * local finite elements by a custom implementation. In particular,
   * heavy use of caches is made.
   *
   * No other local finite elements are supported and it will not be checked if the supplied basis
   * is actually of Gauss-Lobatto Qk type!
   */
  template<class Basis>
  class GaussLobattoIPDGAssembler {
    using GV = typename Basis::GridView;
    static constexpr int dim = GV::dimension;
    using LV = typename Basis::LocalView;
    using Field = double;
    using FV = Dune::FieldVector<Field, dim>;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<Field, 1,1>>; // TODO: This is not necessarily 1x1

    static_assert(dim==2, "Sumfactorized IPDG only for dim=2 currently");

    public:

      GaussLobattoIPDGAssembler(const Basis& b, double penalty=2.0, bool dirichlet=false) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        cache_(std::bind(&GaussLobattoIPDGAssembler::matrixGenerator, this, std::placeholders::_1)),
        rules_(std::bind(&GaussLobattoIPDGAssembler::ruleGenerator, this, std::placeholders::_1))
      {}

      void bind(int degree)
      {
        localDegree_= degree;

        localSize_ = (degree+1)*(degree+1);

        rule_ = &rules_.value(localDegree_);
        matrixPair_ = &cache_.value(localDegree_);

      }

      /** Assembles the laplace matrix (\nabla \phi_i , \nabla \phi_j) on a given element */
      template<class E, class LM>
      void assembleBulk(const E& element, LM& localMatrix, int degree) {
        localMatrix=0.;

        bind(degree);
        computeBulk(element, localMatrix);
      }

      /** Assembles IPDG jump- and penalty terms on a inner edge */
      template<class E, class LC>
      void assembleEdge(const E& edge, LC& matrixContainer, int innerDegree, int outerDegree) {
        matrixContainer = 0.;

        bind(innerDegree);

        computeFace(edge, matrixContainer, outerDegree);
      }

      /** Assembles the IPDG boundary terms on a given boundary(!) edge */
      template<class E, class LM>
      void assembleBoundary(const E& edge, LM& localMatrix, int innerDegree) {
        localMatrix = 0.;
        if (!dirichlet_)
          return;
        bind(innerDegree);
        computeDirichletFace(edge, localMatrix);
      }


    private:

      /** Return the proper 1D Gauss-Lobatto quadrature rule for a given
       * degree. If the rule is not yet present in the cache, it will be added.
       */
      auto ruleGenerator(int degree) {
        int order = 2*degree;
        auto rule = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule
          (Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto);

        std::sort(rule.begin(), rule.end(), [](auto&& a, auto&& b) {
            return a.position() < b.position(); });
        return rule;
      }

      auto matrixGenerator(int degree) {
        const auto& rule = rules_.value(degree);
        return GaussLobatto::ValuesAndDerivatives(degree, rule);
      }

      template<class E, class LC>
      void computeFace(const E& edge, LC& matrixContainer, int outerDegree) {

        auto order = localDegree_;

        // set order
        order = std::max(localDegree_, outerDegree);

        // set rule to the higher rule:
        const auto& rule=rules_.value(order);

        // if the edge is nonconforming, we can not use the precalculated values and need to re-evaluate
        decltype(nonConformingMatrices(edge, rule, 0)) extra_matrices;
        auto* innerEdgeMatrices = matrixPair_;
        auto* outerEdgeMatrices = matrixPair_;

        if (!edge.conforming()) {
          // re-evaluate both inner and outer values
          extra_matrices = nonConformingMatrices(edge, rule, outerDegree);
          innerEdgeMatrices = &((*extra_matrices)[0]);
          outerEdgeMatrices = &((*extra_matrices)[1]);
        }
        // 0th case: inner order == outer order, the pointers set above are correct.
        //
        // 1st case: inner < outer, we need to set the proper order for outer and evaluate mixed for inner
        else if( localDegree_ < outerDegree) {
          extra_matrices = std::make_unique<typename decltype(extra_matrices)::element_type>(); // allocate
          auto& matrices = (*extra_matrices)[0]; // we only need the first slot

          matrices = GaussLobatto::ValuesAndDerivatives(localDegree_, rule);
          innerEdgeMatrices = &matrices;

          // set the outer matrices also:
          outerEdgeMatrices = &(cache_.value(outerDegree));
        }
        // 2nd case: outer < inner
        else if( localDegree_ > outerDegree) {
          extra_matrices = std::make_unique<typename decltype(extra_matrices)::element_type>(); // allocate
          auto& matrices = (*extra_matrices)[0]; // we only need the first slot

          matrices = GaussLobatto::ValuesAndDerivatives(outerDegree, rule);
          outerEdgeMatrices = &matrices;

          // the inner matrix pair is already fine
        }

        // Even though we do not use the outer edge, we still have to find modify the
        // penalty parameter accordingly to the higher degree to reassemble the same matrix
        // blocks as for the actual assembled matrix:
        auto penalty = penalty_ * std::pow(order, 2.0);

        using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;

        // get geometry and store it
        const auto edgeGeometry = edge.geometry();
        const auto insideGeometry = edge.inside().geometry();
        const auto outsideGeometry = edge.outside().geometry();

        const auto edgeLength = edgeGeometry.volume();

        // store gradients
        std::vector<FVdimworld> insideGradients(std::pow(localDegree_+1, 2));
        std::vector<Dune::FieldVector<double, 1>> insideValues(std::pow(localDegree_+1, 2));
        std::vector<FVdimworld> outsideGradients(std::pow(outerDegree+1, 2));
        std::vector<Dune::FieldVector<double, 1>> outerValues(std::pow(outerDegree+1, 2));

        const auto outerNormal = edge.centerUnitOuterNormal();

        // loop over quadrature points
        for (size_t pt=0; pt < rule.size(); ++pt)
        {
          // get quadrature point
          const auto& quadPos = rule[pt].position();

          // get transposed inverse of Jacobian of transformation
          const auto& invJacobian_i = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));
          const auto& invJacobian_o = outsideGeometry.jacobianInverseTransposed(edge.geometryInOutside().global(quadPos));

          // get integration factor
          const auto integrationElement = edgeGeometry.integrationElement(quadPos);

          // get gradients of shape functions on both the inside and outside element
          auto insideReferenceGradients = gradientsOnEdge(pt, *matrixPair_, *innerEdgeMatrices, edge.indexInInside());
          auto outsideReferenceGradients = gradientsOnEdge(pt, cache_.value(outerDegree), *outerEdgeMatrices, edge.indexInOutside());

          // transform gradients
          for (size_t i=0; i<insideGradients.size(); ++i) {
            invJacobian_i.mv(insideReferenceGradients[i], insideGradients[i]);
          }
          for (size_t i=0; i<outsideGradients.size(); ++i) {
            invJacobian_o.mv(outsideReferenceGradients[i], outsideGradients[i]);
          }
          // evaluate basis functions
          auto insideValues = valuesOnEdge(pt, *innerEdgeMatrices, edge.indexInInside());
          auto outsideValues = valuesOnEdge(pt, *outerEdgeMatrices, edge.indexInOutside());

          // compute matrix entries
          auto z = rule[pt].weight() * integrationElement;

          // Basis functions from inside as test functions
          for (size_t i=0; i<insideGradients.size(); ++i)
          {
            // Basis functions from inside as ansatz functions
            for (size_t j=0; j<insideGradients.size(); ++j)
            {
              // M11, see Riviere, p. 54f
              auto zij = -0.5*z*insideValues[i]*(insideGradients[j]*outerNormal);
              zij -= 0.5*z*insideValues[j]*(insideGradients[i]*outerNormal);
              zij += penalty*z/edgeLength*insideValues[i]*insideValues[j];
              matrixContainer[0][0][i][j]+=zij;

            }
            // Basis functions from outside as ansatz functions
            for (size_t j=0; j<outsideGradients.size(); ++j) {
              // M12
              auto zij = -0.5*z*insideValues[i]*(outsideGradients[j]*outerNormal);
              zij += 0.5*z*outsideValues[j]*(insideGradients[i]*outerNormal);
              zij -= penalty*z/edgeLength*insideValues[i]*outsideValues[j];
              matrixContainer[0][1][i][j]+=zij;
            }
          }
          // Basis functions from outside as test functions
          for (size_t i=0; i<outsideGradients.size(); ++i) {
            // Basis functions from inside as ansatz functions
            for (size_t j=0; j<insideGradients.size(); ++j) {
              // M21, see Riviere, p. 54f
              auto zij = 0.5*z*outsideValues[i]*(insideGradients[j]*outerNormal);
              zij -= 0.5*z*insideValues[j]*(outsideGradients[i]*outerNormal);
              zij -= penalty*z/edgeLength*outsideValues[i]*insideValues[j];
              matrixContainer[1][0][i][j]+=zij;
            }
            // Basis functions from outside as ansatz functions
            for (size_t j=0; j<outsideGradients.size(); ++j) {
              // M22
              auto zij = 0.5*z*outsideValues[i]*(outsideGradients[j]*outerNormal);
              zij += 0.5*z*outsideValues[j]*(outsideGradients[i]*outerNormal);
              zij += penalty*z/edgeLength*outsideValues[i]*outsideValues[j];
              matrixContainer[1][1][i][j]+=zij;
            }
          }
        }
      }
      template<class E, class LM>
      void computeDirichletFace(const E& edge, LM& localMatrix) {

        auto penalty = penalty_ * std::pow(localDegree_, 2.0);

        const auto& rule = *rule_;
        using FVdimworld = FieldVector<Field, GV::Grid::dimensionworld>;

        // get geometry and store it
        const auto edgeGeometry = edge.geometry();
        const auto insideGeometry = edge.inside().geometry();

        const auto edgeLength = edgeGeometry.volume();

        // store gradients
        std::vector<FVdimworld> insideGradients(std::pow(localDegree_+1, 2));
        std::vector<Dune::FieldVector<double, 1>> insideValues(std::pow(localDegree_+1, 2));

        const auto outerNormal = edge.centerUnitOuterNormal();

        // loop over quadrature points
        for (size_t pt=0; pt < rule.size(); ++pt)
        {
          // get quadrature point
          const auto& quadPos = rule[pt].position();

          // get transposed inverse of Jacobian of transformation
          const auto& invJacobian_i = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));
          // get integration factor
          const auto integrationElement = edgeGeometry.integrationElement(quadPos);

          // get gradients of shape functions on both the inside and outside element
          auto insideReferenceGradients = gradientsOnEdge(pt, *matrixPair_, *matrixPair_, edge.indexInInside());

          // transform gradients
          for (size_t i=0; i<insideGradients.size(); ++i) {
            invJacobian_i.mv(insideReferenceGradients[i], insideGradients[i]);
          }
          // evaluate basis functions
          auto insideValues = valuesOnEdge(pt, *matrixPair_, edge.indexInInside());

          // compute matrix entries
          auto z = rule[pt].weight() * integrationElement;

          // Basis functions from inside as test functions
          for (size_t i=0; i<insideGradients.size(); ++i)
          {
            // Basis functions from inside as ansatz functions
            for (size_t j=0; j<insideGradients.size(); ++j)
            {
              // M11, see Riviere, p. 54f
              auto zij = -z*insideValues[i]*(insideGradients[j]*outerNormal)
               - z*insideValues[j]*(insideGradients[i]*outerNormal)
               + penalty*z/edgeLength*insideValues[i]*insideValues[j];
              localMatrix[i][j]+=zij;
            }
          }
        }
      }

      template<class E, class LM>
      void computeBulk(const E& element, LM& localMatrix) {
        auto geometry = element.geometry();

        // store gradients of shape functions and base functions
        using Gradient = FieldVector<double, dim>;
        std::vector<Gradient> gradients(localSize_);

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

            // transform gradients
            for (size_t i=0; i<gradients.size(); ++i)
              invJacobian.mv(referenceGradients[i], gradients[i]);

            // compute matrix entries
            auto z = (*rule_)[q0].weight() * (*rule_)[q1].weight() * integrationElement;
            for (size_t i=0; i<localSize_; ++i)
            {
              for (size_t j=i+1; j<localSize_; ++j)
              {
                double zij = (gradients[i] * gradients[j]) * z;
                localMatrix[i][j]+=zij;
                localMatrix[j][i]+=zij;
              }
              localMatrix[i][i]+= (gradients[i] * gradients[i]) * z;
            }

          }
        }
      }

      auto computeGradients(size_t q0, size_t q1) const {

        using Gradient = FieldVector<double, dim>;
        std::vector<Gradient> referenceGradients(localSize_);

        for(size_t i = 0; i < localDegree_+1; i++) {
          for(size_t j = 0; j < localDegree_+1; j++) {
            referenceGradients[flatIndex(i,j, localDegree_)] = {{matrixPair_->derivatives[i][q0]*matrixPair_->values[j][q1], matrixPair_->values[i][q0]*matrixPair_->derivatives[j][q1]}};
          }
        }

        return referenceGradients;
      }

      template<class MP>
      auto valuesOnEdge(size_t quad_nr, const MP& edgeMatrices, int  edgeNumber) const {
        auto deg = edgeMatrices.values.N() -1;

        using Value = FieldVector<double, 1>;
        std::vector<Value> vals(std::pow(deg+1, dim), 0.);

        for(size_t i = 0; i < deg+1; i++) {
          switch(edgeNumber) {
            case 0:
              vals[flatIndex(0,i, deg)] = {{edgeMatrices.values[i][quad_nr]}};
              break;
            case 1:
              vals[flatIndex(deg,i, deg)] = {{edgeMatrices.values[i][quad_nr]}};
              break;
            case 2:
              vals[flatIndex(i,0, deg)] = {{edgeMatrices.values[i][quad_nr]}};
              break;
            case 3:
              vals[flatIndex(i,deg, deg)] = {{edgeMatrices.values[i][quad_nr]}};
              break;
            default:
              DUNE_THROW(Dune::Exception, "Illegal edge number provided");
            }
          }
        return vals;
      }

      template<class MP>
      auto gradientsOnEdge(size_t quad_nr, const MP& fullMatrixPair, const MP& edgeMatrices, int edgeNumber) const {
        auto deg = edgeMatrices.values.N() -1;

        using Gradient = FieldVector<double, dim>;
        std::vector<Gradient> referenceGradients(std::pow(deg+1, dim));


        /* the size of the underlying rules for the fullMatrixpair and the edgeMatrices might differ.
         * This doenst matter, though, because we only use the first and last quadrature node,
         * which are always 0 and 1, respectively, since we use the Gauss-Lobatto quadrature rule.
         */
        auto lastIdx = fullMatrixPair.values.M()-1;
        for (size_t i = 0; i< deg +1; i++) {
          for(size_t j = 0; j < deg+1; j++) {
            switch(edgeNumber) {
              case 0:
                referenceGradients[flatIndex(i,j, deg)] = {{fullMatrixPair.derivatives[i][0]*edgeMatrices.values[j][quad_nr],  fullMatrixPair.values[i][0]*edgeMatrices.derivatives[j][quad_nr]}};
                break;
              case 1:
                referenceGradients[flatIndex(i,j, deg)] = {{fullMatrixPair.derivatives[i][lastIdx]*edgeMatrices.values[j][quad_nr], fullMatrixPair.values[i][lastIdx]*edgeMatrices.derivatives[j][quad_nr]}};
                break;
              case 2:
                referenceGradients[flatIndex(i,j, deg)] = {{edgeMatrices.derivatives[i][quad_nr]*fullMatrixPair.values[j][0], edgeMatrices.values[i][quad_nr]*fullMatrixPair.derivatives[j][0]}};
                break;
              case 3:
                referenceGradients[flatIndex(i,j, deg)] = {{edgeMatrices.derivatives[i][quad_nr]*fullMatrixPair.values[j][lastIdx], edgeMatrices.values[i][quad_nr]*fullMatrixPair.derivatives[j][lastIdx]}};
                break;
              default:
                DUNE_THROW(Dune::Exception, "Illegal edge number provided");
            }
          }
        }
        return referenceGradients;
      }

      template<class IS, class R>
      auto nonConformingMatrices(const IS& is, const R& rule, int outerDegree)
      {
        // Stupid quadrature rule does not allow me to edit the quad points :(
        struct QP {
          typename R::value_type::Vector pos;
          auto& position() {
            return pos;
          }
          const auto& position() const{
            return pos;
          }
        };

        auto mutableCopy = [] (const R& r) {
          std::vector<QP> v(r.size());
          for(std::size_t i = 0; i < r.size(); i++) {
            v[i].pos = r[i].position();
          }
          return v;
        };

        // compute inner and outer matrix pair for the noncorming edge.
        // step 0.: compute new quad notes:
        auto inner_quad = mutableCopy(rule);
        auto outer_quad = mutableCopy(rule);

        for (size_t i = 0; i < rule.size(); i++) {
          inner_quad[i].position() = is.geometryInInside().global(rule[i].position())[is.indexInInside() < 2]; // we only take the coordinate we need, which will be the first for edge number 2 and 3, and the second for 0 and 1
          outer_quad[i].position() = is.geometryInOutside().global(rule[i].position())[is.indexInOutside() < 2];
        }
        auto ret = std::make_unique<std::array<GaussLobatto::ValuesAndDerivatives,2>>(); // contains the ValuesAndDerivatives for the inner (0) and outer (1) position

        (*ret)[0] = GaussLobatto::ValuesAndDerivatives(localDegree_, inner_quad);
        (*ret)[1] = GaussLobatto::ValuesAndDerivatives(outerDegree, outer_quad);

        return ret;
      }
      auto flatIndex(unsigned int i0, unsigned int i1, int k) const {
        return i0 + i1*(k+1);
      }

      const Basis& basis_;
      double penalty_;
      bool dirichlet_;
      MappedCache<GaussLobatto::ValuesAndDerivatives, int> cache_; // contains all lagrange Polynomials at all quadrature points and all derivatives of said polynomials at all quad points
      MappedCache<Dune::QuadratureRule<typename GV::Grid::ctype, 1>, int> rules_;
      const typename decltype(cache_)::mapped_type* matrixPair_; // Current matrix pair
      Dune::QuadratureRule<typename GV::Grid::ctype,1>* rule_;
      int localDegree_;
      size_t localSize_;
  };
}
}
