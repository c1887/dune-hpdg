// -*- tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set ts=8 sw=4 et sts=4:
#ifndef DUNE_FUNCTIONS_WEIGHTED_MASS_ASSEMBLER_HH
#define DUNE_FUNCTIONS_WEIGHTED_MASS_ASSEMBLER_HH

#include <memory>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/matrix.hh>

#include <dune/matrix-vector/addtodiagonal.hh>

#include "dune/fufem/quadraturerules/quadraturerulecache.hh"

#include "dune/fufem/assemblers/localoperatorassembler.hh"

namespace Dune::Fufem {
/**
 * \brief Weighted local mass assembler
 *
 * This local operator assembler allows to assemble the bilinear form
 * \f[
 *   b(u,v) = \int_\Omega f(x) u(x) v(x)\,dx
 * \f]
 * for a fixed weighting function \f$ f:\Omega \to \mathbb{R} \f$.
 **/
template<class GridType,
         class TrialLocalFE,
         class AnsatzLocalFE,
         class FunctionType,
         class T = Dune::FieldMatrix<double, 1, 1>>
class DuneFunctionsWeightedMassAssembler
  : public LocalOperatorAssembler<GridType, TrialLocalFE, AnsatzLocalFE, T>
{
private:
  typedef LocalOperatorAssembler<GridType, TrialLocalFE, AnsatzLocalFE, T> Base;
  static const int dim = GridType::dimension;

public:
  typedef typename Base::Element Element;
  typedef typename Element::Geometry Geometry;
  typedef typename Base::BoolMatrix BoolMatrix;
  typedef typename Base::LocalMatrix LocalMatrix;

  /**
   * \brief Construct DuneFunctionsWeightedMassAssembler
   *
   * \param grid
   */
  DuneFunctionsWeightedMassAssembler(const GridType& grid, // TODO Remove grid
                                     const FunctionType& weight,
                                     QuadratureRuleKey weightQuadKey)
    : weightGridFunction_(weight)
    , weightQuadKey_(weightQuadKey)
  {}

  void indices(const Element& element,
               BoolMatrix& isNonZero,
               const TrialLocalFE& tFE,
               const AnsatzLocalFE& aFE) const
  {
    isNonZero = true;
  }

  void assemble(const Element& element,
                LocalMatrix& localMatrix,
                const TrialLocalFE& tFE,
                const AnsatzLocalFE& aFE) const
  {
    using RangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

    auto lf = localFunction(weightGridFunction_);
    lf.bind(element);

    int rows = localMatrix.N();
    int cols = localMatrix.M();

    // get geometry and store it
    const auto& geometry = element.geometry();

    localMatrix = 0.0;

    // get quadrature rule
    auto quadKey = QuadratureRuleKey(weightQuadKey_)
                     .product(QuadratureRuleKey(tFE))
                     .product(QuadratureRuleKey(aFE));
    const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

    // store values of shape functions
    std::vector<RangeType> values(tFE.localBasis().size());

    // loop over quadrature points
    for (size_t pt = 0; pt < quad.size(); ++pt) {
      // get quadrature point
      const auto& quadPos = quad[pt].position();

      // get integration factor
      const auto integrationElement = geometry.integrationElement(quadPos);

      // evaluate basis functions
      tFE.localBasis().evaluateFunction(quadPos, values);

      auto z = lf(quadPos);

      // compute matrix entries
      z *= quad[pt].weight() * integrationElement;
      for (int i = 0; i < rows; ++i) {
        double zi = values[i] * z;

        for (int j = i + 1; j < cols; ++j) {
          double zij = values[j] * zi;
          Dune::MatrixVector::addToDiagonal(localMatrix[i][j], zij);
          Dune::MatrixVector::addToDiagonal(localMatrix[j][i], zij);
        }
        Dune::MatrixVector::addToDiagonal(localMatrix[i][i], values[i] * zi);
      }
    }
  }

protected:
  const FunctionType& weightGridFunction_;
  QuadratureRuleKey weightQuadKey_;
};
}
#endif
