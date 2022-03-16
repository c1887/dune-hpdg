#ifndef DUNE_FUFEM_WEIGHTED_LAPLACE_ASSEMBLER_HH
#define DUNE_FUFEM_WEIGHTED_LAPLACE_ASSEMBLER_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/matrix.hh>

#include <dune/matrix-vector/addtodiagonal.hh>

#include "dune/fufem/quadraturerules/quadraturerulecache.hh"

#include "dune/fufem/assemblers/localoperatorassembler.hh"

namespace Dune::Fufem {

/** \brief Local assembler for the Laplace problem */
template<class GridType,
         class F,
         class TrialLocalFE,
         class AnsatzLocalFE,
         class T = Dune::FieldMatrix<double, 1, 1>>
class DuneFunctionsWeightedLaplaceAssembler
  : public LocalOperatorAssembler<GridType, TrialLocalFE, AnsatzLocalFE, T>
{
private:
  typedef LocalOperatorAssembler<GridType, TrialLocalFE, AnsatzLocalFE, T> Base;
  static const int dim = GridType::dimension;
  static const int dimworld = GridType::dimensionworld;

public:
  typedef typename Base::Element Element;
  typedef typename Base::BoolMatrix BoolMatrix;
  typedef typename Base::LocalMatrix LocalMatrix;

  DuneFunctionsWeightedLaplaceAssembler(const F& f,
                                        const QuadratureRuleKey& key)
    : f_(f)
    , key_(key)
  {}

  void indices(const Element& element,
               BoolMatrix& isNonZero,
               const TrialLocalFE& tFE,
               const AnsatzLocalFE& aFE) const
  {
    isNonZero = true;
  }

  template<class BoundaryIterator>
  void indices(const BoundaryIterator& it,
               BoolMatrix& isNonZero,
               const TrialLocalFE& tFE,
               const AnsatzLocalFE& aFE) const
  {
    isNonZero = true;
  }

  /** \brief Assemble the local stiffness matrix for a given element
   */
  void assemble(const Element& element,
                LocalMatrix& localMatrix,
                const TrialLocalFE& tFE,
                const AnsatzLocalFE& aFE) const
  {
    typedef typename Dune::template FieldVector<double, dim> FVdim;
    typedef typename Dune::template FieldVector<double, dimworld> FVdimworld;
    typedef typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType
      JacobianType;

    // Make sure we got suitable shape functions
    assert(tFE.type() == element.type());
    assert(aFE.type() == element.type());

    // check if ansatz local fe = test local fe
    //            if (not Base::isSameFE(tFE, aFE))
    //                DUNE_THROW(Dune::NotImplemented, "LaplaceAssembler is only
    //                implemented for ansatz space=test space!");

    const auto rows = localMatrix.N();
    const auto cols = localMatrix.M();

    // get geometry and store it
    const auto geometry = element.geometry();

    localMatrix = 0.0;

    // get quadrature rule
    QuadratureRuleKey tFEquad(tFE);
    QuadratureRuleKey quadKey = tFEquad.derivative().square().product(key_);
    const Dune::template QuadratureRule<double, dim>& quad =
      QuadratureRuleCache<double, dim>::rule(quadKey);

    // store gradients of shape functions and base functions
    std::vector<JacobianType> referenceGradients(tFE.localBasis().size());
    std::vector<FVdimworld> gradients(tFE.localBasis().size());

    auto lf = localFunction(f_);
    lf.bind(element);

    // loop over quadrature points
    for (size_t pt = 0; pt < quad.size(); ++pt) {
      // get quadrature point
      const FVdim& quadPos = quad[pt].position();
      const auto& wval = lf(quadPos);

      // get transposed inverse of Jacobian of transformation
      const auto& invJacobian =
        geometry.jacobianInverseTransposed(quadPos);

      // get integration factor
      const double integrationElement = geometry.integrationElement(quadPos);

      // get gradients of shape functions
      tFE.localBasis().evaluateJacobian(quadPos, referenceGradients);

      // transform gradients
      for (size_t i = 0; i < gradients.size(); ++i) {
        invJacobian.mv(referenceGradients[i][0], gradients[i]);
      }

      // compute matrix entries
      const double z = quad[pt].weight() * integrationElement;
      for (int i = 0; i < rows; ++i) {
        for (int j = i + 1; j < cols; ++j) {
          const double zij = (gradients[i] * gradients[j]) * z * wval;
          Dune::MatrixVector::addToDiagonal(localMatrix[i][j], zij);
          Dune::MatrixVector::addToDiagonal(localMatrix[j][i], zij);
        }
        Dune::MatrixVector::addToDiagonal(
          localMatrix[i][i], (gradients[i] * gradients[i]) * z * wval);
      }
    }
  }

private:
  const F& f_;
  QuadratureRuleKey key_;
};
}
#endif