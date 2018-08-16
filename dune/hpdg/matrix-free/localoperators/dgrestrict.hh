// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_OPERATOR_RESTRICT_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_OPERATOR_RESTRICT_HH
#include<dune/common/fmatrix.hh>
#include<dune/istl/matrix.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper

namespace Dune {
namespace Fufem {
namespace MatrixFree {
  /** Local Operator that restricts a function living on the fine basis to the coarse basis.
   * DGRestriction here means the transpose operation to interpolation.
   * It is assumed that both bases are defined on the SAME GRIDVIEW and all elements have the SAME GEOMETRY!
   * Also, the coarse basis must have constant degree on all elements.
   * Prototypical example would be restricting to (not-DG) Q1.
   */
  template<typename Vector, typename GridView, typename CoarseBasis, typename FineBasis>
  class DGRestrictionOperator {

    public:

    using Entity =  typename GridView::template Codim<0>::Entity;

    DGRestrictionOperator(const CoarseBasis& coarseBasis, const FineBasis& fineBasis) :
      cbasis_(coarseBasis),
      fbasis_(fineBasis),
      clv(cbasis_.localView()),
      flv(fbasis_.localView())
    {

      using CoarseFiniteElement = typename CoarseBasis::LocalView::Tree::FiniteElement;
      using FunctionBaseClass = typename Dune::LocalFiniteElementFunctionBase<CoarseFiniteElement>::type;
      using LocalBasisWrapper = LocalBasisComponentWrapper<typename CoarseFiniteElement::Traits::LocalBasisType, FunctionBaseClass>;

      const auto& gv = fbasis_.gridView();
      for(const auto& element : elements(gv)) {
        flv.bind(element);
        const auto& fineFE = flv.tree().finiteElement();
        // check if we have already assembled for this order:
        int degree = fineFE.localBasis().order();
        {
          auto it = cache_.find(degree);
          if (it!=cache_.end())
            continue;
        }

        clv.bind(element);

        auto matrix = Matrix(flv.size(), clv.size());

        const auto& coarseFE = clv.tree().finiteElement();
        const auto numCoarse = clv.size();

        std::vector<typename CoarseFiniteElement::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);
        LocalBasisWrapper coarseBasisFunction(coarseFE.localBasis(),0);

        for (size_t j = 0; j<numCoarse; j++)
        {
          /* Interpolate values of the j-th coarse function*/
          coarseBasisFunction.setIndex(j);
          //auto globalCoarseIndex = coarseView.index(j);
          fineFE.localInterpolation().interpolate(coarseBasisFunction, values);

          for(std::size_t i = 0; i < flv.size(); i++) {
            matrix[i][j]+=values[i];
          }
        }
        cache_[degree]=std::move(matrix);
      }
    }

    DGRestrictionOperator() = delete;

    void setInput(const Vector& i) {
      input_=&i;
    }

    void setOutput(Vector& o) {
      output_=&o;
    }

    void bind(const Entity& e){
      // fine the right order:
      flv.bind(e);
      clv.bind(e);
      const auto& fineFE = flv.tree().finiteElement();
      int degree = fineFE.localBasis().order();
      currentMat_ = &cache_[degree];
      buffer_.resize(clv.size());
      buffer_=0.0;

      inputArray_ = &((*input_)[flv.index(0)]);
    }

    void compute() {
      // compute matrix^T vector product by hand (we cant be sure on the vector types)

      for(std::size_t i = 0; i < currentMat_->N(); i++) {
        const auto& row = (*currentMat_)[i];
        std::size_t j =0;
        for(auto entry = row.begin(); entry!= row.end(); ++entry, ++j) {
          buffer_[j]+=*entry * inputArray_[i];
        }
      }
    }
    void write(double factor) {
      for(std::size_t i = 0; i < clv.size(); i++) {
        (*output_)[clv.index(i)] += factor*buffer_[i];
      }
    }

    protected:
    using FM = Dune::FieldMatrix<double, 1,1>;
    using Matrix = Dune::Matrix<FM>;
    const CoarseBasis& cbasis_;
    const FineBasis& fbasis_;
    typename CoarseBasis::LocalView clv;
    typename FineBasis::LocalView flv;
    std::map<int, Matrix> cache_;
    Matrix* currentMat_;

    const Vector* input_;
    Vector* output_;
    Dune::BlockVector<Dune::FieldVector<double,1>> buffer_;

    using FV = Dune::FieldVector<double,1>;
    const FV* inputArray_;
  };
}
}
}
#endif
