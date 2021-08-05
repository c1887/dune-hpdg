// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_ASSEMBLER_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_ASSEMBLER_OPERATOR_HH
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/functions/backends/istlvectorbackend.hh>

#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** Operator that uses a dune-fufem LocalAssembler for the element-wise application to
   * a vector.
   *
   * This will in general give worse performance than assembling an actual matrix
   * if the operator is to be applied more than once.
   */
  template<class V, class GV, class Basis, class LA>
  class LocalAssemblerOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<double,1,1>>;

    public:

      LocalAssemblerOperator(const Basis& b, const LA& localAssembler) :
        basis_(b),
        localView_(basis_.localView()),
        localAssembler_(localAssembler) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localMatrix_.setSize(localView_.maxSize(), localView_.maxSize());

        localVector_.resize(localView_.maxSize());

      }

      void compute() {
        localAssembler_(localView_.element(), localMatrix_, localView_, localView_);

        // Add element stiffness matrix onto the global stiffness matrix
        auto inputBackend = Functions::istlVectorBackend(*(this->input_));
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          auto& rowEntry = localVector_[localRow];
          rowEntry = 0;
          for (size_t localCol=0; localCol<localView_.size(); ++localCol)
          {
            auto col = localView_.index(localCol);
            rowEntry+= inputBackend[col]*localMatrix_[localRow][localCol];
          }
        }
      }

      void write(double factor) {

        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;
        if (factor == 0.0)
          return;

        auto outputBackend = Functions::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          auto& rowEntry = outputBackend[localView_.index(localRow)];
          rowEntry += localVector_[localRow];
        }
      }

    private:
      const Basis& basis_;
      LV localView_;
      LocalMatrix localMatrix_;
      const LA& localAssembler_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
  };
}
}
}
#endif
