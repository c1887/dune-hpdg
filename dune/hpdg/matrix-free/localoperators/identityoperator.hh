// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_IDENTITY_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_IDENTITY_OPERATOR_HH
#include "localoperator.hh"
namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /* Just some scaled identity. This is obviously only
   * useful for demonstration and testing.
   */
  template<class V, class GV>
  class IdentiyOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    public:

      void bind(const typename Base::Entity&)
      { /* No need to bind anything here */ }

      void compute() {
      }

      void write(double factor) {
        if (not copied_) {
          this->output_->axpy(factor, *(this->input_));
          copied_=true;
        }
      }

    private:
      bool copied_=false;
  };
}
}
}
#endif// DUNE_FUFEM_MATRIX_FREE_IDENTITY_OPERATOR_HH
