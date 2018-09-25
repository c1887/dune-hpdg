// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_OPERATOR_HH

#include <dune/hpdg/matrix-free/operator.hh>

namespace Dune {
namespace Fufem {
namespace MatrixFree {
  template<typename Vector, typename GridView>
  class LocalOperator {

    template<class VV, class GG, class LS>
    friend class Dune::Fufem::MatrixFree::Operator;

    public:

    using Entity =  typename GridView::template Codim<0>::Entity;

    LocalOperator(const Vector& input, Vector& output, double factor=1.) :
      input_(&input),
      output_(&output),
      factor_(factor) {}

    LocalOperator() :
      input_(nullptr),
      output_(nullptr),
      factor_(1.0) {}

    void setInput(const Vector& i) {
      input_=&i;
    }

    void setOutput(Vector& o) {
      output_=&o;
    }

    /** Get the factor that is currently associated with
     * this local operator */
    auto factor() const {
      return factor_;
    }

    /** Set the factor for this local operator */
    void setFactor(double factor) {
      factor_ = factor;
    }

    /* An implementation should implement the following:
     *
     *  void bind(const Entity& e);
     *    Attach to an entity e.
     *
     *  void compute();
     *    Do the actual computations for the given element. Do not write to the global vector here!
     *
     *  void write(double factor);
     *    Write to the vector. This (if possible) contain as few code as possible because in a parallel setup
     *    this method will lock the other threads.
     *
     */

    protected:

    const Vector* input_;
    Vector* output_;
    double factor_;
  };
}
}
}
#endif// DUNE_FUFEM_MATRIX_FREE_LOCAL_OPERATOR_HH
