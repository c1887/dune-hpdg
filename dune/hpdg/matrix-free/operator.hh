// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_OPERATOR_HH
#define DUNE_FUFEM_MATRIX_FREE_OPERATOR_HH
#include <vector>
#include <dune/common/typetraits.hh>
#include <dune/common/hybridutilities.hh>

#include <dune/grid/common/rangegenerators.hh>

#include "localoperators/localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {
  template<typename Vector, typename GridView, typename LocalOperatorSet>
  struct Operator {

    // if LocalOperatorSet was not a tuple but a single type,
    // wrap it inside a tuple
    using LocalOperators = std::conditional_t<
      Dune::IsTuple<LocalOperatorSet>::value,
      LocalOperatorSet,
      std::tuple<LocalOperatorSet>>;

    Operator(const GridView& gv) :
      gv_(gv) {}

    Operator(const GridView& gv, LocalOperatorSet lo) :
      gv_(gv),
      operators_(std::move(lo)),
      factors_(std::vector<double>(Hybrid::size(operators_), 1.0)) {}

    Operator(const GridView& gv, LocalOperatorSet lo, std::vector<double> factors) :
      gv_(gv),
      operators_(std::move(lo)),
      factors_(std::move(factors)) {}

    auto& factors() {
      return factors_;
    }

    const auto& factors() const {
      return factors_;
    }

    LocalOperators& localOperators() {
      return operators_;
    }

    const LocalOperators& localOperators() const {
      return operators_;
    }


    void apply(const Vector& x, Vector& Ax) {
      Ax=0; // maybe this is not feasible for all types
      namespace H = Hybrid;
      H::forEach(operators_, [&](auto& op) {
        op.setInput(x);
        op.setOutput(Ax);
        });

      for (const auto& e: elements(gv_)) {
        H::forEach(H::integralRange(H::size(operators_)), [&](auto i) {
          if (factors_[i] != 0) { // no need to compute anything if it is multiplied by zero
            auto& op = H::elementAt(operators_, i);
            op.bind(e);
            op.compute();
            op.write(factors_[i]);
          }
        });
      }
    }

    private:
      const GridView& gv_;
      LocalOperators operators_;
      std::vector<double> factors_;
  };
}
}
}
#endif// DUNE_FUFEM_MATRIX_FREE_OPERATOR_HH
