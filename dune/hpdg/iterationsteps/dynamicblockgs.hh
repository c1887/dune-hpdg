// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_ITERATIONSSTEPS_DYNAMIC_BLOCK_GS_HH
#define DUNE_HPDG_ITERATIONSSTEPS_DYNAMIC_BLOCK_GS_HH
#include <vector>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

#include <dune/solvers/iterationsteps/lineariterationstep.hh>
namespace Dune {
  namespace HPDG {
    template<class Matrix, class Vector, class BitVector=std::vector<std::vector<bool>>>
    class DynamicBlockGS : public Dune::Solvers::LinearIterationStep<Matrix, Vector, BitVector>{

      using Base = Dune::Solvers::LinearIterationStep<Matrix, Vector, BitVector>;

      public:
      using Base::setProblem;

      void iterate() {
        const auto& m = *this->mat_;
        const auto& b = *this->rhs_;
        auto& x = *this->x_;
        auto r = b;

        auto blockStep = [&](size_t i) {
          const auto& row_i = m[i];


          // Compute residual
          auto& ri = r[i];
          using MatrixBlock = typename Matrix::block_type;
          const MatrixBlock* diag = nullptr;
          for (auto cIt = row_i.begin(); cIt != row_i.end(); ++cIt) {
            size_t j = cIt.index();
            cIt->mmv(x[j], ri);
            if (j == i)
              diag = &*cIt;
          }

          //using Ignore = std::bitset<V::block_type::dimension>;

          // Update iterate with correction
          auto corr = gs(*diag, ri);
          auto& xi = x[i];
          for (size_t k = 0; k < corr.size(); k++)
            xi[k]+=corr[k];
        };

        //if (direction != Direction::BACKWARD)
          for (size_t i = 0; i < x.size(); ++i)
            blockStep(i);
        //if (direction != Direction::FORWARD)
          //for (size_t i = x.size(); i > 0; --i)
            //blockStep(i - 1);
      }

      private:
      static constexpr const double defaultGsTol = 0.0;

      template<class MB, class VB>
      auto gs(const MB& m, const VB& b, double tol = defaultGsTol) {
        using K = typename VB::field_type;
        std::vector<K> x(b.size());
        for (size_t i = 0; i < m.N(); ++i) {
          const auto& mi = m[i];
          const auto& mii = mi[i];
          if (std::abs(mii) <= tol)
            continue;
          x[i] = b[i];
          const auto& end = mi.end();
          for (auto it = mi.begin(); it != end; ++it) {
            auto j = it.index();
            if (j != i)
              x[i] -= (*it) * x[j];
          }
          x[i] /= mii;
        }
        return x;
      }
    };
  }
}
#endif//DUNE_HPDG_ITERATIONSSTEPS_DYNAMIC_BLOCK_GS_HH
