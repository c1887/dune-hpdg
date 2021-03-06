// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_ITERATIONSSTEPS_DYNAMIC_BLOCK_GS_HH
#define DUNE_HPDG_ITERATIONSSTEPS_DYNAMIC_BLOCK_GS_HH
#include <vector>
#include <functional>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

#include <dune/common/typetraits.hh>

#include <dune/solvers/iterationsteps/lineariterationstep.hh>
namespace Dune {
  namespace HPDG {
    namespace Imp {
      struct GSCore {
        template<typename MB, typename VB>
        auto operator()
        (const MB& m, const VB& b) {
        auto x = autoCopy(b);
        x=0;
        for (size_t i = 0; i < m.N(); ++i) {
          const auto& mi = m[i];
          const auto& mii = mi[i];
          if (std::abs(mii) == 0.)
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

      // for nesting more blocks
      template<typename LocalSolver = GSCore>
      struct GSOuterLoop {
        template<typename MB, typename VB>
        auto operator()
        (const MB& m, const VB& b) {
          auto x = autoCopy(b);
          x = 0;

          for (size_t i = 0; i < x.size(); ++i) {
            const auto& row_i = m[i];

            // Compute residual
            auto ri = autoCopy(b[i]);
            using MatrixBlock = typename MB::block_type;
            const MatrixBlock* diag = nullptr;
            for (auto cIt = row_i.begin(); cIt != row_i.end(); ++cIt) {
              size_t j = cIt.index();
              cIt->mmv(x[j], ri);
            }

            if (m.exists(i, i)) {
              diag = &(m[i][i]);
            } else {
              DUNE_THROW(Dune::Exception,
                         "Diagonal block (" << i << ", " << i << ") is null");
            }

            // Update iterate with correction
            auto corr = LocalSolver()(*diag, ri);
            auto& xi = x[i];
            for (size_t k = 0; k < corr.size(); k++)
              xi[k] += corr[k];
          }
          return x;
        }
      };
    }

    /** This is just a very reduced version of the BlockGS in dune-solvers.
     *
     * The feature of this implementation is that it can be used with dynamic matrix and vector
     * structures.
     */
    template<class Matrix, class Vector, class BitVector=std::vector<std::vector<bool>>, class LocalSolver = Imp::GSCore>
    class DynamicBlockGS : public Dune::Solvers::LinearIterationStep<Matrix, Vector, BitVector>{

      using Base = Dune::Solvers::LinearIterationStep<Matrix, Vector, BitVector>;

      public:
      using Base::setProblem;

      void iterate() {
        const auto& m = *this->mat_;
        const auto& b = *this->rhs_;
        auto& x = *this->x_;
        auto r = autoCopy(b);

        for (size_t i = 0; i < x.size(); ++i) {
          const auto& row_i = m[i];


          // Compute residual
          auto& ri = r[i];
          using MatrixBlock = typename Matrix::block_type;
          const MatrixBlock* diag = nullptr;
          for (auto cIt = row_i.begin(); cIt != row_i.end(); ++cIt) {
            size_t j = cIt.index();
            cIt->mmv(x[j], ri);
          }

          if (m.exists(i,i)) {
            diag = &(m[i][i]);
          }
          else {
            DUNE_THROW(Dune::Exception, "Diagonal block (" << i <<", " <<i <<") is null");
          }

          // Update iterate with correction
          auto corr = LocalSolver()(*diag, ri);
          auto& xi = x[i];
          for (size_t k = 0; k < corr.size(); k++)
            xi[k]+=corr[k];
        }
      }
    };
  }
}
#endif//DUNE_HPDG_ITERATIONSSTEPS_DYNAMIC_BLOCK_GS_HH
