// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_ITERATIONSSTEPS_L1_SMOOTHER_HH
#define DUNE_HPDG_ITERATIONSSTEPS_L1_SMOOTHER_HH
#include <vector>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

#include <dune/solvers/iterationsteps/lineariterationstep.hh>
namespace Dune {
  namespace HPDG {

    /** This is just a very reduced version of the BlockGS in dune-solvers.
     *
     * The feature of this implementation is that it can be used with dynamic matrix and vector
     * structures.
     */
    template<class Matrix, class Vector, class BitVector=std::vector<std::vector<bool>>>
    class L1Smoother : public Dune::Solvers::LinearIterationStep<Matrix, Vector, BitVector>{

      using Base = Dune::Solvers::LinearIterationStep<Matrix, Vector, BitVector>;

      public:
      using Base::setProblem;

      L1Smoother(const std::vector<std::size_t>& ghosts) :
        ghosts_(ghosts)
      {}

      void iterate() {
        const auto& m = *this->mat_;
        const auto& b = *this->rhs_;
        auto& x = *this->x_;
        auto r = b;

        for (size_t i = 0; i < x.size(); ++i) {
          const auto& row_i = m[i];

          // if ghost, continue
#if 0
          {
            auto it = std::find(ghosts_.begin(), ghosts_.end(), i);
            if (it != ghosts_.end())
              continue;
          }
#endif


          // Compute residual
          auto& ri = r[i];
          using MatrixBlock = typename Matrix::block_type;
          const MatrixBlock* diag = nullptr;

          // TODO: More efficient way to copy matrix block
          auto l1_block = Dune::Matrix<FieldMatrix<double,1,1>>(ri.size(), ri.size());

          for (auto cIt = row_i.begin(); cIt != row_i.end(); ++cIt) {
            size_t j = cIt.index();
            cIt->mmv(x[j], ri);
          }

          if (m.exists(i,i)) {
            diag = &(m[i][i]);
            l1_block = *diag;
          }
          else {
            DUNE_THROW(Dune::Exception, "Diagonal block (" << i <<", " <<i <<") is null");
          }

          // TODO: This is O(N^2) !!
          for(const auto& ghost : ghosts_) {
            //if (m.exists(i,ghost) && ghost < i) {
            if (m.exists(i,ghost)) {
              for (std::size_t l = 0; l < l1_block.N(); ++l) {
                auto& d = l1_block[l][l];
                for (const auto& val: row_i[ghost][l]) {
                  d+=std::abs(val);
                }
              }
            }
          }

          // Update iterate with correction
          auto corr = gs(l1_block, ri, 0);
          //auto corr = gs(*diag, ri, 0);
          auto& xi = x[i];
          for (size_t k = 0; k < corr.size(); k++)
            xi[k]+=corr[k];
        }

      }

      private:
      static constexpr const double defaultGsTol = 0.0;
      const std::vector<std::size_t>& ghosts_;

      template<class MB, class VB>
      auto gs(const MB& m, const VB& b, double tol = defaultGsTol) {
        auto x =b;
        x=0;
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
