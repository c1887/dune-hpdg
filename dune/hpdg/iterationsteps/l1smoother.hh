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

      void preprocess() {
        ghostRegularization_ = makeDynamicBlockVector(*this->mat_);
        ghostRegularization_ = 0.;

        assert(ghostRegularization_.size() == this->mat_->M());
        const auto& m = *this->mat_;

        for(auto&& ghostIdx : ghosts_) {
          // search through ghost row to find the adjacent elements
          const auto& mg = m[ghostIdx];
          for(auto it = mg.begin(); it!=mg.end(); ++it) {

            // we found a row, now store the l1 norm of the row-ghost block
            auto row = it.index();
            if (row==ghostIdx)
              continue;
            const auto& ghostBlock = m[row][ghostIdx];
            auto& dii = ghostRegularization_[row];
            for(std::size_t j = 0; j < ghostBlock.N(); j++) {
              // add l1-norm of the rows
              for(const auto& entry: ghostBlock[j]) {
                dii[j]+=std::abs(entry);
              }
            }
          }
        }
      }

      void iterate() {
        const auto& m = *this->mat_;
        const auto& b = *this->rhs_;
        auto& x = *this->x_;
        auto r = b;

        assert(ghostRegularization_.size() == this->mat_->M());
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
          auto corr = gs(*diag, ri, ghostRegularization_[i], 0);
          //auto corr = gs(*diag, ri, 0);
          auto& xi = x[i];
          for (size_t k = 0; k < corr.size(); k++)
            xi[k]+=corr[k];
        }

      }

      private:
      static constexpr const double defaultGsTol = 0.0;
      const std::vector<std::size_t>& ghosts_;
      DynamicBlockVector<FieldVector<double,1>> ghostRegularization_;

      /** Performs Gauss-Seidel locally where a diagonal
       * vector d is added onto the matrix diagonal for regularization
       */
      template<class MB, class VB, class Regularized>
      auto gs(const MB& m, const VB& b, const Regularized& d, double tol = defaultGsTol) {
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
          x[i] /= (mii+d[i]);
        }
        return x;
      }
    };
  }
}
#endif
