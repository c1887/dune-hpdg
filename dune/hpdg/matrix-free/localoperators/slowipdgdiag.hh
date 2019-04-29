// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_SLOW_IPDG_BLOCK_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_SLOW_IPDG_BLOCK_JACOBI_HH
#include <dune/common/fmatrix.hh>
#include <dune/common/math.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/variableipdg.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/istl/matrix.hh>

#include <dune/hpdg/common/indexedcache.hh>
#include <dune/hpdg/common/mutablequadraturenodes.hh>
#include <dune/hpdg/matrix-free/localoperators/gausslobattomatrices.hh>

/** This is a factory for the diagonal block of an IPDG Laplace discretization.
 * After binding to an element via bind(), you can get the corresponding
 * diagonal matrix block of the stiffness matrix via matrix().
 *
 * This is particulary useful for block Jacobi algorithms.
 */
namespace Dune {
namespace HPDG {

  template<class Basis>
  class SlowIPDGDiagonalBlock {
    using GV = typename Basis::GridView;
    static constexpr int dim = GV::dimension;
    using LV = typename Basis::LocalView;
    using Field = double;
    using FV = Dune::FieldVector<Field, dim>;
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<Field, 1,1>>; // TODO: This is not necessarily 1x1

    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    public:

      SlowIPDGDiagonalBlock(const Basis& b, double penalty=2.0, bool dirichlet=false) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        localView_(basis_.localView())
      {}

      template<class Entity>
      void bind(const Entity& e)
      {
        localView_.bind(e);

        localMatrix_.setSize(localView_.size(), localView_.size());
        localMatrix_=0.;
      }

      /** Get the matrix for the bound element */
      const auto& matrix() {
        compute();
        return localMatrix_;
      }

    private:
      // compute matrix diagonal block
      // and solve block for input part
      void compute() {
        computeBulk();
        computeFace();
      }

      void computeFace() {
        const auto& fe = localView_.tree().finiteElement();
        using FE = std::decay_t<decltype(fe)>;
        auto localAssembler = VInteriorPenaltyDGAssembler<typename GV::Grid, FE, FE>(penalty_, dirichlet_);

        // this is of course way too much work :(
        // TODO: We only need the inner x inner terms but we do compute all of the stuff
        // with the slow assembler. This is a big no.
        using MatrixContainer = Dune::Matrix<LocalMatrix>;
        auto mc = MatrixContainer(2,2);

        auto outerView = basis_.localView();

        for (const auto& is: intersections(basis_.gridView(), localView_.element())) {
          if (not is.neighbor()) {
            // TODO Das funktioniert hier irgendwie nicht. Tun wir erstmal so, als sei das Neumann
            //continue; // TODO <-- Die Zeile dann löschen
            auto tmp =localMatrix_;

            localAssembler.assemble(is, tmp, fe, fe);
            //localMatrix_+=tmp;
            for(std::size_t i = 0; i < fe.size(); i++) {
              auto row = localView_.index(i)[1];
              for(std::size_t j = 0; j < fe.size(); j++) {
                auto col = localView_.index(j)[1];
                localMatrix_[row][col]+=tmp[i][j];
              }
            }
          }
          else {
            outerView.bind(is.outside());
            const auto& ofe = outerView.tree().finiteElement();

            mc[0][0].setSize(fe.size(), fe.size());
            mc[0][1].setSize(fe.size(), ofe.size());
            mc[1][0].setSize(ofe.size(), fe.size());
            mc[1][1].setSize(ofe.size(), ofe.size());

            localAssembler.assembleBlockwise(is, mc, fe, fe, ofe, ofe);

            //localMatrix_+=mc[0][0];
            //
            const auto& tmp = mc[0][0];

            for(std::size_t i = 0; i < fe.size(); i++) {
              auto row = localView_.index(i)[1];
              for(std::size_t j = 0; j < fe.size(); j++) {
                auto col = localView_.index(j)[1];
                localMatrix_[row][col]+=tmp[i][j];
              }
            }
          }
        }
      }

      void computeBulk() {
        const auto& fe = localView_.tree().finiteElement();
        using FE = std::decay_t<decltype(fe)>;
        auto localAssembler = LaplaceAssembler<typename GV::Grid, FE, FE>();
        auto tmp = localMatrix_;
        localAssembler.assemble(localView_.element(), tmp, fe, fe);
        //localMatrix_+=tmp;

            for(std::size_t i = 0; i < fe.size(); i++) {
              auto row = localView_.index(i)[1];
              for(std::size_t j = 0; j < fe.size(); j++) {
                auto col = localView_.index(j)[1];
                localMatrix_[row][col]+=tmp[i][j];
              }
            }
      }



      // members:
      const Basis& basis_;
      double penalty_;
      bool dirichlet_;
      LV localView_;
      LocalMatrix localMatrix_;
  };
}
}
#endif
