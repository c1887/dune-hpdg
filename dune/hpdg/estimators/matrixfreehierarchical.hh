// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/tbb_operator.hh>
#include <dune/hpdg/matrix-free/localoperators/sfipdg.hh>
//#include <dune/hpdg/matrix-free/localoperators/ipdgoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/laplaceoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgblockprojectedjacobi.hh>

namespace Dune {
namespace HPDG {

  template<class Vector, class Basis, class MatrixCreator, class MFOperator>
  void matrixFreeBlockProjectedJacobi(
      Vector& x,
      Vector rhs,
      Vector lower,
      Vector upper,
      MatrixCreator matrixCreator,
      MFOperator matrixFreeOperator,
      const Basis& basis,
      double penalty,
      size_t localIterations = 5,
      size_t globalIterations =1)
  {
    // Block Projected Jacobi:

    using GV = typename Basis::GridView;
    const auto& gv = basis.gridView();


    // Block Projected Jacobi:
    // TODO: den tnnmg löser so verpacken, dass man ihn hier nutzen kann
    //auto localSolver = gaussSeidelLocalSolver(Dune::TNNMG::ScalarObstacleSolver());
    // set up local solver: // Das ist natürlich nur die dümmstmögliche Variante
    auto gs = [&](const auto& m, const auto& bb, auto& xx, const auto& lower, const auto& upper) {

      auto x = xx;

      for (size_t iteration = 0; iteration < localIterations; iteration++) {

        auto b = bb;
        //m.mmv(xx, b);
        for(std::size_t i = 0; i < m.N(); i++) {
          for(std::size_t j = 0; j < m.M(); j++) {
            b[i]-=m[i][j]*xx[j];
          }
        }

        for (size_t i = 0; i < m.N(); ++i) {
          const auto& mi = m[i];
          const auto& mii = mi[i];
          x[i] = b[i];
          const auto& end = mi.end();
          for (auto it = mi.begin(); it != end; ++it) {
            auto j = it.index();
            if (j != i)
              x[i] -= (*it) * x[j];
          }
          x[i] /= mii;
          // truncate:
          if(x[i]<lower[i]-xx[i])
            x[i]=lower[i]-xx[i];
          if(x[i]>upper[i]-xx[i])
            x[i]=upper[i]-xx[i];
        }

        for(std::size_t i = 0; i < xx.size(); i++) {
          xx[i]+=x[i];
        }
      }
    };
    using GS = decltype(gs);
    auto jacobi = Dune::Fufem::MatrixFree::IPDGBlockProjectedJacobi<Vector, GV, Basis, GS, MatrixCreator>(basis, gs, matrixCreator, penalty);
    // Compute defect obstacle:

    jacobi.setObstacles(&lower, &upper);

    auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(jacobi)>(gv, jacobi);

    auto c = Vector::uninitializedCopy(x);
    for (size_t i=0; i < globalIterations; i++) {
      // compute residual
      auto Ac = Vector::uninitializedCopy(c);
      matrixFreeOperator.apply(x, Ac);
      rhs-=Ac;

      // compute defect obstacle:
      lower -= x;
      upper -= x;

      // reset correction vector
      c=0;

      // apply one block Jacobi step
      op.apply(rhs,c);

      x+=c;
    }
  }
}}
