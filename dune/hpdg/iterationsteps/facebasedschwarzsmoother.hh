#ifndef DUNE_SOLVERS_FACEBASEDSCHWARZSMOOTHER
#define DUNE_SOLVERS_FACEBASEDSCHWARZSMOOTHER
#include <cmath>
#include <algorithm>
#include <dune/solvers/iterationsteps/lineariterationstep.hh>
#include <dune/solvers/common/defaultbitvector.hh>
#include <dune/istl/matrix.hh>
#include <dune/matrix-vector/subbcrsmatrixview.hh>


/* this whole thing can just work for 2d right now! */

namespace Dune {
  namespace Solvers {
    template<class MatrixType, class Vector, int k, int overlapSize=1>
      class FaceBasedSchwarzSmoother : public LinearIterationStep<MatrixType, Vector, Dune::Solvers::DefaultBitVector_t<Vector> >
    {


      // determine the relative position of two elements through their indices (dangerous?)
      template<class M>
        int getPosition(M m) {
          double EPSILON = 1e-15;
          if (std::abs(m[k+1][2*k+1])   > EPSILON)  return 0; // left
          if (std::abs(m[2*k+1][k+1])   > EPSILON)  return 1; // right
          if (std::abs(m[k*(k+1)+1][1]) > EPSILON)  return 2; // top 
          if (std::abs(m[1][k*(k+1)+1]) > EPSILON)  return 3; // below 

          DUNE_THROW(Dune::Exception, "failed to analyze adjacency pattern"); return -1; 
        }

      public:
      void preprocess() {
        auto& mat = *this->mat_;
        positions.resize(mat.N());
        for (std::size_t i=0; i< mat.N(); i++) {
          auto& entry = positions[i];
          entry = {-1,-1,-1,-1};
          auto cIt = mat[i].begin();
          const auto cEnd = mat[i].end();
          for (; cIt!=cEnd; ++cIt) {
            auto j = cIt.index();
            if (i==j) continue;
            auto& block = *cIt;
            // determine Position of the adjacent element
            entry[getPosition(block)]=j;
          }
        }
      }

      /* first very basic and stupid prototype */
      void iterate() {
        const int overlap = std::min(overlapSize, k);
        auto& mat = *this->mat_;
        //auto& x = *this->x_;
        auto x = *this->x_;
        //auto& rhs = *this->rhs_;
        auto rhs = *this->rhs_;
        Dune::MatrixVector::subtractProduct(rhs, mat, x);
        x=0;
        auto gs = Dune::Solvers::BlockGSStepFactory<MatrixType, Vector, Dune::Solvers::DefaultBitVector_t<Vector> >::create(Dune::Solvers::BlockGS::LocalSolvers::gs(0,0));
        // first take edges that are in y direction (left, right), then in x-direction (below, top)
        for (int i=0; i<mat.N(); i++) {
          for (std::size_t d = 0; d<2; d++) {
            auto j = positions[i][d];
            // Don't do anything on border edges
            if (j==-1) {
              continue;
            }
            else if (i<j)
              continue; // each edge should only be handled once

            std::vector<int> idx {i,j};

            int iUp, iDown, jUp, jDown;
            if (overlap>0) {

            // Add indices of the top and below elements (if there are any)
            if (positions[i][2] != -1) {
              idx.emplace_back(positions[i][2]);
              iUp = idx.size()-1;
            }
            if (positions[j][2] != -1) {
              idx.emplace_back(positions[j][2]);
              jUp = idx.size()-1;
            }
            if (positions[i][3] != -1) {
              idx.emplace_back(positions[i][3]);
              iDown = idx.size()-1;
            }
            if (positions[j][3] != -1) {
              idx.emplace_back(positions[j][3]);
              jDown = idx.size()-1;
            }
            }

            auto subMatrix = Dune::MatrixVector::SubMatrix<MatrixType>(mat, idx, idx);
            auto subX= Dune::MatrixVector::SubVector<Vector, decltype(idx)>(x, idx);
            auto subRHS = Dune::MatrixVector::SubVector<Vector, decltype(idx)>(rhs, idx);

            using Ignore = Dune::Solvers::DefaultBitVector_t<Vector>;
            Ignore ig(subMatrix.matrix().N(), false);
            // we have to ignore a couple of edge nodes. This is some index voodoo again

            // first, ignore the nodes on the opposite edges
            {
              int r = k;
              int l = 0;
              while (r < (k+1)*(k+1)) {
                ig[0][d==0 ? r : l]=true;
                ig[1][d==0 ? l : r]=true;
                if (overlap>0) {
                if (positions[i][2] != -1) ig[iUp][d==0 ? r : l]=true;
                if (positions[i][3] != -1) ig[iDown][d==0 ? r : l]=true;
                if (positions[j][2] != -1) ig[jUp][d==0 ? l : r]=true;
                if (positions[j][3] != -1) ig[jDown][d==0 ? l : r]=true;
                }
                l+=k+1;
                r+=k+1;
              }

              // ignore the zone above and below the overlap
              if (overlap>0) {
              if (positions[i][3] != -1) {
                for (int q = overlap*(k+1); q < (k+1)*(k+1); q++)
                  ig[iDown][q]=true;
              }
              if (positions[j][3] != -1) {
                for (int q = overlap*(k+1); q < (k+1)*(k+1); q++)
                  ig[jDown][q]=true;
              }
              if (positions[i][2] != -1) {
                for (int q = (k+1)*(k+1)-1; q >(k+1-overlap)*(k+1)-1; q--)
                  ig[iUp][q]=true;
              }
              if (positions[j][2] != -1) {
                for (int q = (k+1)*(k+1)-1; q >(k+1-overlap)*(k+1)-1; q--)
                  ig[jUp][q]=true;
              }
              }
            }

            // "Solve" the subsystem via Gauss Seidel
            //truncateMatrix(subMatrix.matrix(), ig);
            truncateMatrix(subMatrix.matrix(), subX.vector(), subRHS.vector(), ig);
            gs.setProblem(subMatrix.matrix(), subX.vector(), subRHS.vector());
            //gs.setIgnore(ig);
            for (unsigned z = 0; z < iterations; z++)
              gs.iterate();
            auto& sx = subX.vector();

            // copy the new values into the global vector
            x[i]=sx[0];
            x[j]=sx[1];
            //(*this->x_)[i]+=x[i];
            //(*this->x_)[j]+=x[j];
            //x[i]=x[j]=0;
            if(overlap>0) {
            if (positions[i][2] != -1) x[positions[i][2]]=sx[iUp];
            if (positions[i][3] != -1) x[positions[i][3]]=sx[iDown];
            if (positions[j][2] != -1) x[positions[j][2]]=sx[jUp];
            if (positions[j][3] != -1) x[positions[j][3]]=sx[jDown];
            }
            //Dune::MatrixVector::subtractProduct(rhs, mat, x);
            x*=0.25;
            *this->x_ += x;
            x=0;
            //copyVector(x[i],sx[0],ig[0]);
            //copyVector(x[j],sx[1],ig[1]);
            //if (positions[i][2] != -1) copyVector(x[positions[i][2]],sx[iUp], ig[iUp]);
            //if (positions[i][3] != -1) copyVector(x[positions[i][3]],sx[iDown], ig[iDown]);
            //if (positions[j][2] != -1) copyVector(x[positions[j][2]],sx[jUp], ig[jUp]);
            //if (positions[j][3] != -1) copyVector(x[positions[j][3]],sx[jDown], ig[jDown]);
          }
        }

        /* update */

        // now the edges in x direction (top, below). This is a lot of code written twice...
        for (int i=0; i<mat.N(); i++) {
          for (std::size_t d = 2; d<4; d++) {
            auto j = positions[i][d];
            if (j==-1) {
              continue;
            }
            else if (i<j)
              continue; // each edge should only be handled once

            std::vector<int> idx {i,j};

            int iLeft, iRight, jLeft, jRight;
            if(overlap>0) {

            // Add indices of the left and right elements (if there are any)
            if (positions[i][0] != -1) {
              idx.emplace_back(positions[i][0]);
              iLeft = idx.size()-1;
            }
            if (positions[j][0] != -1) {
              idx.emplace_back(positions[j][0]);
              jLeft = idx.size()-1;
            }
            if (positions[i][1] != -1) {
              idx.emplace_back(positions[i][1]);
              iRight= idx.size()-1;
            }
            if (positions[j][1] != -1) {
              idx.emplace_back(positions[j][1]);
              jRight = idx.size()-1;
            }
            }

            auto subMatrix = Dune::MatrixVector::SubMatrix<MatrixType>(mat, idx, idx);
            auto subX= Dune::MatrixVector::SubVector<Vector, decltype(idx)>(x, idx);
            auto subRHS = Dune::MatrixVector::SubVector<Vector, decltype(idx)>(rhs, idx);

            using Ignore = Dune::Solvers::DefaultBitVector_t<Vector>;
            Ignore ig(subMatrix.matrix().N(), false);
            // we have to ignore a couple of edge nodes. This is some index voodoo again

            // first, ignore the nodes on the opposite edges
            {
              int t =k*(k+1);
              for (int b =0; b<=k; b++) {
                t+=b;
                ig[0][d==2 ? b : t]=true;
                ig[1][d==2 ? t : b]=true;
                if (overlap>0) {
                if (positions[i][0] != -1) ig[iLeft][d==2 ? b : t]=true;
                if (positions[i][1] != -1) ig[iRight][d==2 ? b : t]=true;
                if (positions[j][0] != -1) ig[jLeft][d==2 ? t : b]=true;
                if (positions[j][1] != -1) ig[jRight][d==2 ? t : b]=true;
                }
              }

              // ignore the zone left and right of the overlap
              if (overlap>0) {
              if (positions[i][0] != -1) {
                for (int q = 0; q <= k-overlap; q++) {
                  int qq = q;
                  while (qq < (k+1)*(k+1)) {
                    ig[iLeft][qq]=true;
                    qq+=k+1;
                  }
                }
              }
              if (positions[j][0] != -1) {
                for (int q = 0; q <= k-overlap; q++) {
                  int qq = q;
                  while (qq < (k+1)*(k+1)) {
                    ig[jLeft][qq]=true;
                    qq+=k+1;
                  }
                }
              }
              if (positions[i][1] != -1) {
                for (int q = overlap; q <= k; q++) {
                  int qq = q;
                  while (qq < (k+1)*(k+1)) {
                    ig[iRight][qq]=true;
                    qq+=k+1;
                  }
                }
              }
              if (positions[j][1] != -1) {
                for (int q = overlap; q <= k; q++) {
                  int qq = q;
                  while (qq < (k+1)*(k+1)) {
                    ig[jRight][qq]=true;
                    qq+=k+1;
                  }
                }
              }
              }
            }

            // Solve again
            truncateMatrix(subMatrix.matrix(), subX.vector(), subRHS.vector(), ig);
            gs.setProblem(subMatrix.matrix(), subX.vector(), subRHS.vector());
            //gs.setIgnore(ig);

            for (unsigned z = 0; z < iterations; z++)
              gs.iterate();
            auto& sx = subX.vector();
            x[i]=sx[0];
            x[j]=sx[1];
            //(*this->x_)[i]+=x[i];
            //(*this->x_)[j]+=x[j];
            //x[i]=x[j]=0;
            if (overlap>0) {
            if (positions[i][0] != -1) x[positions[i][0]]=sx[iLeft];
            if (positions[i][1] != -1) x[positions[i][1]]=sx[iRight];
            if (positions[j][0] != -1) x[positions[j][0]]=sx[jLeft];
            if (positions[j][1] != -1) x[positions[j][1]]=sx[jRight];
            }
            //Dune::MatrixVector::subtractProduct(rhs, mat, x);
            x*=0.25;
            *this->x_ += x;
            x=0;
            //copyVector(x[i],sx[0],ig[0]);
            //copyVector(x[j],sx[1],ig[1]);
            //if (positions[i][0] != -1) copyVector(x[positions[i][0]],sx[iLeft], ig[iLeft]);
            //if (positions[i][1] != -1) copyVector(x[positions[i][1]],sx[iRight], ig[iRight]);
            //if (positions[j][0] != -1) copyVector(x[positions[j][0]],sx[jLeft], ig[jLeft]);
            //if (positions[j][1] != -1) copyVector(x[positions[j][1]],sx[jRight], ig[jRight]);
          }
        }
        //*this->x_+=x;
      }
      int iterations = 3;

      private:

      template<class M, class V, class I>
      void truncateMatrix(M& m, const V& x, V& r, const I& ignore) {
        for (std::size_t i=0; i< m.N(); i++) {
          auto cIt = m[i].begin();
          auto cEnd = m[i].end();
          for (; cIt!=cEnd; ++cIt) {
            auto j = cIt.index();
            for (std::size_t m=0; m<cIt->N(); ++m) {
              if (!ignore[i][m])
                continue;
              r[i][m]=x[i][m]; // put the current value into the rhs such that the value of x wont be changed
              for (std::size_t l=0; l<cIt->M(); ++l) 
                (*cIt)[m][l]=(*cIt)[l][m]= (i==j and m==l);
            }
          }
        }
      }
      template<class V, class I>
      void copyVector(V&v,const V& x,  const I& ignore) {
        for (std::size_t i=0; i< v.size(); i++)
          if (not ignore[i]) v[i]=x[i];
      }
      public:
      std::vector<std::array<int, 4> > positions;
    };
  }
}
#endif
