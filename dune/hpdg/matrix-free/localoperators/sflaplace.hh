// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/geometry/quadraturerules.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
#include <dune/hpdg/common/mmmatrix.hh>
#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** computes (∇φ_j, ∇u) for all j on a given element.
   * This works via sumfactorization and is only valid if
   * Basis is a Dune::HPDG based Gauss-Lobatto DG-Basis!
   * In particular, it relies on several implementation details
   *
   * TODO:
   *
   * - Generalize to dimensons != 2
   *
   * - Right now, this is in an extremely messy state and also
   * severly inefficient!
   */
  template<class V, class GV, class Basis>
  class SumFactLaplaceOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;

    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<double, 1, 1>>;

    static constexpr int dim = GV::dimension;
    static_assert(dim==2, "Sumfactorized Laplace only for dim=2 currently");

    public:

      SumFactLaplaceOperator(const Basis& b) :
        basis_(b),
        localView_(basis_.localView()) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for (auto& entry: localVector_)
          entry=0;

        // get order:
        localDegree_ = basis_.preBasis().degree(e);

        // check if order is already in cache:
        {
          auto it = cache_.find(localDegree_);
          if (it != cache_.end()) {
            matrixPair_ = &(it->second);
            rule_ = &(rules_[localDegree_]);
            return;
          }
        }

        int order = 2*localDegree_ - 1;
        // get quadrature rule:
        auto gauss_lobatto = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        rules_[localDegree_] = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order+2, Dune::QuadratureType::GaussLobatto); // TODO Welche Ordnung braucht man wirklich?

        rule_ = &(rules_[localDegree_]);

        // sort quad points (they're also ordered for the basis)
        std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });


        // save all derivatives of 1-d basis functions at the quad points, i.e.
        // matrixPair_ij = l_i' (xi_j)
        LocalMatrix lp(localDegree_+1, rule_->size());
        // same for the function values
        LocalMatrix l(localDegree_+1, rule_->size());
        for (std::size_t i = 0; i < lp.N(); i++) {
          for (std::size_t j = 0; j < lp.M(); j++) {
            lp[i][j]=lagrangePrime((*rule_)[j].position(),i, gauss_lobatto);
            l[i][j]=lagrange((*rule_)[j].position(),i, gauss_lobatto);
          }
        }
        cache_[localDegree_]=std::array<LocalMatrix, 2>{{std::move(l), std::move(lp)}};
        matrixPair_ = &(cache_[localDegree_]);

      }


      void compute() {
        // compute needed order:
        //auto degree = localDegree_ +1; // if order p, we want p+1 nodes


        const auto& geometry = localView_.element().geometry();

        // TODO: If grid is very simple, one can compute jac and gamma once here.

        // find the first entry in input vector. Because we know DG indices are contiguous and
        // the local 0 idx is also the lowest global index, we can perform this little
        // hack to circumvent using a buffer and several calls do localView.index(foo).
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        const auto* coeffs = &(inputBackend(localView_.index(0))); // using DG structure here

        using FV = Dune::FieldVector<double, dim>;

        // precompute the derivatives of the local function at each (tensor-product) quad point
        auto dx_u = Dune::HPDG::BtUL((*matrixPair_)[0], coeffs, (*matrixPair_)[1]); // matrix contains \partial_x u(xi_{i0, i1})
        auto dy_u = Dune::HPDG::BtUL((*matrixPair_)[1], coeffs, (*matrixPair_)[0]); // matrix contains \partial_y u(xi_{i0, i1})

        // i0 and i1 are indices wrt to the RULE, not the basis
        auto computeX= [&](auto i0, auto i1, const auto& jacT) {
          FV gradU {dx_u[i0][i1], dy_u[i0][i1]};
          FV dummy;
          jacT.mv(gradU, dummy);
          jacT.mtv(dummy, gradU);
          return gradU;
        };

        auto i_length = rule_->size(); // number of quadrature points

        std::array<Dune::Matrix<Dune::FieldMatrix<double,1,1>>, dim> innerValues; // TODO: document exactly what's in here
        for (auto& iv: innerValues)
          iv.setSize(i_length, i_length);

        // pre-compute inner values
        for(std::size_t i1 = 0; i1 < i_length; i1++) {
          FV pos{0, (*rule_).at(i1).position()};
          for(std::size_t i0 = 0; i0 < i_length; i0++) {
            pos[0]=(*rule_).at(i0).position();
            // compute jacobian
            const auto& jac = geometry.jacobianInverseTransposed(pos);
            auto gamma = geometry.integrationElement(pos);

            auto x_values = computeX(i0,i1,jac);
            x_values *= gamma*(*rule_)[i0].weight()*(*rule_)[i1].weight();
            for (size_t r=0; r<dim; r++) {
              innerValues[r][i0][i1]=x_values[r];
            }
          }
        }

        // compute all the integrals for the given dimension at once
        for(size_t r=0; r <dim; r++) {
          const auto& matrix0 = (*matrixPair_)[r==0 ? 1 : 0];
          const auto& matrix1 = (*matrixPair_)[r==1 ? 1 : 0];

          Dune::HPDG::CplusAXtBt(matrix1, innerValues[r], matrix0, localVector_.data());

          // This here is a nice use of the common DG hack. However, be aware that this would
          // probably not be thread-safe!
          //auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
          //auto* out = &(outputBackend(localView_.index(0)));
          //Dune::HPDG::CplusAXtBt(matrix1, innerValues[r], matrix0, out);
        }
      }

      void write(double factor) {
        // if the DG hack was used in compute, return here:
        // return;

        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;
        if (factor == 0.0)
          return;

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          auto& rowEntry = outputBackend(localView_.index(localRow));
          rowEntry += localVector_[localRow];
        }
      }

    private:

      template<class X, class Q>
      inline double lagrangePrime(const X& x, size_t i, const Q& quad) const {
        double result = 0.;

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            double prod= 1.0/(quad[i].position()-quad[j].position());
            for (size_t l=0; l<quad.size(); l++)
              if (l!=i && l!=j)
                prod *= (x-quad[l].position())/(quad[i].position()-quad[l].position());
            result += prod;
          }
        return result;
      }

      template<class X, class Q>
      inline double lagrange(const X& x, size_t i, const Q& quad) const {
        double result = 1.;

        auto xi= quad[i].position();

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            result*= (x-quad[j].position())/(xi-quad[j].position());
          }
        return result;
      }

      const Basis& basis_;
      LV localView_;
      std::map<size_t, std::array<LocalMatrix, 2>> cache_; // contains all lagrange Polynomials at all quadrature points and all derivatives of said polynomials at all quad points
      std::map<size_t, Dune::QuadratureRule<typename GV::Grid::ctype, 1>> rules_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      int localDegree_;
      const typename decltype(cache_)::mapped_type* matrixPair_; // Current matrix pair
      Dune::QuadratureRule<typename GV::Grid::ctype,1>* rule_;
  };
}
}
}
