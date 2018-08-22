// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/localfunctions/lagrange/qk/qklocalcoefficients.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
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
    using LM = LocalMatrix; // TODO Remove this. Nur fürs schnellere schreiben

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
            lp_ = &(it->second);
            rule_ = &(rules_[localDegree_]);
            return;
          }
        }

        auto order = 2*localDegree_ - 1;
        // get quadrature rule:
        rules_[localDegree_] = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto);

        rule_ = &(rules_[localDegree_]);

        // sort quad points (they're also ordered for the basis)
        std::sort(rule_->begin(), rule_->end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });


        // save all derivatives of 1-d basis functions at the quad points, i.e.
        // lp_ij = l_i' (xi_j)
        LocalMatrix lp(localDegree_+1, localDegree_+1);
        for (std::size_t i = 0; i < lp.N(); i++) {
          for (std::size_t j = 0; j < lp.M(); j++) {
            lp[i][j]=lagrangePrime((*rule_)[i].position(),j, *rule_);
          }
        }
        cache_[localDegree_]=std::move(lp);
        lp_ = &(cache_[localDegree_]);

      }


      void compute() {
        // compute needed order:
        auto degree = localDegree_ +1; // if order p, we want p+1 nodes


        const auto& geometry = localView_.element().geometry();

        // TODO: If grid is very simple, one can compute jac and gamma once here.

        // find the first entry in input vector. Because we know DG indices are contiguous and
        // the local 0 idx is also the lowest global index, we can perform this little
        // hack to circumvent using a buffer and several calls do localView.index(foo).
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        const auto* coeffs = &(inputBackend(localView_.index(0))); // using DG structure here

        using FV = Dune::FieldVector<double, dim>;

        // returns \partial_direction \hat{u}(\xi_{i0, i1})
        auto localDev =[&](const auto& i0, const auto& i1, const auto& direction) {
          double res =0;
          auto pos = direction == 0 ? i0 : i1; // the relevant index, the other one won't be used, see below.

          // here, we use the fact that \phi_alpha(x_beta) = alpha==beta, therefore we only need
          // one for-loop.
          for(std::size_t i = 0; i < degree; i++) {
            auto idx = flatindex({{ direction == 0 ? i : i0, direction == 1 ? i: i1}}, degree-1);
            res+= coeffs[idx]*((*lp_)[pos][i]);
          }
          return res;
        };

        auto computeX= [&](auto i0, auto i1, const auto& jacT) {
          FV p;
          p[0]=localDev(i0, i1, 0);
          p[1]=localDev(i0, i1, 1);
          FV dummy;
          jacT.mv(p, dummy);
          jacT.mtv(dummy, p);
          return p;
        };

        LocalMatrix X(lp_->N(), lp_->M());
        for(size_t i0=0; i0<lp_->N(); i0++) {
          for(size_t i1=0; i1<lp_->M(); i1++) {
            FV pos{(*rule_).at(i0).position(), (*rule_).at(i1).position()};
            const auto& jac = geometry.jacobianInverseTransposed(pos);
            auto gamma = geometry.integrationElement(pos);

            auto weight = (*rule_)[i0].weight()*(*rule_)[i1].weight();

            auto x = computeX(i0,i1, jac);
            x*=weight;
            x*=gamma;
            X[i0][i1]=x[0]+x[1];
          }
        }

        auto prod = *lp_ * X;

        for(size_t i0=0; i0<prod.N(); i0++) {
          for(size_t i1=0; i1<prod.M(); i1++) {
            localVector_[flatindex({{i0, i1}}, localDegree_)]=prod[i0][i1];
          }
        }
      }

      void write(double factor) {

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
      std::vector<unsigned int> multiindex (unsigned int i, unsigned int k) const
      {
        std::vector<unsigned int> alpha(k);
        for (int j=0; j<dim; j++)
        {
          alpha[j] = i % (k+1);
          i = i/(k+1);
        }
        return alpha;
      }

      unsigned int flatindex(std::array<size_t, 2> multi, unsigned int k) const {
        return multi[0] + multi[1]*(k+1);
      }

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

      const Basis& basis_;
      LV localView_;
      std::map<size_t, LocalMatrix> cache_;
      std::map<size_t, Dune::QuadratureRule<typename GV::Grid::ctype, 1>> rules_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      size_t localDegree_;
      const LocalMatrix* lp_;
      Dune::QuadratureRule<typename GV::Grid::ctype,1>* rule_;
  };
}
}
}
