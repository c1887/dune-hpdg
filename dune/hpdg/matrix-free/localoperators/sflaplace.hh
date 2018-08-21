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
   * Right now, this is in an extremely messy state and also
   * severly inefficient!
   */
  template<class V, class GV, class Basis>
  class SumFactLaplaceperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;
    using Field = typename V::field_type;

    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<double, 1, 1>>;
    using LM = LocalMatrix; // TODO Remove this. Nur fürs schnellere schreiben

    static constexpr int dim = GV::dimension;

    public:

      SumFactLaplaceperator(const Basis& b) :
        basis_(b),
        localView_(basis_.localView()) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for (auto& entry: localVector_)
          entry=0;
      }


      void compute() {

        const auto& fe = localView_.tree().finiteElement();

        // compute needed order:
        auto degree = fe.localBasis().order() +1; // if order p, we want p+1 nodes
        auto order = 2*degree - 3;
        // get quadrature rule:
        auto rule = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto);

        // sort quad points (they're also ordered for the basis)
        std::sort(rule.begin(), rule.end(), [](auto&& a, auto&& b) {
            return a.position() < b.position();
            });

        const auto& geometry = localView_.element().geometry();

        // we need the coefficients at every quadrature point. We extract and order them once:
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        const auto* coeffs = &(inputBackend(localView_.index(0))); // using DG structure here

        assert(dim==2); // TODO. Implement other cases

        using FV = Dune::FieldVector<double, dim>;

        // returns \partial_direction \hat{u}(\xi_{i0, i1})
        auto localDev =[&](const auto& i0, const auto& i1, const auto& direction) {
          double res =0;
          auto pos = direction == 0 ? i0 : i1; // the relevant index, the other one won't be used, see below.

          // here, we use the fact that \phi_alpha(x_beta) = alpha==beta, therefore we only need
          // one for-loop.
          for(std::size_t i = 0; i < degree; i++) {
            auto idx = flatindex({{ direction == 0 ? i : i0, direction == 1 ? i: i1}}, degree-1);
            res+= coeffs[idx]*lagrangePrime(rule[pos].position(), i, rule);
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

        for (size_t k = 0; k < localView_.size(); k++) {
          double out=0.;
          auto mm = multiindex(k, degree-1);
          for(std::size_t r = 0; r < dim; r++) {
            for(std::size_t i0 = 0; i0 < rule.size(); i0++) {
              if (r!=0 and mm[0]!=i0)
                continue;
              for(std::size_t i1 = 0; i1 < rule.size(); i1++) {
                if (r!=1 and mm[1]!=i1)
                  continue;

                // get jac: this should actually be always the same for a aligned grid.
                FV pos{rule[i0].position(), rule[i1].position()};
                const auto& jac = geometry.jacobianInverseTransposed(pos);
                auto gamma = geometry.integrationElement(pos);
                auto weight = rule[i0].weight()*rule[i1].weight();

                // compute x factor
                auto x = computeX(i0,i1, jac)[r]*gamma*weight;
                out+= computeFaktor(rule, 0, r, mm[0], i0)*computeFaktor(rule, 1, r, mm[1], i1)*x;

              }
            }
          }
        localVector_[k] = out;
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
      inline double lagrange(const X& x, size_t i, const Q& quad) const {
       double result = 1.;
        for (size_t j=0; j<quad.size(); j++)
          if (j!=i) result *= (x-quad.at(j).position())/(quad.at(i).position()-quad.at(j).position());
        return result;
      }

      template<class X, class Q>
      inline double lagrangePrime(const X& x, size_t i, const Q& quad) const {
        double result = 0.;

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            double prod= 1.0/(quad.at(i).position()-quad.at(j).position());
            for (size_t l=0; l<quad.size(); l++)
              if (l!=i && l!=j)
                prod *= (x-quad.at(l).position())/(quad.at(i).position()-quad.at(l).position());
            result += prod;
          }
        return result;
      }

      template<class Q>
      double computeFaktor(const Q& quad, size_t q, size_t r, size_t alpha, size_t beta) const {
        if (q!=r)
          return alpha==beta ? 1. : 0.; // we use the underlying quadrature formula for the quadrature here
        else
          return lagrangePrime(quad[beta].position(), alpha, quad);
      }


      const Basis& basis_;
      LV localView_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
  };
}
}
}
