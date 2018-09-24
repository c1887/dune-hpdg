// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/localfunctions/lagrange/qk/qklocalcoefficients.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
#include <dune/hpdg/common/mmmatrix.hh>
#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  template<class V, class GV, class Basis>
  class SumFactMassOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using Field = typename V::field_type;
    using IndexPair = std::array<int, 2>; // The first index describes the degree of the basis function and the second one describes _the order_ of the quadrature rule!
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<double, 1, 1>>;
    using BV = Dune::BlockVector<Dune::FieldVector<double,1>>;

    static constexpr int dim = GV::dimension;
    static_assert(dim==2, "Sumfactorized Mass only for dim=2 currently");
    using FV = Dune::FieldVector<double, dim>;

    public:

      SumFactMassOperator(const Basis& b) :
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

        int order = 2*localDegree_ - 1;

        // check if order is already in cache:
        {
          auto it = cache_.find(localDegree_);
          if (it != cache_.end()) {
            matrix_ = &(it->second);
            rule_ = &(rules_[order]);
            return;
          }
        }

        // compute the Lagrange polynomials and derivatives at the quad points
        setupMatrixPair(localDegree_);

        matrix_ = &(cache_[localDegree_]);

        rule_ = &(rules_[order]);

      }

      void compute() {
        computeBulk();
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

      void computeBulk() {
        const auto& geometry = localView_.element().geometry();

        // find the first entry in input vector. Because we know DG indices are contiguous and
        // the local 0 idx is also the lowest global index, we can perform this little
        // hack to circumvent using a buffer and several calls do localView.index(foo).
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        const auto* coeffs = &(inputBackend(localView_.index(0))); // using DG structure here

        auto lu = Dune::HPDG::BtUL(*matrix_, coeffs, *matrix_);

        // pre-compute inner values
        auto i_length = rule_->size();
        for(std::size_t i0 = 0; i0 < i_length; i0++) {
          FV pos{(*rule_).at(i0).position(), 0};
          for(std::size_t i1 = 0; i1 < i_length; i1++) {
            pos[1]=(*rule_).at(i1).position();
            // compute jacobian
            auto gamma = geometry.integrationElement(pos);
            lu[i0][i1]*=gamma*(*rule_)[i0].weight()*(*rule_)[i1].weight();
          }
        }

        Dune::HPDG::CplusAXtBt(*matrix_, lu, *matrix_, localVector_.data());

      }

      void setupMatrixPair(int idx) {

        const auto& basis_degree = idx;
        // get quadrature rule:
        int order = 2*basis_degree -1;
        auto gauss_lobatto = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        rules_[order] = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order+1, Dune::QuadratureType::GaussLobatto); // TODO Welche Ordnung braucht man wirklich?
        auto& rule = rules_[order];

        // sort node points (they're also ordered for the basis)
        std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        // sort quadrature points
        std::sort(rule.begin(), rule.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        assert(gauss_lobatto.size() == basis_degree+1);

        // save all derivatives of 1-d basis functions at the quad points, i.e.
        LocalMatrix l(basis_degree+1, rule.size());
        for (std::size_t i = 0; i < l.N(); i++) {
          for (std::size_t j = 0; j < l.M(); j++) {
            l[i][j]=lagrange(rule[j].position(),i, gauss_lobatto);
          }
        }
        cache_[idx]=std::move(l);
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

      // members:
      const Basis& basis_;
      LV localView_;
      std::map<int, LocalMatrix> cache_; // contains all lagrange Polynomials at all quadrature points and all derivatives of said polynomials at all quad points
      std::map<int, Dune::QuadratureRule<typename GV::Grid::ctype, 1>> rules_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      int localDegree_;
      const typename decltype(cache_)::mapped_type* matrix_; // Current matrix pair
      Dune::QuadratureRule<typename GV::Grid::ctype,1>* rule_;
  };
}
}
}
