// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <dune/common/fmatrix.hh>
#include <dune/common/boundschecking.hh>
#include <dune/istl/matrix.hh>
#include <dune/geometry/quadraturerules.hh>

namespace Dune {
namespace HPDG {
namespace GaussLobatto {
  class Values : public Dune::Matrix<FieldMatrix<double, 1, 1>> {
    using Base = Dune::Matrix<FieldMatrix<double, 1, 1>>;

    public:
    template<typename Q>
    Values(int basisDegree, const Q& rule) {
      compute(basisDegree, rule);
    }

    Values() {
      Base();
    }
    /** Compute (and store) the function values of the 1D
     * Gauss-Lobatto Lagrange Polynomials at the quadrature points
     * of a given quadrature rule.
     */
    template<typename Q>
    void compute(int degree, const Q& nodes) {
      this->setSize(degree+1, nodes.size());
      int order = 2*degree - 1;
      auto gauss_lobatto = Dune::QuadratureRules<double,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
      // sort quad points (they're also ordered for the basis)
      std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
        return a.position() < b.position(); });

      for (int i = 0; i < degree+1; i++) {
        for (std::size_t j = 0; j < nodes.size(); j++) {
          (*this)[i][j]=lagrange(nodes[j].position(),i, gauss_lobatto);
        }
      }
    }

    private:
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
  };

  class Derivatives : public Dune::Matrix<FieldMatrix<double, 1, 1>> {
    using Base = Dune::Matrix<FieldMatrix<double, 1, 1>>;

    public:
    template<typename Q>
    Derivatives(int basisDegree, const Q& rule) {
      compute(basisDegree, rule);
    }

    Derivatives() {
      Base();
    }
    /** Compute (and store) the derivatives of the 1D
     * Gauss-Lobatto Lagrange Polynomials at the quadrature points
     * of a given quadrature rule.
     */
    template<typename Q>
    void compute(int degree, const Q& nodes) {
      this->setSize(degree+1, nodes.size());
      int order = 2*degree - 1;
      auto gauss_lobatto = Dune::QuadratureRules<double,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
      // sort quad points (they're also ordered for the basis)
      std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
        return a.position() < b.position(); });

      for (int i = 0; i < degree+1; i++) {
        for (std::size_t j = 0; j < nodes.size(); j++) {
          (*this)[i][j]=lagrangePrime(nodes[j].position(),i, gauss_lobatto);
        }
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
  };

  struct ValuesAndDerivatives {
    template<class Q>
    ValuesAndDerivatives(int basisDegree, const Q& rule):
      values(basisDegree, rule),
      derivatives(basisDegree, rule)
    {}

    ValuesAndDerivatives() = default;

    Values values;
    Derivatives derivatives;
  };

}}}
