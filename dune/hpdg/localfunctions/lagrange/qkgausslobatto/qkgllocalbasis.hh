// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_HPDG_LOCALFUNCTIONS_GAUSS_LOBATTO_QKLOCALBASIS_HH
#define DUNE_HPDG_LOCALFUNCTIONS_GAUSS_LOBATTO_QKLOCALBASIS_HH

#include <numeric>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/power.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/common/localbasis.hh>
#include <dune/localfunctions/common/localfiniteelementtraits.hh>


namespace Dune
{
  /**@ingroup LocalBasisImplementation
     \brief Lagrange shape functions of order k on the reference cube using
     Gauss-Lobatto nodes.

     Also known as \f$Q^k\f$.

     \tparam D Type to represent the field in the domain.
     \tparam R Type to represent the field in the range.
     \tparam k Polynomial degree
     \tparam d Dimension of the cube

     \nosubgrouping
   */
  template<class D, class R, int k, int d>
  class QkGaussLobattoLocalBasis
  {
    enum { n = StaticPower<k+1,d>::power };

    // ith Lagrange polynomial of degree k in one dimension
    static R p (int i, D x)
    {
      GaussLobattoQuadratureRule1D<double> gaussLobattoNodes_(k+1);
      // Not so easy to find the right order :/
      while (gaussLobattoNodes_.size() != k+1)
          gaussLobattoNodes_ = GaussLobattoQuadratureRule1D<double>(gaussLobattoNodes_.order()+1);

      std::vector<double> xx(k+1); // Nodes
      for (size_t n =0; n<=k; n++)
          xx[n] = gaussLobattoNodes_[n].position();

      // GL nodes are sometimes not ordered what screws up the indexing
      std::sort(xx.begin(), xx.end());

      R result(1.0);
      for (int j=0; j<=k; j++)
        if (j!=i) result *= (x-xx[j])/(xx[i]-xx[j]);
      return result;
    }

    // derivative of ith Lagrange polynomial of degree k in one dimension
    static R dp (int i, D x)
    {
      GaussLobattoQuadratureRule1D<double> gaussLobattoNodes_(k+1);

      // Not so easy to find the right order :/
      while (gaussLobattoNodes_.size() != k+1)
          gaussLobattoNodes_ = GaussLobattoQuadratureRule1D<double>(gaussLobattoNodes_.order()+1);

      std::vector<double> xx(k+1); // Nodes
      for (size_t n =0; n<=k; n++)
          xx[n] = gaussLobattoNodes_[n].position();

      // GL nodes are sometimes not ordered what screws up the indexing
      std::sort(xx.begin(), xx.end());

      R result(0.0);

      for (int j=0; j<=k; j++)
        if (j!=i)
        {
          R prod( 1.0/(xx[i]-xx[j]) );
          for (int l=0; l<=k; l++)
            if (l!=i && l!=j)
              prod *= (x-xx[l])/(xx[i]-xx[l]);
          result += prod;
        }
      return result;
    }

    // Return i as a d-digit number in the (k+1)-nary system
    static Dune::FieldVector<int,d> multiindex (int i)
    {
      Dune::FieldVector<int,d> alpha;
      for (int j=0; j<d; j++)
      {
        alpha[j] = i % (k+1);
        i = i/(k+1);
      }
      return alpha;
    }

  public:
    typedef LocalBasisTraits<D,d,Dune::FieldVector<D,d>,R,1,Dune::FieldVector<R,1>,Dune::FieldMatrix<R,1,d>, 1> Traits;

    //! \brief number of shape functions
    unsigned int size () const
    {
      return StaticPower<k+1,d>::power;
    }

    //! \brief Evaluate all shape functions
    inline void evaluateFunction (const typename Traits::DomainType& in,
                                  std::vector<typename Traits::RangeType>& out) const
    {
      out.resize(size());
      for (size_t i=0; i<size(); i++)
      {
        // convert index i to multiindex
        Dune::FieldVector<int,d> alpha(multiindex(i));

        // initialize product
        out[i] = 1.0;

        // dimension by dimension
        for (int j=0; j<d; j++)
          out[i] *= p(alpha[j],in[j]);
      }
    }

    /** \brief Evaluate Jacobian of all shape functions
     * \param in position where to evaluate
     * \param out The return value
     */
    inline void
    evaluateJacobian (const typename Traits::DomainType& in,
                      std::vector<typename Traits::JacobianType>& out) const
    {
      out.resize(size());

      // Loop over all shape functions
      for (size_t i=0; i<size(); i++)
      {
        // convert index i to multiindex
        Dune::FieldVector<int,d> alpha(multiindex(i));

        // Loop over all coordinate directions
        for (int j=0; j<d; j++)
        {
          // Initialize: the overall expression is a product
          // if j-th bit of i is set to -1, else 1
          out[i][0][j] = dp(alpha[j],in[j]);

          // rest of the product
          for (int l=0; l<d; l++)
            if (l!=j)
              out[i][0][j] *= p(alpha[l],in[l]);
        }
      }
    }

    /** \brief Evaluate partial derivatives of any order of all shape functions
     * \param order Order of the partial derivatives, in the classic multi-index notation
     * \param in Position where to evaluate the derivatives
     * \param[out] out Return value: the desired partial derivatives
     */
    inline void partial(const std::array<unsigned int,d>& order,
                        const typename Traits::DomainType& in,
                        std::vector<typename Traits::RangeType>& out) const
    {
      auto totalOrder = std::accumulate(order.begin(), order.end(), 0);

      switch (totalOrder)
      {
        case 0:
          evaluateFunction(in,out);
          break;
        case 1:
        {
          out.resize(size());

          // Loop over all shape functions
          for (size_t i=0; i<size(); i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex(i));

            // Initialize: the overall expression is a product
            out[i][0] = 1.0;

            // rest of the product
            for (std::size_t l=0; l<d; l++)
              out[i][0] *= (order[l]) ? dp(alpha[l],in[l]) : p(alpha[l],in[l]);
          }
          break;
        }
        default:
          DUNE_THROW(NotImplemented, "Desired derivative order is not implemented");
      }
    }

    /** \brief Evaluate derivative in a given direction
     * \param [in]  direction The direction to derive in
     * \param [in]  in        Position where to evaluate
     * \param [out] out       The return value
     */
    template<int diffOrder>
    inline void evaluate(
      const std::array<int,1>& direction,
      const typename Traits::DomainType& in,
      std::vector<typename Traits::RangeType>& out) const
    {
      static_assert(diffOrder == 1, "We only can compute first derivatives");
      out.resize(size());

      // Loop over all shape functions
      for (size_t i=0; i<size(); i++)
      {
        // convert index i to multiindex
        Dune::FieldVector<int,d> alpha(multiindex(i));

        // Loop over all coordinate directions
        std::size_t j = direction[0];

        // Initialize: the overall expression is a product
        // if j-th bit of i is set to -1, else 1
        out[i][0] = dp(alpha[j],in[j]);

        // rest of the product
        for (std::size_t l=0; l<d; l++)
          if (l!=j)
            out[i][0] *= p(alpha[l],in[l]);
      }
    }

    //! \brief Polynomial order of the shape functions
    unsigned int order () const
    {
      return k;
    }

  };
}

#endif
