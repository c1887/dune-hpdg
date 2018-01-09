// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_HPDG_LOCALFUNCTIONS_QK_GAUSS_KRONROD_HH
#define DUNE_HPDG_LOCALFUNCTIONS_QK_GAUSS_KRONROD_HH

#include <dune/localfunctions/lagrange/p0.hh>
#include <dune/localfunctions/lagrange/pk.hh>
#include <dune/localfunctions/lagrange/q1.hh>
#include <dune/localfunctions/lagrange/qk.hh>
#include <dune/localfunctions/lagrange/prismp1.hh>
#include <dune/localfunctions/lagrange/prismp2.hh>
#include <dune/localfunctions/lagrange/pyramidp1.hh>
#include <dune/localfunctions/lagrange/pyramidp2.hh>


#include <dune/localfunctions/common/localbasis.hh>
#include <dune/localfunctions/common/localkey.hh>
#include <dune/localfunctions/common/localfiniteelementtraits.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/hpdg/localfunctions/lagrange/qkdynamicordercache.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslegendre.hh>
#include <dune/hpdg/geometry/quadraturerules/gausskronrod.hh>
// this is a modified header of dune-pdelab ported to support Gauss-Kronrod based
// Qk local finite elements
namespace Dune
{
  namespace QkStuff
  {

    //! Lagrange polynomials at Gauss-Lobatto points
    template<class D, class R, int k>
    class GaussKronrodLagrangePolynomials
    {
      R xi_gl[k+1];
      R w_gl[k+1];

    public:
      GaussKronrodLagrangePolynomials ()
      {
        // rationale: Every Gauss-Kronrod rule has size 2*n+1, hence, you'd get
        // polynamials of order 2*n, which obviously can only be even
        if (k%2!=0)
          DUNE_THROW(Dune::NotImplemented, "No odd-sized Gauss-Kronrod rules!");

        auto rule = Dune::HPDG::GaussKronrod1DRule(k/2);
        auto matched_size = rule.size();
        assert(matched_size == k+1);

        size_t count=0;
        for (auto it=rule.begin(); it!=rule.end(); ++it)
        {
          size_t group=count/2;
          size_t member=count%2;
          size_t newj;
          if (member==1) newj=group; else newj=k-group;
          xi_gl[newj] = it->position()[0];
          w_gl[newj] = it->weight();
          count++;
        }
        for (int j=0; j<matched_size/2; j++) {
          if (xi_gl[j]>0.5)
          {
            R temp=xi_gl[j];
            xi_gl[j] = xi_gl[k-j];
            xi_gl[k-j] = temp;
            temp=w_gl[j];
            w_gl[j] = w_gl[k-j];
            w_gl[k-j] = temp;
          }
        }
      }

      // ith Lagrange polynomial of degree k in one dimension
      R p (int i, D x) const
      {
        R result(1.0);
        for (int j=0; j<=k; j++)
          if (j!=i) result *= (x-xi_gl[j])/(xi_gl[i]-xi_gl[j]);
        return result;
      }

      // derivative of ith Lagrange polynomial of degree k in one dimension
      R dp (int i, D x) const
      {
        R result(0.0);

        for (int j=0; j<=k; j++)
          if (j!=i)
            {
              R prod( 1.0/(xi_gl[i]-xi_gl[j]) );
              for (int l=0; l<=k; l++)
                if (l!=i && l!=j) prod *= (x-xi_gl[l])/(xi_gl[i]-xi_gl[l]);
              result += prod;
            }
        return result;
      }

      // get ith Lagrange point
      R x (int i) const
      {
        return xi_gl[i];
      }

      // get weight of ith Lagrange point
      R w (int i) const
      {
        return w_gl[i];
      }
    };

    /**@ingroup LocalBasisImplementation
       \brief Lagrange shape functions of order k on the reference cube.

       Also known as \f$Q^k\f$.

       \tparam D Type to represent the field in the domain.
       \tparam R Type to represent the field in the range.
       \tparam k Polynomial degree
       \tparam d Dimension of the cube

       \nosubgrouping
    */
    template<class D, class R, int k, int d>
    class QkGaussKronrodLocalBasis
    {
      enum{ n = QkSize<k,d>::value };
      GaussKronrodLagrangePolynomials<D,R,k> poly;

    public:
      typedef LocalBasisTraits<D,d,Dune::FieldVector<D,d>,R,1,Dune::FieldVector<R,1>,Dune::FieldMatrix<R,1,d> > Traits;

      //! \brief number of shape functions
      unsigned int size () const
      {
        return QkSize<k,d>::value;
      }

      //! \brief Evaluate all shape functions
      inline void evaluateFunction (const typename Traits::DomainType& in,
                                    std::vector<typename Traits::RangeType>& out) const
      {
        out.resize(size());
        for (size_t i=0; i<size(); i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // initialize product
            out[i] = 1.0;

            // dimension by dimension
            for (int j=0; j<d; j++)
              out[i] *= poly.p(alpha[j],in[j]);
          }
      }

      //! \brief Evaluate Jacobian of all shape functions
      inline void
      evaluateJacobian (const typename Traits::DomainType& in,         // position
                        std::vector<typename Traits::JacobianType>& out) const      // return value
      {
        out.resize(size());

        // Loop over all shape functions
        for (size_t i=0; i<size(); i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // Loop over all coordinate directions
            for (int j=0; j<d; j++)
              {
                // Initialize: the overall expression is a product
                // if j-th bit of i is set to -1, else 1
                out[i][0][j] = poly.dp(alpha[j],in[j]);

                // rest of the product
                for (int l=0; l<d; l++)
                  if (l!=j)
                    out[i][0][j] *= poly.p(alpha[l],in[l]);
              }
          }
      }

      //! \brief Polynomial order of the shape functions
      unsigned int order () const
      {
        return k;
      }

    inline void partial(const std::array<unsigned int,d>& order,
                        const typename Traits::DomainType& in,
                        std::vector<typename Traits::RangeType>& out) const
    {
      DUNE_THROW(Dune::NotImplemented, "partial not implemented");
      //auto totalOrder = std::accumulate(order.begin(), order.end(), 0);

      //switch (totalOrder)
      //{
        //case 0:
          //evaluateFunction(in,out);
          //break;
        //case 1:
        //{
          //out.resize(size());

          //// Loop over all shape functions
          //for (size_t i=0; i<size(); i++)
          //{
            //// convert index i to multiindex
            //Dune::FieldVector<int,d> alpha(multiindex(i));

            //// Initialize: the overall expression is a product
            //out[i][0] = 1.0;

            //// rest of the product
            //for (std::size_t l=0; l<d; l++)
              //out[i][0] *= (order[l]) ? dp(alpha[l],in[l]) : p(alpha[l],in[l]);
          //}
          //break;
        //}
        //default:
          //DUNE_THROW(NotImplemented, "Desired derivative order is not implemented");
      //}
    }
    };

    /** \todo Please doc me! */
    template<int k, int d, class LB>
    class QkGaussKronrodLocalInterpolation
    {
      GaussKronrodLagrangePolynomials<double,double,k> poly;

    public:

      //! \brief Local interpolation of a function
      template<typename F, typename C>
      void interpolate (const F& f, std::vector<C>& out) const
      {
        typename LB::Traits::DomainType x;
        typename LB::Traits::RangeType y;

        out.resize(QkSize<k,d>::value);

        for (int i=0; i<QkSize<k,d>::value; i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // Generate coordinate of the i-th Lagrange point
            for (int j=0; j<d; j++)
              x[j] = poly.x(alpha[j]);

            f.evaluate(x,y); out[i] = y;
          }
      }
    };

    /** \todo Please doc me! */
    template<int d, class LB>
    class QkGaussKronrodLocalInterpolation<0,d,LB>
    {
    public:
      //! \brief Local interpolation of a function
      template<typename F, typename C>
      void interpolate (const F& f, std::vector<C>& out) const
      {
        typename LB::Traits::DomainType x(0.5);
        typename LB::Traits::RangeType y;
        f.evaluate(x,y);
        out.resize(1);
        out[0] = y;
      }
    };

  }

  /** \todo Please doc me !
   */
  template<class D, class R, int k, int d>
  class QkGaussKronrodLocalFiniteElement
  {
    typedef QkStuff::QkGaussKronrodLocalBasis<D,R,k,d> LocalBasis;
    typedef QkStuff::QkDGLocalCoefficients<k,d> LocalCoefficients;
    typedef QkStuff::QkGaussKronrodLocalInterpolation<k,d,LocalBasis> LocalInterpolation;

  public:
    // static number of basis functions
    enum{ n = QkStuff::QkSize<k,d>::value };

    /** \todo Please doc me !
     */
    typedef LocalFiniteElementTraits<LocalBasis,LocalCoefficients,LocalInterpolation> Traits;

    /** \todo Please doc me !
     */
    QkGaussKronrodLocalFiniteElement ()
      : gt(GeometryTypes::cube(d))
    {}

    /** \todo Please doc me !
     */
    const typename Traits::LocalBasisType& localBasis () const
    {
      return basis;
    }

    /** \todo Please doc me !
     */
    const typename Traits::LocalCoefficientsType& localCoefficients () const
    {
      return coefficients;
    }

    /** \todo Please doc me !
     */
    const typename Traits::LocalInterpolationType& localInterpolation () const
    {
      return interpolation;
    }

    /** \todo Please doc me !
     */
    GeometryType type () const
    {
      return gt;
    }

    QkGaussKronrodLocalFiniteElement* clone () const
    {
      return new QkGaussKronrodLocalFiniteElement(*this);
    }

    auto size() const {
      return n;
    }

  private:
    LocalBasis basis;
    LocalCoefficients coefficients;
    LocalInterpolation interpolation;
    GeometryType gt;
  };

 template<class D, class R, int dim>
  struct DynamicOrderQkGaussKronrodLocalFiniteElementFactory
  {
    //typedef typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits T;
    typedef typename Q1LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FiniteElementType;
    template<int k>
    using QkGaussKronrod = QkGaussKronrodLocalFiniteElement<D, R, k, dim>;
    //using QkGaussKronrod = QkGaussLobattoLocalFiniteElement<D, R, dim, k>;


    //! create finite element for given GeometryType
    static FiniteElementType* create(const size_t& gt)
    {
          // TODO: smart ptr
      switch(gt) {
        case(1):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<1>>(QkGaussKronrod<1>());
        case(2):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<2>>(QkGaussKronrod<2>());
        case(3):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<3>>(QkGaussKronrod<3>());
        case(4):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<4>>(QkGaussKronrod<4>());
        case(5):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<5>>(QkGaussKronrod<5>());
        case(6):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<6>>(QkGaussKronrod<6>());
        case(7):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<7>>(QkGaussKronrod<7>());
        case(8):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<8>>(QkGaussKronrod<8>());
        case(9):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<9>>(QkGaussKronrod<9>());
        case(10):
          return new LocalFiniteElementVirtualImp<QkGaussKronrod<10>>(QkGaussKronrod<10>());
        default:
          DUNE_THROW(Dune::NotImplemented, "Gauss-Kronrod only up to order 10");
      }
    }
  };


  template<class D, class R, int dim>
  using DynamicOrderQkGaussKronrodLocalFiniteElementCache = DynamicOrderQkLocalFiniteElementCache<D, R, dim, DynamicOrderQkGaussKronrodLocalFiniteElementFactory<D,R,dim>>;

}
#endif
