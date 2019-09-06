// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_HPDG_LOCALFUNCTIONS_QK_GAUSS_LOBATTO_LOCALFINITEELEMENT_HH
#define DUNE_HPDG_LOCALFUNCTIONS_QK_GAUSS_LOBATTO_LOCALFINITEELEMENT_HH

#include "qkgausslobatto/qkgllocalinterpolation.hh"
#include "qkgausslobatto/qkgllocalbasis.hh"
#include <dune/localfunctions/lagrange/lagrangecube.hh> // Makes no difference whether GL or equidistant nodes are used

namespace Dune
{
  /** \brief General Lagrange finite element for cubes with arbitrary dimension and polynomial order up to 17
   * using Gauss Lobatto interpolation nodes instead of equidistant points.
   *
   * \tparam D type used for domain coordinates
   * \tparam R type used for function values
   * \tparam d dimension of the reference element
   * \tparam k polynomial order
   */
  template<class D, class R, int d, int k>
  class QkGaussLobattoLocalFiniteElement {

    typedef QkGaussLobattoLocalBasis<D,R,k,d> LocalBasis;
    typedef Dune::Impl::LagrangeCubeLocalCoefficients<k,d> LocalCoefficients;
    typedef QkGaussLobattoLocalInterpolation<k,d,LocalBasis> LocalInterpolation;

  public:

    /** \todo Please doc me !
     */
    typedef LocalFiniteElementTraits<LocalBasis,Impl::LagrangeCubeLocalCoefficients<k,d>,LocalInterpolation> Traits;

    /** \todo Please doc me !
     */
    QkGaussLobattoLocalFiniteElement ()
      : gt(GeometryTypes::cube(d)) {}

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

    /** \brief Number of shape functions in this finite element */
    unsigned int size () const
    {
      return basis.size();
    }

    /** \todo Please doc me !
     */
    GeometryType type () const
    {
      return gt;
    }

    QkGaussLobattoLocalFiniteElement* clone () const
    {
      return new QkGaussLobattoLocalFiniteElement(*this);
    }

    int order() const {
      return k;
    }

  private:
    LocalBasis basis;
    LocalCoefficients coefficients;
    LocalInterpolation interpolation;
    GeometryType gt;
  };

}

#endif
