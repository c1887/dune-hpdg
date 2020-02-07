// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#pragma once


#include <dune/localfunctions/common/localfiniteelementvariantcache.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
#include <dune/common/indices.hh>

#include <tuple>
#include <utility>

namespace Dune {

namespace Impl {

  /** Helper construct that returns a tuple of pairs
   * where each pair is of the form
   * (j, f(j)), j=0,...i-1 in a static way.
   */
  template <class F, class I, I... i>
  constexpr auto static_enumerate(F &&f, std::integer_sequence<I, i...>) {
    return std::make_tuple(
        std::make_pair(i, f(std::integral_constant<I, i>()))...);
  }

  template<class D, class R, std::size_t dim, int maxOrder>
  struct QkGLFiniteElements
  {
    static auto getImplementations()
    {
      // generate all orders up to order maxOrder-1
      auto seq = std::make_integer_sequence<int, maxOrder>();

      return static_enumerate(
          [](auto k){ return [k](){ return Dune::QkGaussLobattoLocalFiniteElement<D, R, dim, k.value>();};},
          std::move(seq)
      );
    }

    /** In this case, this is just identity */
    constexpr static auto index(int idx) {
      return idx;
    }
  };

} // namespace Impl

/** LFE Cache that contains all orders of the Qk local finite element with Gauss--Lobatto nodes up to
 * a given order 'maxOrder'.
 */
template<class D, class R, std::size_t dim, std::size_t maxOrder>
using QkGLVaryingOrderCache = LocalFiniteElementVariantCache<Impl::QkGLFiniteElements<D,R,dim,maxOrder>>;

} // namespace Dune


