// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <vector>
namespace Dune {
namespace HPDG {
namespace Impl {

  // Stupid quadrature rule does not allow me to edit the quad points :(
  template<class R>
  struct QP {
    typename R::value_type::Vector pos;
    auto& position() {
      return pos;
    }
    const auto& position() const{
      return pos;
    }
  };
}
/** Extracts the _nodes_ (not the weights) from
 * a quadrature rule and gives mutable access via
 * position() on each element.
 *
 * This is to hack around the fact that calling
 * position() on a Dune quadrature rule gives only
 * const access.
 */
template<class Rule>
auto mutableQuadratureNodes(const Rule& r) {
  std::vector<Impl::QP<Rule>> v(r.size());
  for(std::size_t i = 0; i < r.size(); i++) {
    v[i].pos = r[i].position();
  }
  return v;
};
}
}
