// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once

#include <dune/common/math.hh>

#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/type.hh>

#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/resizehelper.hh>
#include <dune/hpdg/common/indexedcache.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

namespace Dune {
namespace HPDG {

namespace LumpedMassImpl { // do not clutter Impl namespace too much

  // group what belongs together
  template<class ctype, int dim>
  struct GaussLobattoQuadPoint {
    FieldVector<ctype, dim> position;
    double weight;
  };

  /* Though this is several times implemented in localfunctions,
   * k has to be constexpr there.
   *
   * @param i flat index
   * @param k polynomial order
   */
  template<int dim>
  auto flatIndexToMultiIndex (int i, int k){
    std::array<unsigned int,dim> alpha;
    for (int j=0; j<dim; j++)
    {
      alpha[j] = i % (k+1);
      i = i/(k+1);
    }
    return alpha;
  };

  template<int dim, typename Q>
  auto quadPointFromFlatIndex(const Q& quad, std::size_t flatIdx) {
    const int k = quad.size()-1;

    auto idx = flatIndexToMultiIndex<dim>(flatIdx, k);

    auto pos = FieldVector<double, dim>();
    for(std::size_t i = 0; i < dim; i++) {
      pos[i] = quad[idx[i]].position();
    }

    return pos;
  }

  template<int dim, typename Q>
  auto weightFromFlatIndex(const Q& quad, std::size_t flatIdx) {
    const int k = quad.size()-1;
    auto idx = flatIndexToMultiIndex<dim>(flatIdx, k);

    double weight = 1.;
    for(auto i : idx) {
      weight *= quad[i].weight();
    }

    return weight;
  }

}

  template<typename GV>
  auto gaussLobattoLumpedMass(const Dune::Functions::DynamicDGQkGLBlockBasis<GV>& basis) {

    /* Extract the dimension from the GridView.
     *
     * For some reason, g++ 6.3 has problems accepting this as an int
     * later when calling power(), thus we (redundantly) cast to int.
     */
    constexpr const int dim = static_cast<int>(GV::dimension);

    using ctype = typename GV::Grid::ctype;

    auto generator = [&](auto d) {
        auto size = power(d+1, dim);
        auto ret = std::vector<LumpedMassImpl::GaussLobattoQuadPoint<ctype, dim>>(size);

        int order = 2*d- 1;
        auto gauss_lobatto = Dune::QuadratureRules<double,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        // sort quad points (they're also ordered for the basis)
        std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
            return a.position() < b.position(); });

        for(std::size_t i=0; i< size; ++i) {
          ret[i].position = LumpedMassImpl::quadPointFromFlatIndex<dim>(gauss_lobatto, i);
          ret[i].weight = LumpedMassImpl::weightFromFlatIndex<dim>(gauss_lobatto, i);
        }

        return ret;
      };

    using Cache = HPDG::IndexedCache< std::vector<LumpedMassImpl::GaussLobattoQuadPoint<ctype, dim>> >;
    Cache cache(generator);


    //using Cache = QuadratureRuleCache<ctype, dim>;
    const auto& gv = basis.gridView();
    using V = DynamicBlockVector<Dune::FieldVector<double,1>>;

    V mass;
    resizeFromBasis(mass, basis);

    auto lv = basis.localView();
    for(const auto& element : elements(gv)) {
      lv.bind(element);

      // get element index ( ie. the first of the multiindex)
      auto idx = lv.index(0)[0];
      auto& xi = mass[idx];

      // get local ansatz degree
      auto degree = basis.preBasis().degree(element);

      auto quad = cache[degree];

      assert(quad.size() == lv.size());

      auto geo = element.geometry();

      for(std::size_t j = 0; j < quad.size(); ++j) {
        xi[j] = quad[j].weight*geo.integrationElement(quad[j].position);
      }
    }
    return mass;
  }
}
}
