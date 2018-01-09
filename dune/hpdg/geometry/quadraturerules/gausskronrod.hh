// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_GEOMETRY_GAUSS_KRONROD_HH
#define DUNE_HPDG_GEOMETRY_GAUSS_KRONROD_HH
#include <vector>
#include <array>
#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>

namespace Dune {
namespace HPDG {

  // For now, we hardcode FieldVector<double>
  class GaussKronrod1DRule :
    public QuadratureRule<double, 1> {


    public:
    /* \brief Return Gauss-Kronrod rule of size 2n+1
     *
     * This will have the n nodes from a corresponding Gauss-Legendre
     * rule plus n+1 additional Gauss-Kronrod nodes
     */
    GaussKronrod1DRule(int n):
        QuadratureRule<double, 1>(GeometryTypes::line)
    {
      // the nodes and weights are generated from the other file:
      using DT = Dune::FieldVector<double, 1>;
#include "gausskronrod_table.hh"
      for (size_t i = 0; i < weights_.size(); i++) {
        this->push_back(QuadraturePoint<double,1>(nodes_[i], weights_[i]));
      }

    }

    GaussKronrod1DRule() = delete;
  };

}
}
#endif//DUNE_HPDG_GEOMETRY_GAUSS_KRONROD_HH
