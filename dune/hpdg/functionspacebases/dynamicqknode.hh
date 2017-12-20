// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_QK_NODE_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_QK_NODE_HH

#include <array>
#include <dune/common/exceptions.hh>

#include <dune/hpdg/localfunctions/lagrange/qkglfactory.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslegendre.hh>

#include <dune/typetree/leafnode.hh>

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/flatmultiindex.hh>

#include <dune/grid/common/mcmgmapper.hh>

namespace Dune {
namespace Functions {


template<typename GV, typename TP, typename Mapper, typename DGM, typename Cache>
class DynamicQkNode :
  public LeafBasisNode<std::size_t, TP>
{
  static const int dim = GV::dimension;

  using Base = LeafBasisNode<std::size_t,TP>;
  using FiniteElementCache = Cache;
  using DegreeMap = DGM;

public:

  using size_type = std::size_t;
  using TreePath = TP;
  using Element = typename GV::template Codim<0>::Entity;
  using FiniteElement = typename FiniteElementCache::FiniteElementType;

  DynamicQkNode(const TreePath& treePath, const Mapper* m, const DegreeMap* dgm) :
    Base(treePath),
    finiteElement_(nullptr),
    element_(nullptr),
    mcmgMapper_(m),
    degreeMap_(dgm)
  {}

  //! Return current element, throw if unbound
  const Element& element() const
  {
    return *element_;
  }

  /** \brief Return the LocalFiniteElement for the element we are bound to
   *
   * The LocalFiniteElement implements the corresponding interfaces of the dune-localfunctions module
   */
  const FiniteElement& finiteElement() const
  {
    return cache_.get(degree_);
  }

  //! Bind to element.
  void bind(const Element& e)
  {
    element_ = &e;
    auto i = mcmgMapper_->index(e);

    degree_ = degreeMap_->at(i);
    finiteElement_ = &(cache_.get(degree_));
    this->setSize(finiteElement_->size());
  }

protected:

  FiniteElementCache cache_;
  const FiniteElement* finiteElement_;
  const Element* element_;
  const Mapper* mcmgMapper_;
  const DegreeMap* degreeMap_;
  int degree_ = 1;
};

// for backward-compatibility
template<typename GV, typename TP, typename Mapper, typename DGM>
using DynamicQkGLNode = DynamicQkNode<GV, TP, Mapper, DGM, Dune::DynamicOrderQkGLLocalFiniteElementCache<typename GV::ctype, double, GV::dimension>>;

template<typename GV, typename TP, typename Mapper, typename DGM>
using DynamicQkGaussLegendreNode = DynamicQkNode<GV, TP, Mapper, DGM, Dune::DynamicOrderQkGaussLegendreLocalFiniteElementCache<typename GV::ctype, double, GV::dimension>>;

} // end namespace Functions
} // end namespace Dune

#endif
