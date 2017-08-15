// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_QK_NODE_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_QK_NODE_HH

#include <array>
#include <dune/common/exceptions.hh>

#include <dune/hpdg/localfunctions/lagrange/qkglfactory.hh>

#include <dune/typetree/leafnode.hh>

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/flatmultiindex.hh>

#include <dune/grid/common/mcmgmapper.hh>

namespace Dune {
namespace Functions {


template<typename GV, typename TP, typename Mapper, typename DGM>
class DynamicQkGLNode :
  public LeafBasisNode<std::size_t, TP>
{
  static const int dim = GV::dimension;

  using Base = LeafBasisNode<std::size_t,TP>;
  using FiniteElementCache = typename Dune::DynamicOrderQkGLLocalFiniteElementCache<typename GV::ctype, double, dim>;
  using DegreeMap = DGM;

public:

  using size_type = std::size_t;
  using TreePath = TP;
  using Element = typename GV::template Codim<0>::Entity;
  using FiniteElement = typename FiniteElementCache::FiniteElementType;

  DynamicQkGLNode(const TreePath& treePath, const Mapper* m, const DegreeMap* dgm) :
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
    return *finiteElement_;
  }

  //! Bind to element.
  void bind(const Element& e)
  {
    element_ = &e;
    auto i = mcmgMapper_->index(e);

    finiteElement_ = &(cache_.get(degreeMap_->at(i)));
    this->setSize(finiteElement_->size());
  }

protected:

  FiniteElementCache cache_;
  const FiniteElement* finiteElement_;
  const Element* element_;
  const Mapper* mcmgMapper_;
  const DegreeMap* degreeMap_;
};


} // end namespace Functions
} // end namespace Dune

#endif
