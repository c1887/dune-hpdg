// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_DG_QK_GL_BASIS_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_DG_QK_GL_BASIS_HH

#include <array>
#include <dune/common/exceptions.hh>
#include <dune/common/math.hh>

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/defaultglobalbasis.hh>
#include <dune/hpdg/functionspacebases/dynamicqknode.hh>

#include <dune/grid/common/mcmgmapper.hh>



namespace Dune {
namespace Functions {



// *****************************************************************************
// This is the reusable part of the basis. It contains
//
//   DynamicDGQkGLNodeFactory
//
// The factory allows to create the others and is the owner of possible shared
// state. These three components do _not_ depend on the global basis or index
// set and can be used without a global basis.
// *****************************************************************************



template<typename GV>
class DynamicDGQkGLNodeFactory
{
  static const int dim = GV::dimension;

public:

  /** \brief The grid view that the FE space is defined on */
  using GridView = GV;
  using size_type = std::size_t;

  using DegreeMap = std::vector<int>;
  using Node = DynamicQkGLNode<GV, MultipleCodimMultipleGeomTypeMapper<GridView>, DegreeMap>;

  static constexpr size_type maxMultiIndexSize = 2;
  static constexpr size_type minMultiIndexSize = 2;
  static constexpr size_type multiIndexBufferSize = 2;

  /** \brief Constructor for a given grid view object */
  DynamicDGQkGLNodeFactory(const GridView& gv, int k = 1) :
    gridView_(gv),
    mcmgMapper_(gv, mcmgElementLayout())
  {
    // fill Degreemap
    degreeMap_.resize(mcmgMapper_.size());
    for (auto& v: degreeMap_)
      v=k;
  }

  DynamicDGQkGLNodeFactory(const GridView& gv, const DegreeMap& degrees) :
    gridView_(gv),
    mcmgMapper_(gv, mcmgElementLayout())
  {
    setDegrees(degrees);
  }

  void initializeIndices()
  {
  }

  /** \brief Obtain the grid view that the basis is defined on
   */
  const GridView& gridView() const
  {
    return gridView_;
  }

  void update(const GridView& gv) {
    gridView_ = gv;
    mcmgMapper_.update(gv);
    degreeMap_.resize(mcmgMapper_.size()); // TODO: should all updates happen here?
  }

  Node makeNode() const
  {
    return Node{&mcmgMapper_, &degreeMap_};
  }


  size_type size() const
  {
    // return the number of elements.
    // This is exactly the number of possible values
    // in the next multi-index digit.
    return gridView_.size(0);
  }

  //! Return number possible values for next position in multi index
  template<typename SizePrefix>
  size_type size(const SizePrefix& prefix) const
  {
    if (prefix.size() == 0)
      return size();
    else if (prefix.size() == 1)
      return static_cast<size_type>(power(degreeMap_.at(prefix[0])+1, dim));
    else
      return 0;
  }

  /** \todo This method has been added to the interface without prior discussion. */
  size_type dimension() const
  {
    size_type count = 0;
    for (const auto& e: degreeMap_)
      count += power(e+1, dim);
    return count;
  }

  size_type maxNodeSize() const
  {
    // find highest degree
    auto m = *std::max_element(std::begin(degreeMap_), std::end(degreeMap_));

    // return tensor-product size
    return power((size_type)m+1, dim);
  }

  template<typename Element>
  const auto& degree(const Element& e) const {
    return degreeMap_.at(mcmgMapper_.index(e));
  }

  template<typename Element>
  auto& degree(const Element& e){
    return degreeMap_.at(mcmgMapper_.index(e));
  }

  /** Get vector that maps (mcmg) element index to local degree */
  DegreeMap degreeMap() const {
    return degreeMap_;
  }

  void setDegrees(const DegreeMap& degrees) {
    if(degrees.size() != mcmgMapper_.size())
      DUNE_THROW(Dune::Exception, "Supplied map in setDegrees() has wrong size!");
    degreeMap_=degrees;
  }

  template<typename It>
  It indices(const Node& node, It it) const
  {
    const auto& gridIndexSet = gridView_.indexSet();
    const auto& element = node.element();
    auto elementIdx = gridIndexSet.subIndex(element, 0, 0);
    for (size_t i = 0, end = node.finiteElement().size(); i < end; ++i, ++it) {

      // Our Gauss-Lobatto basis is defined for tensor-product bases on cubic
      // grids only. Hence, we do not have to differentiate between the
      // different geometries and dimensions.
      *it = { { elementIdx, i } };
    }
    return it;
  }

//protected:
  GridView gridView_;
  MultipleCodimMultipleGeomTypeMapper<GridView> mcmgMapper_;
  DegreeMap degreeMap_;
};

namespace BasisBuilder {

auto dynamicDG() {
  return [](const auto& gridView) {
    return DynamicDGQkGLNodeFactory<std::decay_t<decltype(gridView)>>(gridView);
  };
}

} // end namespace BasisBuilder


// *****************************************************************************
// This is the actual global basis implementation based on the reusable parts.
// *****************************************************************************

/** \brief Basis of a scalar k-th-order Discontinuous Galerkin finite element space
 *
 * \ingroup FunctionSpaceBasesImplementations
 *
 * \tparam GV The GridView that the space is defined on
 */
template<typename GV>
using DynamicDGQkGLBlockBasis = DefaultGlobalBasis<DynamicDGQkGLNodeFactory<GV> >;


} // end namespace Functions
} // end namespace Dune


#endif
