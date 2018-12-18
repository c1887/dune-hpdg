// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_DG_QK_GaussKronrod_BASIS_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DYNAMIC_DG_QK_GaussKronrod_BASIS_HH

#include <array>
#include <dune/common/exceptions.hh>

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/defaultglobalbasis.hh>
#include <dune/functions/functionspacebases/flatmultiindex.hh>
#include <dune/hpdg/functionspacebases/dynamicqknode.hh>

#include <dune/grid/common/mcmgmapper.hh>



namespace Dune {
namespace Functions {



// *****************************************************************************
// This is the reusable part of the basis. It contains
//
//   DynamicDGQkGaussKronrodNodeFactory
//   DynamicDGQkGaussKronrodNodeIndexSet
//   DynamicQkGaussKronrodNode
//
// The factory allows to create the others and is the owner of possible shared
// state. These three components do _not_ depend on the global basis or index
// set and can be used without a global basis.
// *****************************************************************************


template<typename GV, class MI>
class DynamicDGQkGaussKronrodNodeIndexSet;


template<typename GV, class MI>
class DynamicDGQkGaussKronrodNodeFactory
{
  static const int dim = GV::dimension;

public:

  /** \brief The grid view that the FE space is defined on */
  using GridView = GV;
  using size_type = std::size_t;

  using DegreeMap = std::vector<int>;
  using Node = DynamicQkGaussKronrodNode<GV, MultipleCodimMultipleGeomTypeMapper<GridView>, DegreeMap>;

  using IndexSet = DynamicDGQkGaussKronrodNodeIndexSet<GV, MI>;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  using SizePrefix = Dune::ReservedVector<size_type, 2>;

  /** \brief Constructor for a given grid view object */
  DynamicDGQkGaussKronrodNodeFactory(const GridView& gv, int k = 1) :
    gridView_(gv),
    mcmgMapper_(gv, mcmgElementLayout())
  {
    // fill Degreemap
    degreeMap_.resize(mcmgMapper_.size());
    for (const auto& e : elements(gridView_))
      degreeMap_[mcmgMapper_.index(e)]=k;
  }


  void initializeIndices()
  {
    switch (dim)
    {
      case 1: { break;
      }
      case 2:
      {
        auto triangle = GeometryTypes::triangle;
        quadrilateralOffset_ = gridView_.size(triangle);
        break;
      }
      case 3:
      {
        auto tetrahedron = GeometryTypes::simplex(3);
        prismOffset_ = gridView_.size(tetrahedron);

        auto prism = GeometryTypes::prism;
        hexahedronOffset_ = prismOffset_ + gridView_.size(prism);

        auto hexahedron = GeometryTypes::cube(3);
        pyramidOffset_ = hexahedronOffset_ + gridView_.size(hexahedron);
        break;
      }
    }
  }

  /** \brief Obtain the grid view that the basis is defined on
   */
  const GridView& gridView() const
  {
    return gridView_;
  }

  void update(const GridView& gv) {
    gridView_ = gv;
    mcmgMapper_.update();
    degreeMap_.resize(mcmgMapper_.size()); // TODO: should all updates happen here?
  }

  Node makeNode() const
  {
    return Node{&mcmgMapper_, &degreeMap_};
  }


  IndexSet makeIndexSet() const
  {
    return IndexSet{*this};
  }

  size_type size() const
  {
    switch (dim)
    {
      case 1:
        return gridView_.size(0);
      case 2:
        {
          auto quad = GeometryTypes::quadrilateral;
          return gridView_.size(quad);
        }
      case 3:
        {
          auto tetrahedron = GeometryTypes::tetrahedron;
          auto pyramid = GeometryTypes::pyramid;
          auto prism = GeometryTypes::prism;
          auto hexahedron = GeometryTypes::cube(3);
          return gridView_.size(tetrahedron) + gridView_.size(pyramid)
            + gridView_.size(prism) + gridView_.size(hexahedron);
        }
    }
  }

  //! Return number possible values for next position in multi index
  size_type size(const SizePrefix prefix) const
  {
    if (prefix.size() == 0)
      return size();
    else if (prefix.size() == 1)
      return static_cast<size_type>(std::pow(degreeMap_.at(prefix[0])+1, dim));
    else
      return 0;
  }

  /** \todo This method has been added to the interface without prior discussion. */
  size_type dimension() const
  {
    size_type count = 0;
    for (const auto& e: degreeMap_)
      count += std::pow(e+1, dim);
    return count;
  }

  size_type maxNodeSize() const
  {
    // find highest degree
    auto m = *std::max_element(std::begin(degreeMap_), std::end(degreeMap_));

    // return tensor-product size
    return std::pow((size_type)m+1, dim);
  }

  template<typename Element>
  auto& degree(const Element& e){
    return degreeMap_.at(mcmgMapper_.index(e));
  }

//protected:
  GridView gridView_;
  MultipleCodimMultipleGeomTypeMapper<GridView> mcmgMapper_;
  DegreeMap degreeMap_;

  size_t quadrilateralOffset_;
  size_t pyramidOffset_;
  size_t prismOffset_;
  size_t hexahedronOffset_;
};



template<typename GV, class MI>
class DynamicDGQkGaussKronrodNodeIndexSet
{
  // Cannot be an enum -- otherwise the switch statement below produces compiler warnings
  static const int dim = GV::dimension;

public:

  using size_type = std::size_t;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  using NodeFactory = DynamicDGQkGaussKronrodNodeFactory<GV, MI>;
  using PreBasis = NodeFactory;

  using Node = typename NodeFactory::Node;

  DynamicDGQkGaussKronrodNodeIndexSet(const NodeFactory& nodeFactory) :
    nodeFactory_(&nodeFactory)
  {}

  /** \brief Bind the view to a grid element
   *
   * Having to bind the view to an element before being able to actually access any of its data members
   * offers to centralize some expensive setup code in the 'bind' method, which can save a lot of run-time.
   */
  void bind(const Node& node)
  {
    node_ = &node;
  }

  /** \brief Unbind the view
   */
  void unbind()
  {
    node_ = nullptr;
  }

  /** \brief Size of subtree rooted in this node (element-local)
   */
  size_type size() const
  {
    return node_->finiteElement().size();
  }


  //! Maps from subtree index set [0..size-1] to a globally unique multi index in global basis
  template<class It>
  It indices(It it) const
  {
    for (size_t i =0, end = node_->finiteElement().size(); i <end; i++, ++it) {
      const auto& gridIndexSet = nodeFactory_->gridView().indexSet();
      const auto& element = node_->element();

      // Our Gauss-Lobatto basis is defined for tensor-product bases on cubic grids only. Hence, we do not have
      // to differentiate between the different geometries and dimensions.
      *it= {{gridIndexSet.subIndex(element, 0, 0), i}};
    }
    return it;
  }

protected:
  const NodeFactory* nodeFactory_;

  const Node* node_;
};



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
using DynamicDGQkGaussKronrodBlockBasis = DefaultGlobalBasis<DynamicDGQkGaussKronrodNodeFactory<GV, std::array<std::size_t, 2> > >;


} // end namespace Functions
} // end namespace Dune


#endif
