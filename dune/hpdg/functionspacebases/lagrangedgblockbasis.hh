// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_LAGRANGEDGBLOCKBASIS_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_LAGRANGEDGBLOCKBASIS_HH

#include <array>
#include <dune/common/exceptions.hh>

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/defaultglobalbasis.hh>
#include <dune/functions/functionspacebases/flatmultiindex.hh>
#include <dune/functions/functionspacebases/pqknodalbasis.hh>




namespace Dune {
namespace Functions {



// *****************************************************************************
// This is the reusable part of the basis. It contains
//
//   LagrangeDGBlockNodeFactory
//   LagrangeDGBlockNodeIndexSet
//   LagrangeDGBlockNode
//
// The factory allows to create the others and is the owner of possible shared
// state. These three components do _not_ depend on the global basis or index
// set and can be used without a global basis.
// *****************************************************************************

template<typename GV, int k, typename TP>
using LagrangeDGBlockNode = PQkNode<GV, k, TP>;

template<typename GV, int k, class MI, class TP>
class LagrangeDGBlockNodeIndexSet;


template<typename GV, int k, class MI>
class LagrangeDGBlockNodeFactory
{
  static const int dim = GV::dimension;

public:

  /** \brief The grid view that the FE space is defined on */
  using GridView = GV;
  using size_type = std::size_t;


  // Precompute the number of dofs per entity type
  const static int dofsPerEdge        = k+1;
  const static int dofsPerTriangle    = (k+1)*(k+2)/2;
  const static int dofsPerQuad        = (k+1)*(k+1);
  const static int dofsPerTetrahedron = (k+1)*(k+2)*(k+3)/6;
  const static int dofsPerPrism       = (k+1)*(k+1)*(k+2)/2;
  const static int dofsPerHexahedron  = (k+1)*(k+1)*(k+1);
  const static int dofsPerPyramid     = (k+1)*(k+2)*(2*k+3)/6;


  template<class TP>
  using Node = LagrangeDGBlockNode<GV, k, TP>;

  template<class TP>
  using IndexSet = LagrangeDGBlockNodeIndexSet<GV, k, MI, TP>;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  using SizePrefix = Dune::ReservedVector<size_type, 3>;

  /** \brief Constructor for a given grid view object */
  LagrangeDGBlockNodeFactory(const GridView& gv) :
    gridView_(gv)
  {}


  void initializeIndices()
  {
    switch (dim)
    {
      case 1:
      {
        break;
      }
      case 2:
      {
        GeometryType triangle;
        triangle.makeTriangle();
        quadrilateralOffset_ = dofsPerTriangle * gridView_.size(triangle);
        break;
      }
      case 3:
      {
        GeometryType tetrahedron;
        tetrahedron.makeSimplex(3);
        prismOffset_         = dofsPerTetrahedron * gridView_.size(tetrahedron);

        GeometryType prism;
        prism.makePrism();
        hexahedronOffset_    = prismOffset_         +   dofsPerPrism * gridView_.size(prism);

        GeometryType hexahedron;
        hexahedron.makeCube(3);
        pyramidOffset_       = hexahedronOffset_    +   dofsPerHexahedron * gridView_.size(hexahedron);
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

  template<class TP>
  Node<TP> node(const TP& tp) const
  {
    return Node<TP>{tp};
  }

  template<class TP>
  IndexSet<TP> indexSet() const
  {
    return IndexSet<TP>{*this};
  }

  size_type size() const
  {
    switch (dim)
    {
      case 1:
        return gridView_.size(0);
      case 2:
      {
        GeometryType triangle, quad;
        triangle.makeTriangle();
        quad.makeQuadrilateral();
        return gridView_.size(triangle) + gridView_.size(quad);
      }
      case 3:
      {
        GeometryType tetrahedron, pyramid, prism, hexahedron;
        tetrahedron.makeTetrahedron();
        pyramid.makePyramid();
        prism.makePrism();
        hexahedron.makeCube(3);
        return gridView_.size(tetrahedron) + gridView_.size(pyramid)
             + gridView_.size(prism) + gridView_.size(hexahedron);
      }
    }
    DUNE_THROW(Dune::NotImplemented, "No size method for " << dim << "d grids available yet!");
  }

  //! Return number possible values for next position in multi index
  size_type size(const SizePrefix prefix) const
  {
    if (prefix.size() == 0)
      return size();
    else if (prefix.size()== 1)
      return maxNodeSize();
    else
      return 0;
    assert(false);
  }

  /** \todo This method has been added to the interface without prior discussion. */
  size_type dimension() const
  {
    return size()*maxNodeSize();
  }

  size_type maxNodeSize() const
  {
    return StaticPower<(k+1),GV::dimension>::power;
  }

//protected:
  const GridView gridView_;

  size_t quadrilateralOffset_;
  size_t pyramidOffset_;
  size_t prismOffset_;
  size_t hexahedronOffset_;
};



template<typename GV, int k, class MI, class TP>
class LagrangeDGBlockNodeIndexSet
{
  // Cannot be an enum -- otherwise the switch statement below produces compiler warnings
  static const int dim = GV::dimension;

public:

  using size_type = std::size_t;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  using NodeFactory = LagrangeDGBlockNodeFactory<GV, k, MI>;

  using Node = typename NodeFactory::template Node<TP>;

  LagrangeDGBlockNodeIndexSet(const NodeFactory& nodeFactory) :
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
  MultiIndex index(size_type i) const
  {
    const auto& gridIndexSet = nodeFactory_->gridView().indexSet();
    const auto& element = node_->element();

    return {gridIndexSet.subIndex(element, 0, 0) , i};
  }

protected:
  const NodeFactory* nodeFactory_;

  const Node* node_;
};



// Eventually, make the size_type a template parameter. Then again, this complicates things maybe too much
using TwoLevelIndex = std::array<size_t, 2>;

// *****************************************************************************
// This is the actual global basis implementation based on the reusable parts.
// *****************************************************************************

/** \brief Basis of a scalar k-th-order Lagrangean-DG finite element space
 *
 * \ingroup FunctionSpaceBasesImplementations
 *
 * \tparam GV The GridView that the space is defined on
 * \tparam k The order of the basis
 */
template<typename GV, int k>
using LagrangeDGBlockBasis = DefaultGlobalBasis<LagrangeDGBlockNodeFactory<GV, k, TwoLevelIndex> >;



} // end namespace Functions
} // end namespace Dune


#endif // DUNE_FUNCTIONS_FUNCTIONSPACEBASES_LAGRANGEDGBLOCKBASIS_HH
