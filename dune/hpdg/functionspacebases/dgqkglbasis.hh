// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DG_QK_GL_BASIS_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_DG_QK_GL_BASIS_HH

#include <array>
#include <dune/common/exceptions.hh>

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/defaultglobalbasis.hh>
#include <dune/functions/functionspacebases/flatmultiindex.hh>
#include <dune/hpdg/functionspacebases/qkglnodalbasis.hh>




namespace Dune {
namespace Functions {



// *****************************************************************************
// This is the reusable part of the basis. It contains
//
//   DGQkGLNodeFactory
//   DGQkGLNodeIndexSet
//   DGQkGLNode
//
// The factory allows to create the others and is the owner of possible shared
// state. These three components do _not_ depend on the global basis or index
// set and can be used without a global basis.
// *****************************************************************************

template<typename GV, int k>
using DGQkGLNode = QkGLNode<GV, k>;

template<typename GV, int k, class MI, bool useTwoLevelIndex=false>
class DGQkGLNodeIndexSet;


template<typename GV, int k, class MI, bool useTwoLevelIndex=false>
class DGQkGLNodeFactory
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


  using Node = DGQkGLNode<GV, k>;

  using IndexSet = DGQkGLNodeIndexSet<GV, k, MI, useTwoLevelIndex>;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  using SizePrefix = Dune::ReservedVector<size_type, 2>;

  /** \brief Constructor for a given grid view object */
  DGQkGLNodeFactory(const GridView& gv) :
    gridView_(gv)
  {}

  /** Dummy function such that the concept passes */
  void update(const GridView&) {};

  void initializeIndices()
  {
    if (useTwoLevelIndex) {
      switch (dim)
      {
        case 1:
        {
          break;
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
    else {
      switch (dim)
      {
        case 1:
        {
          break;
        }
        case 2:
        {
          auto triangle = GeometryTypes::triangle;
          quadrilateralOffset_ = dofsPerTriangle * gridView_.size(triangle);
          break;
        }
        case 3:
        {
          auto tetrahedron = GeometryTypes::simplex(3);
          prismOffset_         = dofsPerTetrahedron * gridView_.size(tetrahedron);

          auto prism = GeometryTypes::prism;
          hexahedronOffset_    = prismOffset_         +   dofsPerPrism * gridView_.size(prism);

          auto hexahedron = GeometryTypes::cube(3);
          pyramidOffset_       = hexahedronOffset_    +   dofsPerHexahedron * gridView_.size(hexahedron);
          break;
        }
      }
    }
  }

  /** \brief Obtain the grid view that the basis is defined on
   */
  const GridView& gridView() const
  {
    return gridView_;
  }

  Node makeNode() const
  {
    return Node{};
  }

  IndexSet makeIndexSet() const
  {
    return IndexSet{*this};
  }

  size_type size() const
  {
    if (useTwoLevelIndex) {
      switch (dim)
      {
        case 1:
          return gridView_.size(0);
        case 2:
        {
          auto triangle = GeometryTypes::triangle;
          auto quad = GeometryTypes::quadrilateral;
          return gridView_.size(triangle) + gridView_.size(quad);
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
    else {
      switch (dim)
      {
        case 1:
          return dofsPerEdge*gridView_.size(0);
        case 2:
        {
          auto triangle = GeometryTypes::triangle;
          auto quad = GeometryTypes::quadrilateral;
          return dofsPerTriangle*gridView_.size(triangle) + dofsPerQuad*gridView_.size(quad);
        }
        case 3:
        {
          auto tetrahedron = GeometryTypes::tetrahedron;
          auto pyramid = GeometryTypes::pyramid;
          auto prism = GeometryTypes::prism;
          auto hexahedron = GeometryTypes::cube(3);
          return dofsPerTetrahedron*gridView_.size(tetrahedron) + dofsPerPyramid*gridView_.size(pyramid)
            + dofsPerPrism*gridView_.size(prism) + dofsPerHexahedron*gridView_.size(hexahedron);
        }
      }
      DUNE_THROW(Dune::NotImplemented, "No size method for " << dim << "d grids available yet!");
    }
  }

  //! Return number possible values for next position in multi index
  size_type size(const SizePrefix prefix) const
  {
    if (useTwoLevelIndex) {
      if (prefix.size() == 0)
        return size();
      else if (prefix.size() == 1)
        return maxNodeSize();
      else
        return 0;
      assert(false);
    }

    assert(prefix.size() == 0 || prefix.size() == 1);
    return (prefix.size() == 0) ? size() : 0;
  }

  /** \todo This method has been added to the interface without prior discussion. */
  size_type dimension() const
  {
    if (useTwoLevelIndex) return size()*maxNodeSize();
    return size();
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



template<typename GV, int k, class MI, bool useTwoLevelIndex>
class DGQkGLNodeIndexSet
{
  // Cannot be an enum -- otherwise the switch statement below produces compiler warnings
  static const int dim = GV::dimension;

public:

  using size_type = std::size_t;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  using NodeFactory = DGQkGLNodeFactory<GV, k, MI, useTwoLevelIndex>;
  using PreBasis = NodeFactory;

  using Node = typename NodeFactory::Node;

  DGQkGLNodeIndexSet(const NodeFactory& nodeFactory) :
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
    const auto& gridIndexSet = nodeFactory_->gridView().indexSet();
    const auto& element = node_->element();
    for (size_t i =0, end = node_->finiteElement().size(); i<end; ++i, ++it) {
      if (useTwoLevelIndex) {
        switch (dim)
        {
          case 1:
            *it= {{gridIndexSet.subIndex(element, 0, 0), i}};
            continue;
          case 2:
          {
            if (element.type().isQuadrilateral()) {
              *it= {{nodeFactory_->quadrilateralOffset_ + gridIndexSet.subIndex(element, 0,0), i}};
              continue;
            }
            else
              DUNE_THROW(Dune::NotImplemented, "2d elements have to be quadrilaterals");
          }
          case 3:
            if (element.type().isHexahedron()) {
              *it= {{nodeFactory_->hexahedronOffset_+ gridIndexSet.subIndex(element, 0,0), i}};
              continue;
            }
            else
              DUNE_THROW(Dune::NotImplemented, "2d elements have to be triangles or quadrilaterals");
        }
      }

      switch (dim)
      {
        case 1:
        {
          *it= {{nodeFactory_->dofsPerEdge*gridIndexSet.subIndex(element,0,0) + i}};
          continue;
        }
        case 2:
        {
          if (element.type().isQuadrilateral())
            *it= {{nodeFactory_->quadrilateralOffset_ + nodeFactory_->dofsPerQuad*gridIndexSet.subIndex(element,0,0) + i}};
          else
            DUNE_THROW(Dune::NotImplemented, "DGQkGL in 2d is only implemented on quadrilaterals");
          continue;
        }
        case 3:
        {
          if (element.type().isHexahedron())
          {
            *it = {{nodeFactory_->hexahedronOffset_ + nodeFactory_->dofsPerHexahedron*gridIndexSet.subIndex(element,0,0) + i}};
          }
          else
            DUNE_THROW(Dune::NotImplemented, "3d elements have to be tetrahedrons, prisms, hexahedrons or pyramids");
        }
      }
      DUNE_THROW(Dune::NotImplemented, "No index method for " << dim << "d grids available yet!");
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
 * \tparam k The order of the basis
 */
template<typename GV, int k>
using DGQkGLBasis = DefaultGlobalBasis<DGQkGLNodeFactory<GV, k, FlatMultiIndex<std::size_t>> >;

template<typename GV, int k>
using DGQkGLBlockBasis = DefaultGlobalBasis<DGQkGLNodeFactory<GV, k, std::array<std::size_t, 2>, true> >;


} // end namespace Functions
} // end namespace Dune


#endif // DUNE_FUNCTIONS_FUNCTIONSPACEBASES_LAGRANGEDGBASIS_HH
