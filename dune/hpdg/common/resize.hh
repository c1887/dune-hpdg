// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_COMMON_RESIZE_HH
#define DUNE_HPDG_COMMON_RESIZE_HH

#include <dune/common/indices.hh>
#include <dune/common/concept.hh>
#include <dune/common/typetraits.hh>

#include <dune/solvers/common/resize.hh>

#include <dune/hpdg/common/dynamicbvector.hh>

namespace Dune {
namespace Solvers {
namespace Impl {

// Specialization for DynamicBlockVector, which does not have a resize() method.
template<class K, class Value>
void resizeInitialize(Dune::HPDG::DynamicBlockVector<K>& x, const Dune::HPDG::DynamicBlockVector<K>& y, const Value& value)
{
  x=y;
  x=value;
}

} // namespace Impl

/**
 * \brief Resize and initialization vector to match size of given vector
 *
 * \param x Vector to resize
 * \param y Model for resizing
 * \param value Value to use for initialization
 *
 * This will resize the given vector x to match
 * the size of the given vector and assign the given
 * value to it.
 */
template<class K, class Value>
void resizeInitialize(HPDG::DynamicBlockVector<K>& x, const HPDG::DynamicBlockVector<K>& y, Value&& value)
{
  Impl::resizeInitialize(x, y, value);
}

/**
 * \brief Resize and initialization vector to match size of given vector
 *
 * \param x Vector to resize
 * \param y Model for resizing
 *
 * This will resize the given vector x to match
 * the size of the given vector and initialize
 * all entries with zero.
 */
template<class K>
void resizeInitializeZero(HPDG::DynamicBlockVector<K>& x, const HPDG::DynamicBlockVector<K>& y)
{
  resizeInitialize(x, y, 0);
}
} // end namespace Solvers
} // end namespace Dune

#endif // DUNE_HPDG_COMMON_RESIZE_HH
