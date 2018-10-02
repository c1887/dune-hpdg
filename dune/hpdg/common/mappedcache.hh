// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <map>
#include <functional>

namespace Dune {
namespace HPDG {

/** \brief A general-purpose cache that uses a std::map
 * as the underlying data structure. Every lookup will hence have the same complexity
 * as for a map.
 *
 * The user supplies a function with range T and domain IndexType (could be for example
 * a std::function, a lambda, or a classic function pointer.
 *
 * If the result for a requested index is not yet available,
 * the entry will be generated via the supplied function.
 *
 */
template<typename T, typename IndexType=size_t>
class MappedCache {

  public:

    using mapped_type = T;
    using key_type = IndexType;

    template<typename F>
    MappedCache(F&& generator_function) :
      generator_(std::forward<F>(generator_function))
    {}

    /** Return (and generate, if necessary) the cached
     * value for the index idx */
    T& value(const IndexType& idx)
    {
      auto it = cache_.find(idx);
      if (it == std::end(cache_)) {
        cache_[idx] = generator_(idx);
        return cache_[idx];
      }

      return it->second;
    }

    /** Return (and generate, if necessary) the cached
     * value for the index idx */
    const T& value(const IndexType& idx) const
    {
      auto it = cache_.find(idx);
      if (it == std::end(cache_)) {
        cache_[idx] = generator_(idx);
        return cache_[idx];
      }

      return it->second;
    }

    T& operator[](const IndexType& idx) {
      return value(idx);
    }

    const T& operator[](const IndexType& idx) const {
      return value(idx);
    }

  private:
    std::function<T(IndexType)> generator_;
    mutable std::map<IndexType, T> cache_;
};
}
}
