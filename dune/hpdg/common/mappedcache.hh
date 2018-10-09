// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <map>
#include <functional>

#include <dune/common/exceptions.hh>

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

    MappedCache(const std::function<T(IndexType)>& generator_function) :
      generator_(generator_function)
    {}

    MappedCache() = default;

    /** Return (and generate, if necessary) the cached
     * value for the index idx using a user-supplied
     * generator function.
     */
    template<class F>
    T& value(const IndexType& idx, F&& generator)
    {
      auto it = cache_.find(idx);
      if (it == std::end(cache_)) {
        cache_[idx] = generator(idx);
        return cache_[idx];
      }

      return it->second;
    }

    /** Return (and generate, if necessary) the cached
     * value for the index idx using a user-supplied
     * generator function.
     */
    template<class F>
    const T& value(const IndexType& idx, F&& generator) const
    {
      auto it = cache_.find(idx);
      if (it == std::end(cache_)) {
        cache_[idx] = generator(idx);
        return cache_[idx];
      }

      return it->second;
    }

    /** Return cached value at idx.
     *
     *
     * Note that calling this with an idx
     * that has not yet been called is only valid
     * if the generator function has been set before.
     *
     */
    const T& value(const IndexType& idx) const {
      if (generator_)
        return value(idx, generator_);

      auto it = cache_.find(idx);
      if (it == std::end(cache_))
        DUNE_THROW(Dune::Exception, "Index not yet cached and no generator function known");

      return it->second;
    }

    /** Return cached value at idx.
     *
     *
     * Note that calling this with an idx
     * that has not yet been called is only valid
     * if the generator function has been set before.
     *
     */
    T& value(const IndexType& idx) {
      if (generator_)
        return value(idx, generator_);

      auto it = cache_.find(idx);
      if (it == std::end(cache_))
        DUNE_THROW(Dune::Exception, "Index not yet cached and no generator function known");

      return it->second;
    }
    /** Return cached value at idx.
     *
     *
     * Note that calling this with an idx
     * that has not yet been called is only valid
     * if the generator function has been set before.
     *
     */
    T& operator[](const IndexType& idx) {
      return value(idx);
    }

    /** Return cached value at idx.
     *
     *
     * Note that calling this with an idx
     * that has not yet been called is only valid
     * if the generator function has been set before.
     *
     */
    const T& operator[](const IndexType& idx) const {
      return value(idx);
    }

    template<typename F>
    void setGenerator(F&& generator) {
      generator_ = std::forward<F>(generator);
    }

  private:
    std::function<T(IndexType)> generator_;
    mutable std::map<IndexType, T> cache_;
};
}
}
