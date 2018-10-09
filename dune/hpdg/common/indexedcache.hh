// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <vector>
#include <functional>

#include <dune/common/std/optional.hh>
#include <dune/common/exceptions.hh>

namespace Dune {
namespace HPDG {

/** \brief A general-purpose cache that uses a std::vector
 * as the underlying data structure. 
 * Hence, only std::size_t or other integer types are valid index types.
 * For more general index types, consider HPDG::MappedCache.
 *
 * The user supplies a function with range T either in form as a std::function
 * or directly when asking for values.
 *
 * If the result for a requested index is not yet available,
 * the entry will be generated via the supplied function.
 *
 */
template<typename T>
class IndexedCache {
  using IndexType = std::size_t;

  public:

    using mapped_type = T;
    using key_type = IndexType;

    IndexedCache(const std::function<T(IndexType)>& generator_function, std::size_t expectedCacheSize=0) :
      generator_(generator_function),
      cache_(expectedCacheSize)
    {}

    IndexedCache() = default;

    IndexedCache(std::size_t expectedCacheSize):
      cache_(expectedCacheSize)
    {}

    /** Return (and generate, if necessary) the cached
     * value for the index idx using a user-supplied
     * generator function.
     */
    template<class F>
    T& value(const IndexType& idx, F&& generator)
    {
      if (idx>=cache_.size())
        cache_.resize(idx+1);
      if (not cache_[idx]) {
        cache_[idx] = generator(idx);
      }
      return *cache_[idx];
    }

    /** Return (and generate, if necessary) the cached
     * value for the index idx using a user-supplied
     * generator function.
     */
    template<class F>
    const T& value(const IndexType& idx, F&& generator) const
    {
      if (idx>=cache_.size())
        cache_.resize(idx+1);
      if (not cache_[idx]) {
        cache_[idx] = generator(idx);
      }
      return *cache_[idx];
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

      if (idx >= cache_.size() or not cache_[idx])
        DUNE_THROW(Dune::Exception, "Index not yet cached and no generator function known");

      return *cache_[idx];
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

      if (idx >= cache_.size() or not cache_[idx])
        DUNE_THROW(Dune::Exception, "Index not yet cached and no generator function known");

      return *cache_[idx];
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
    mutable std::vector<Std::optional<T>> cache_;
};
}
}
