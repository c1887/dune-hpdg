#pragma once

#include <dune/hpdg/common/mappedcache.hh>
#include <dune/hpdg/common/indexedcache.hh>
#include <dune/common/std/optional.hh>
#include <unordered_map>
#include <vector>

namespace Dune {
namespace HPDG {
namespace Impl {
  template<typename LFE>
  class CachedLocalBasis {
    using LB = typename LFE::Traits::LocalBasisType;
    using Domain = typename LB::Traits::DomainType;
    using Jac = typename LB::Traits::JacobianType;
    using Range = typename LB::Traits::RangeType;

    using ArrayDomain = std::array<typename Domain::block_type, Domain::dimension>; // wrap fieldvector into array

    public:
    CachedLocalBasis(const LFE* lfe):
      lfe_(lfe),
      jacobians_(
          [lfe](const auto& pos) {
            std::vector<Jac> out;
            lfe->localBasis().evaluateJacobian(reinterpret_cast<const Domain&>(pos), out);
            return out;
          }
          ),
      functions_(
          [lfe](const auto& pos) {
            std::vector<Range> out;
            lfe->localBasis().evaluateFunction(reinterpret_cast<const Domain&>(pos), out);
            return out;
          }
          )
     {}
    auto size() const {
      return lfe_->localBasis().size();
    }

    auto order() const {
      return lfe_->localBasis().order();
    }

    const auto& localFE() const {
      return *lfe_;
    }

    void evaluateJacobian(const Domain& in, std::vector<Jac>& out) const {
      out = jacobians_.value(reinterpret_cast<const ArrayDomain&>(in));
    }

    void evaluateFunction(const Domain& in, std::vector<Range>& out) const {
      out = functions_.value(reinterpret_cast<const ArrayDomain&>(in));
    }

    private:
    const LFE* lfe_ = nullptr;
    /* Yes, using doubles as map keys is stupid.
     * But they should come from quadrature rules and therefore be
     * at least consistent all the time. Maybe, find a better solution later TODO*/
    mutable MappedCache<std::vector<Jac>, ArrayDomain> jacobians_;
    mutable MappedCache<std::vector<Range>, ArrayDomain> functions_;
  };
}

/** This is a wrapper around a local finite element such that
 * the computations of evaluateJacobian and evaluateFunction are cached.
 * This is particularly useful for higher orders, where these are expensive.
 *
 * Note that this cache expects that the local finite elements that are inserted have
 * lifetimes as long as this cache is used. No LFE is copied.
 * In particular, for dune-functions bases, this probably means that your localView
 * should live long enough, as it typically holds the FE cache.
 *
 * Also, we differentiate the LFE only by their orders. More complex caches for other situations
 * could be implemented, of course.
 */
template<typename LFE>
class AssemblyCache {
  public:

  using Traits = typename LFE::Traits;

  /** Bind to a local finite element */
  void bind(const LFE* lfe) {
    current_ = &cache_.value(lfe->localBasis().order(), [&](auto&&) {
        return Impl::CachedLocalBasis<LFE>(lfe);
      });
  }

  /** Get a wrapper of the localBasis of the current LFE.
   *
   * This will mostly look like the actual localBasis but
   * the evaluate of functions and jacobians will be cached and thus
   * only have to be computed once per point in the domain.
   */
  const auto& localBasis() const {
    assert(current_ != nullptr);
    return *current_;
  }

  /** Return the type of the current bound LFE */
  auto type() const {
    return current_->localFE().type();
  }


  private:
    /** A vector that holds the cached LFE.
     *
     * The 10 is a default size. If one attempts to store a higher order, this will automatically resize
     */
    mutable IndexedCache<Impl::CachedLocalBasis<LFE>> cache_ = IndexedCache<Impl::CachedLocalBasis<LFE>>(10);

    /** Pointer to the current state*/
    Impl::CachedLocalBasis<LFE>* current_ = nullptr;
};

}
}
