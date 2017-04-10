#ifndef DUNE_HPDG_COMMON_ARRAYTOTUPLE_HH
#define DUNE_HPDG_COMMON_ARRAYTOTUPLE_HH
#include <array>
#include <tuple>

// Convert array into a tuple (shamelessly stolen from cppreference.com)
namespace Dune {
namespace HPDG {

namespace Impl {
template <typename Array, std::size_t... I>
decltype(auto) a2t_impl(const Array &a, std::index_sequence<I...>) {
  return std::make_tuple(a[I]...);
}
}

template <typename T, std::size_t N,
          typename Indices = std::make_index_sequence<N>>
decltype(auto) arrayToTuple(const std::array<T, N> &a) {
  return Impl::a2t_impl(a, Indices());
}
}
}
#endif
