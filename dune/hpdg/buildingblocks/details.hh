#ifndef DUNE_HPDG_BUILDINGBLOCKS_DETAILS
#define DUNE_HPDG_BUILDINGBLOCKS_DETAILS

#include <algorithm>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/dynamicbvector_specializations.hh>
#include <dune/solvers/common/defaultbitvector.hh>

namespace Dune::HPDG::BuildingBlocks {
namespace Detail {

/**
 * @brief Returns the maximal Degree of a dune-hpdg style DG basis.
 *
 * @tparam Basis Should be Dune::Functions::DynamicDGQkGLBlockBasis or similar.
 * @param basis The basis
 * @return auto The maximal degree
 */
template<typename Basis>
auto
maxDegree(const Basis& basis)
{
  const auto& degrees = basis.preBasis().degreeMap();
  return *std::max_element(degrees.begin(), degrees.end());
}

/**
 * @brief Returns a bitvector which is false everywhere and has the same
 * dimensions as the input vector `model_vector`.
 *
 * @tparam Vector  This should be a DynamicBlockVector of some sort.
 * @param model_vector This vector serves as model for how the output bit vector
 * should be sized(number and sizes of the blocks)
 * @return auto a bitvector which is false everywhere
 */
template<typename Vector>
auto
allFalseBitVector(const Vector& model_vector)
{
  using BitSetVector = Solvers::DefaultBitVector_t<Vector>;
  auto dummyBits = BitSetVector(model_vector.size());
  for (size_t i = 0; i < model_vector.size(); i++) {
    dummyBits[i].resize(model_vector[i].size(), false);
  }
  return dummyBits;
}

} // end namespace Detail
} // end namespace Dune::HPDG::BuildingBlocks
#endif
