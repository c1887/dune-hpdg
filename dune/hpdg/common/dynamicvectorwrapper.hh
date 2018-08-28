// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_HPDGPACEBASES_HIERARCHICVECTORWRAPPER_HH
#define DUNE_HPDG_HPDGPACEBASES_HIERARCHICVECTORWRAPPER_HH

#include <dune/common/concept.hh>
#include <dune/common/hybridutilities.hh>

#include <dune/typetree/utility.hh>

#include <dune/functions/common/indexaccess.hh>
#include <dune/functions/common/utility.hh>
#include <dune/functions/common/type_traits.hh>
#include <dune/functions/functionspacebases/concepts.hh>


namespace Dune {
namespace HPDG {



namespace Imp {

  // Construct default coefficent type from vector and multiindex type
  // This requires that MultiIndex has a static size. Otherwise the
  // vector type itself is returned.
  template<class V, class MultiIndex>
  struct CoefficientType
  {
    template<class E, std::size_t size>
    struct DefaultCoefficientTypeHelper
    {
      using E0 = decltype(std::declval<E>()[Dune::TypeTree::Indices::_0]);
      using type = typename DefaultCoefficientTypeHelper<E0, size-1>::type;
    };

    template<class E>
    struct DefaultCoefficientTypeHelper<E, 0>
    {
      using type = E;
    };

    template<class MI,
      typename std::enable_if<Functions::HasStaticSize<MI>::value, int>::type = 0>
    static constexpr std::size_t getStaticSizeOrZero()
    {
      return Functions::StaticSize<MI>::value;
    }

    template<class MI,
      typename std::enable_if<not Functions::HasStaticSize<MI>::value, int>::type = 0>
    static constexpr std::size_t getStaticSizeOrZero()
    {
      return 0;
    }

    using type = typename DefaultCoefficientTypeHelper<V, getStaticSizeOrZero<MultiIndex>()>::type;
  };

} // namespace Imp



/**
 * \brief A wrapper providing multiindex access to vector entries
 *
 * The coefficient type should be a type such that the coefficients
 * entries for each global basis function can be cast to this type.
 * This is necessary because the wrapper cannot determine this type
 * automatically for multi-type containers and non-uniform indices.
 * The reason for this is, that the multi-index type will then be
 * dynamically sized such that the index depth cannot statically
 * be determined from the multi-indices. However, the compiler needs
 * a fixed termination criterion for instantiation of recursive
 * functions.
 *
 * If no coefficient type is given, the wrapper tries to determine
 * the coefficient type on its own assuming that the multi-indices
 * have fixed size.
 *
 * \tparam V Type of the raw wrapper vector
 * \tparam CO Coefficient type
 */
template<class V, class CO=void>
class HierarchicVectorWrapper
{
  template<class MultiIndex>
  using Coefficient = typename std::conditional< std::is_same<void,CO>::value and Functions::HasStaticSize<MultiIndex>::value,
            typename Imp::CoefficientType<V, MultiIndex>::type,
            CO
            >::type;


  using size_type = std::size_t;


public:

  using Vector = V;

  template<class MultiIndex>
  using Entry = Coefficient<MultiIndex>;

  HierarchicVectorWrapper(Vector& vector) :
    vector_(&vector)
  {}

  template<class SizeProvider>
  void resize(const SizeProvider& sizeProvider)
  {
    // empty
  }

  template<class MultiIndex>
  const Entry<MultiIndex>& operator[](const MultiIndex& index) const
  {
      return Functions::hybridMultiIndexAccess<const Entry<MultiIndex>&>(*vector_, index);
  }

  template<class MultiIndex>
  Entry<MultiIndex>& operator[](const MultiIndex& index)
  {
      return Functions::hybridMultiIndexAccess<Entry<MultiIndex>&>(*vector_, index);
  }

  template<class MultiIndex>
  const Entry<MultiIndex>& operator()(const MultiIndex& index) const
  {
      return (*this)[index];
  }

  template<class MultiIndex>
  Entry<MultiIndex>& operator()(const MultiIndex& index)
  {
      return (*this)[index];
  }

  const Vector& vector() const
  {
    return *vector_;
  }

  Vector& vector()
  {
    return *vector_;
  }

  template<typename E>
  HierarchicVectorWrapper& operator=(const E& e) {
    *vector_ = e;
    return *this;
  }
private:

  Vector* vector_;
};




template<class V>
HierarchicVectorWrapper< V > hierarchicVector(V& v)
{
  return HierarchicVectorWrapper<V>(v);
}


} // namespace Dune::HPDG
} // namespace Dune


#endif // DUNE_HPDG_HPDGPACEBASES_HIERARCHICVECTORWRAPPER_HH
