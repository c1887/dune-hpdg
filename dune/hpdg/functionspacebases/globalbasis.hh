// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_FUNCTIONSPACEBASES_GLOBALBASIS_HH
#define DUNE_HPDG_FUNCTIONSPACEBASES_GLOBALBASIS_HH

#include <dune/functions/functionspacebases/defaultglobalbasis.hh>

namespace Dune {
namespace Functions {

/* This is just a very simple addition to the
 * DefaultGlobalBasis
 * that allows to access the NodeFactory in a non-const way
 */
template<class NF>
class GlobalBasis : public DefaultGlobalBasis<NF>
{
  using Base = DefaultGlobalBasis<NF>;

  public:
  template<class... T,
    disableCopyMove<GlobalBasis, T...> = 0,
    enableIfConstructible<typename Base::NodeFactory, T...> = 0>
  GlobalBasis(T&&... t) :
    Base(std::forward<T>(t)...) {}

  /** Access NodeFactory in a non-const way */
  auto& nodeFactory()
  {
    return this->nodeFactory_;
  }

};


} // end namespace Functions
} // end namespace Dune


#endif // DUNE_HPDG_FUNCTIONSPACEBASES_GLOBALBASIS_HH
