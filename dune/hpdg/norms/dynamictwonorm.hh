// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_NORMS_TWONORM_HH
#define DUNE_HPDG_NORMS_TWONORM_HH
#include <dune/solvers/norms/twonorm.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

// no proper namespaces as the dune-solvers one doesnt have these :(
//namespace Dune {
//namespace HPDG {

//! Wrapper around the two_norm() method of a DynamicBlockVector
template <class K>
class TwoNorm<Dune::HPDG::DynamicBlockVector<K>> : public Norm<Dune::HPDG::DynamicBlockVector<K>>
{
  using VectorType = Dune::HPDG::DynamicBlockVector<K>;

  public:

  virtual ~TwoNorm() {};

  //! Compute the norm of the given vector
  virtual double operator()(const VectorType& f) const
  {
    return f.two_norm();
  }

  //! Compute the square of the norm of the given vector
  virtual double normSquared(const VectorType& f) const
  {
    return f.two_norm2();
  }

  //! Compute the norm of the difference of two vectors
  virtual double diff(const VectorType& f1, const VectorType& f2) const
  {
    assert(f1.size() == f2.size());
    auto f1copy = f1;
    f1copy -= f2;
    auto r = f1copy.two_norm2();

    return std::sqrt(r);
  }

};

//}
//}
#endif//DUNE_HPDG_NORMS_TWONORM_HH
