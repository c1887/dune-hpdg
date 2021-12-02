#ifndef DUNE_HPDG_CONSTANT_FUNCTION
#define DUNE_HPDG_CONSTANT_FUNCTION
namespace Dune::HPDG {
template<typename R>
class ConstantGridViewFunction
{
public:
  using Range = R;

  ConstantGridViewFunction(const Range& r)
    : value_(r)
  {}

  template<typename Any>
  constexpr Range operator()(const Any&) const
  {
    return value_;
  }

  /** Dummy to pass the interace */
  template<typename Element>
  void bind(const Element&) const
  {}

  friend auto localFunction(const ConstantGridViewFunction& in) { return in; }

private:
  const Range value_;
};
}
#endif