#ifndef DUNE_HPDG_BUILDINGBLOCKS_ESTIMATING_HH
#define DUNE_HPDG_BUILDINGBLOCKS_ESTIMATING_HH
#include <algorithm>
#include <cmath>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdglocalnorm.hh>
#include <dune/hpdg/matrix-free/operator.hh>

namespace Dune::HPDG::BuildingBlocks {
template<typename GridView, typename Vector>
auto
ipdgLocalNorm(const Functions::DynamicDGQkGLBlockBasis<GridView>& basis,
              const Vector& coeffs,
              double penalty)
{
  using Basis = std::decay_t<decltype(basis)>;

  auto errors = Vector(coeffs.size(), 1);
  errors.update();
  errors = 0.;

  std::vector<double> localErrors(coeffs.size());
  auto laplace_diag =
    Dune::Fufem::MatrixFree::IPDGLocalNorm<Vector, GridView, Basis>(
      basis, penalty, false);
  auto op =
    Dune::Fufem::MatrixFree::Operator<Vector, GridView, decltype(laplace_diag)>(
      basis.gridView(), laplace_diag);
  op.apply(coeffs, errors);

  // for (auto&& val : errors)
  //   error += val[0];
  // error = std::sqrt(error);

  // local errors:
  for (std::size_t j = 0; j < localErrors.size(); j++) {
    localErrors[j] = errors[j][0]; // error integral on the j-th element
  }

  auto isnancheck = [](const auto& v) {
    for (const auto& entry : v)
      if (std::isnan(entry))
        return false;
    return true;
  };
  assert(isnancheck(localErrors));
  
  return localErrors;
}

double
globalError(const std::vector<double>& errors)
{
  using namespace std;
  auto ret = accumulate(begin(errors), end(errors), 0.0);
  return sqrt(ret);
}
}
#endif // DUNE_HPDG_BUILDINGBLOCKS_ESTIMATING_HH