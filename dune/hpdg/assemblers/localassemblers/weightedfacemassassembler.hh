#ifndef DUNE_HPDG_LOCALASSEMBLERS_WEIGHTED_FACE_MASS
#define DUNE_HPDG_LOCALASSEMBLERS_WEIGHTED_FACE_MASS
#include <algorithm>
#include <dune/common/fvector.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

namespace Dune::HPDG {

/**
 * @brief Assemble the weighted term \int_e [[w]]{\nabla w}}{phi_i phi_j} on a
 * face e.
 *
 * @param edge
 * @param matrices 2x2 set of matrices which (see
 * dunefunctionsoperatorassembler.hh in fufem)
 * @param insideLocalView
 * @param outsideLocalView
 * @param weight
 * @param gradWeight
 */
template<typename Edge,
         typename MC,
         typename LV,
         typename Weight,
         typename DWeight>
void
assembleWeightedFaceMass(const Edge& edge,
                         MC& matrices,
                         LV&& insideLocalView,
                         LV&& outsideLocalView,
                         Weight&& weight,
                         DWeight&& gradWeight,
                         double penalty)
{

  matrices = 0.0;


  if (edge.boundary())
    return; // assuming Neumann data

  const auto& outside_element = edge.outside();
  const auto& inside_element = edge.inside();
  auto insideLF = localFunction(weight);
  insideLF.bind(inside_element);
  auto outsideLF = localFunction(weight);
  outsideLF.bind(outside_element);

  auto insideGradLF = localFunction(gradWeight);
  insideGradLF.bind(inside_element);
  auto outsideGradLF = localFunction(gradWeight);
  outsideGradLF.bind(outside_element);

  const auto& iFE = insideLocalView.tree().finiteElement();
  const auto& oFE = outsideLocalView.tree().finiteElement();
  using RangeType =
    typename std::decay_t<decltype(iFE)>::Traits::LocalBasisType::Traits::RangeType;

  const auto maxOrder = std::max(iFE.localBasis().order(), oFE.localBasis().order());

  const auto genQuadKey = [&]() {
    auto key = QuadratureRuleKey(edge.type(), maxOrder)
                 .square()
                 .square(); // TODO ist das genug/zu viel?
    return key;
  };
  const auto key = genQuadKey();

  const auto& quad = QuadratureRuleCache<double, 1>::rule(key);

  // store values of shape functions
  std::vector<RangeType> insideValues(iFE.localBasis().size());
  std::vector<RangeType> outsideValues(oFE.localBasis().size());

  const auto& geo = edge.geometry();

  for (size_t pt = 0; pt < quad.size(); ++pt) {
    const auto& quadPos = quad[pt].position();

    const auto integrationElement = geo.integrationElement(quadPos);
    // evaluate basis functions
    iFE.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos),
                                      insideValues);
    oFE.localBasis().evaluateFunction(edge.geometryInOutside().global(quadPos),
                                      outsideValues);
    const auto jump = insideLF(edge.geometryInInside().global(quadPos)) - outsideLF(edge.geometryInOutside().global(quadPos));

    if (jump == 0.0)
      continue;
    const auto avg = 0.5 * ((insideGradLF(edge.geometryInInside().global(quadPos)) + outsideGradLF(edge.geometryInOutside().global(quadPos))) * edge.centerUnitOuterNormal());
    if (avg == 0.0)
      continue;

    const auto factor = integrationElement * (-2.0 * jump * avg + penalty/edge.geometry().volume() * jump * jump);
    // if (factor < 0.0)
    //   std::cout << "Warning: Penalty might be too small!\n";

    for (size_t i = 0; i < iFE.localBasis().size(); ++i) {
      for (size_t j = 0; j < iFE.localBasis().size(); ++j) {
        matrices[0][0][i][j] += factor * 0.5 * insideValues[i] * insideValues[j];
      }
      for (size_t j = 0; j < oFE.localBasis().size(); ++j) {
        matrices[0][1][i][j] += factor * 0.5 * insideValues[i] * outsideValues[j];
      }
    }

    for (size_t i = 0; i < oFE.localBasis().size(); ++i) {
      for (size_t j = 0; j < iFE.localBasis().size(); ++j) {
        matrices[1][0][i][j] += factor * 0.5 * outsideValues[i] * insideValues[j];
      }
      for (size_t j = 0; j < oFE.localBasis().size(); ++j) {
        matrices[1][1][i][j] += factor * 0.5 * outsideValues[i] * outsideValues[j];
      }
    }
  }
}

}
#endif // DUNE_HPDG_LOCALASSEMBLERS_WEIGHTED_FACE_MASS