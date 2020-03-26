#ifndef DUNE_HPDG_ESTIMATORS_SMOOTHNESSINDICATOR_HH
#define DUNE_HPDG_ESTIMATORS_SMOOTHNESSINDICATOR_HH

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/hpdg/localfunctions/lagrange/qkcache.hh>
#include <dune/functions/common/functionfromcallable.hh>

namespace Dune {
  namespace HPDG {

    // class to estimate local smoothness based on the Legendre coefficients
    // of the given function. See
    // Houston, Süli "A note on the design of hp-adaptive finite elements
    // methods for elliptic partial differential equations"
    template<class GV, int dim, class field_type=double>
    struct SmoothnessIndicator {
      template<class E, class LF>
      auto computeSmoothness(const E& element, LF&& localFunction, int k) {
        // get Legendre FE:
        const auto& fe = feCache_.get(k);
        auto interpolationValues = std::vector<field_type>();

        // interpolate:
        using LocalDomain = typename E::Geometry::LocalCoordinate;
        using FiniteElement = std::decay_t<decltype(fe)>;
        using FiniteElementRange = typename FiniteElement::Traits::LocalBasisType::Traits::RangeType;
        using FunctionBaseClass = typename Dune::LocalFiniteElementFunctionBase<FiniteElement>::type;
        using FunctionFromCallable = typename Dune::Functions::FunctionFromCallable<FiniteElementRange(LocalDomain), std::decay_t<LF>, FunctionBaseClass>;

        auto ffc = FunctionFromCallable(std::forward<LF>(localFunction));

        fe.localInterpolation().interpolate(ffc, interpolationValues);

        // set value to |log(|value|)|
        for (size_t i = 0; i < interpolationValues.size(); i++) {
          interpolationValues[i]=std::abs(std::log(std::abs(interpolationValues[i])));
        }

        // now, compute slope m of |log(|value[i]|)| = i*m +b via least squares
        auto slope=leastSquaresSlope(interpolationValues, k);

        // TODO: What if some values are zero? We get inf from the logarithms :(
        // for now, we treat this as being a sign for smoothness and hence return a low
        // value. However, this should be justified at some point.
        if (std::isnan(slope))
          return 0.0;
        return std::exp(-slope);
      }

      private:

      auto leastSquaresSlope(const std::vector<field_type>& values, int k) const {
        // compute mean of independent var.
        auto xmean = 0.0;
        for (int i =0; i < values.size(); i++) {
          auto mi = multiindex(i,k);
          auto sum=0;
          for (const auto& entry : mi) sum+=entry;
          xmean+=sum;
        }
        xmean/=values.size();

        // compute mean of dependent var.
        auto ymean=0.0;
        for (const auto& val: values) ymean+=val;
        ymean/= values.size();

        auto cov = 0.0;
        auto var = 0.0;
        for (size_t i = 0; i < values.size(); i++) {
          auto mi = multiindex(i, k);
          auto sum = 0;
          for (const auto& entry : mi) sum+=entry;
          cov += (values[i]-ymean)*(sum-xmean);
          var += (sum-xmean)*(sum-xmean);
        }
        return cov/var;
      }

      auto multiindex(int i, int k) const {
        Dune::FieldVector<int,dim> alpha;
        for (int j=0; j<dim; j++)
        {
          alpha[j] = i % (k+1);
          i = i/(k+1);
        }
        return alpha;
      }

      QkGaussLegendreVaryingOrderCache<field_type, field_type, dim, 14> feCache_;
    };
  }
}
#endif//DUNE_HPDG_ESTIMATORS_SMOOTHNESSINDICATOR_HH
