// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUNCTIONS_FUNCTIONSPACEBASES_GRIDADAPTOR_HH
#define DUNE_FUNCTIONS_FUNCTIONSPACEBASES_GRIDADAPTOR_HH
#include <vector>
#include <memory>

#include <dune/common/exceptions.hh>
#include <dune/grid/utility/persistentcontainer.hh>
namespace Dune {
namespace Functions {
namespace Impl {

  template<typename B>
  struct PersistentBasis;

  template<class B, class BE>
  struct LocalCoarseFunction {

    LocalCoarseFunction(const PersistentBasis<B>& pb, const BE& coeff, bool allowRebind=true) :
      persistentBasis_(pb),
      coefficients_(coeff),
      allowRebind_(allowRebind) {}

    using Element = typename B::GridView::template Codim<0>::Entity;
    template<class X, class Y>
    void evaluate (X x, Y& y) const {

      auto values = std::vector<Y>();
      Element e = *element_;

      // iterate through father elements until one is found where our basis is known
      while (persistentBasis_.isNewElement(e)) {
        if (not e.hasFather())
          DUNE_THROW(Dune::Exception, "Elements fathers are unknown to persistent basis");
        x = e.geometryInFather().global(x);
        e = e.father();
      }

      auto lv = persistentBasis_.localView(e);
      auto li = persistentBasis_.localIndexSet(e);
      // some bases do not like if you bind you're views again. Others, on the other hand, need this.
      if (allowRebind_)
        lv.bind(e);

      // note: the local index set `li` MUST NOT be bound to `lv` as indices might be recalculated
      // given the NEW gridview.

      lv.tree().finiteElement().localBasis().evaluateFunction(x,values);
      y=0;
      for (size_t i = 0; i < values.size(); i++) {
        y+=coefficients_[li.index(i)]*values[i];
      }

    }

    void bind(const Element& e) {
      element_ = &e;
    }
    private:
    const Element* element_;
    const PersistentBasis<B>& persistentBasis_;
    const BE& coefficients_;
    bool allowRebind_;
  };


  template<typename B>
  struct PersistentBasis {
    using Basis = B;
    PersistentBasis(const Basis&b) :
      basis_(&b),
      stateMap_(basis_->gridView().grid(), 0, 0) // first 0 is codim, second 0 is the default value
    {
      save();
    }

    /** \brief Save state of the basis BEFORE the basis or the grid will be changed */
    void save() {
      size_t location = 1;
      const auto& gv = basis_->gridView();
      data_.resize(gv.size(0)+1); // + 1 as index 0 encodes new elements and hence will not be used.

      auto localView = basis_->localView();
      auto localIndexSet = basis_->localIndexSet();
      for (const auto& e: elements(gv)) {
        localView.bind(e);
        localIndexSet.bind(localView);
        data_[location]=std::make_unique<State>(std::make_pair(localView, localIndexSet));
        stateMap_[e]=location;
        location++;
      }
    }

    /** \brief Return the local index set as it was when the given element has been part of the
     * old grid view.
     *
     * Do not bind this one again, as it would give an index set that is recalculated
     * with the new grid view!
     */
    template<typename Element>
    auto localIndexSet(const Element& e) const {
      if (stateMap_[e]==0)
        DUNE_THROW(Dune::Exception, "No state was saved for given element in PersistentBasis");

      return data_[stateMap_[e]]->second;
    }

    template<typename Element>
    auto localView(const Element& e) const {
      if (stateMap_[e]==0)
        DUNE_THROW(Dune::Exception, "No state was saved for given element in PersistentBasis");

      return data_[stateMap_[e]]->first;
    }

    /** \brief Check if a given element is new, i.e. not part of the saved basis */
    template<typename Element>
    auto isNewElement(const Element& e) const {
      return (stateMap_[e]==0);
    }

    // TODO: This name is misleading and suggest it does the same as save(), which is not true.
    // This function must be called when the original basis has been updated.
    // If you want to save the current state, use save().
    void update() {
      stateMap_.resize(0);
    }

    private:
    using State = std::pair<typename Basis::LocalView, typename Basis::LocalIndexSet>;
    using DataStorage = std::vector<std::unique_ptr<State>>;
    using ElementToState = PersistentContainer<typename Basis::GridView::Grid, size_t>;

    const Basis* basis_;
    ElementToState stateMap_;
    DataStorage data_;
  };
}

  /* Adaptor class to transfer a coefficient vector of a given basis
   * to the same basis on an updated GridView.
   * Typical workflow would be:
   *
   * auto basis = ...;
   * // compute some coefficients x of a function defined on basis
   * Vector x = ...;
   *
   * // create a GridAdaptor object to save basis state
   * auto gridadaptor = GridAdaptor<Basis>(basis);
   * // adapt grid
   * grid.adapt();
   * // update GridView in basis
   * basis.update(gv);
   *
   * // now, your coefficients x are invalid on the updated basis.
   * // create new ones:
   * Vector xNew;
   * xNew.resize(basis.size());
   *
   * // transfer to updated basis
   * gridadaptor.adapt(x, xNew);
   *
   */
  template<class B>
  struct GridAdaptor {
    GridAdaptor(const B& basis, bool allowRebind=true) :
      basis_(basis),
      persistentBasis_(basis),
      allowRebind_(allowRebind)
    {}

    /* \brief Set if re-binding of old elements is allowed
     *
     * In the process of interpolating on coarser elements
     * from before adaption, the corresponding LocalViews can
     * be bound again to the elements. For most cases, this will be
     * necessary and should be allowed. Custom basis implementations, however,
     * might not behave as expected, hence you can disallow re-binding.
     */
    void setAllowRebind(bool allowedFlag) {
      allowRebind_=allowedFlag;
    }

    /* \brief Adapt coefficients to new basis.
     *
     * Note, the new coeffcient vector must have already the right size.
     */
    template<class V1, class V2>
    void adapt(const V1& coarseCoeff, V2&& fineCoeff) {
      // tell the saved basis the gridview (might) has been updated
      persistentBasis_.update();
      // create a local coarse function that works with the persistent basis
      Impl::LocalCoarseFunction<B, V1> lcf(persistentBasis_, coarseCoeff, allowRebind_);

      auto fineView = basis_.localView();
      auto fineIndexSet = basis_.localIndexSet();
      using Range = typename std::decay_t<decltype(fineView.tree().finiteElement().localBasis())>::Traits::RangeType;
      std::vector<Range> values;
      for (const auto& e: elements(basis_.gridView())) {
        fineView.bind(e);
        fineIndexSet.bind(fineView);
        lcf.bind(e);
        fineView.tree().finiteElement().localInterpolation().interpolate(lcf, values);
        for (size_t i = 0; i < fineView.tree().finiteElement().size(); i++) {
          fineCoeff[fineIndexSet.index(i)] = values[i];
        }
      }
    }

    /* \brief Save current state of the basis */
    void save() {
      persistentBasis_.update();
      persistentBasis_.save();
    }

    private:
      const B& basis_;
      Impl::PersistentBasis<B> persistentBasis_;
      bool allowRebind_;
  };
}
}
#endif
