// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <vector>
#include <dune/hpdg/functionspacebases/persistentgridview.hh>
#include <dune/grid/common/mcmgmapper.hh>

namespace Dune {
namespace HPDG {
  /** Utility class to transfer data from a PersistentGridView to a
   * elements of the refined GridView.
   */
  template<typename BaseGridView, typename DataType>
  class PersistentGridViewDataTransfer{
    public:

      using PGV = Dune::Functions::Experimental::PersistentGridView<BaseGridView>;
      PersistentGridViewDataTransfer(const PGV& pgv) :
        pgv_(pgv),
        mapper_(pgv_, mcmgElementLayout()),
        data_(mapper_.size()){}

      /** Access the data stored with respect to an element.
       * If the given element is not found in the
       * PersistentGridView, it's father element will be considered.
       */
      template<typename E>
      DataType& operator[](const E& element) {
        if (!pgv_.contains(element)) {
          return (*this)[element.father()];
        }
        return data_[mapper_.index(element)];
      }

      /** Read the data stored with respect to an element.
       * If the given element is not found in the
       * PersistentGridView, it's father element will be considered.
       */
      template<typename E>
      const DataType& operator[](const E& element) const {
        if (!pgv_.contains(element)) {
          return (*this)[element.father()];
        }
        return data_[mapper_.index(element)];
      }

    private:
      const PGV& pgv_;
      MultipleCodimMultipleGeomTypeMapper<PGV> mapper_;
      std::vector<DataType> data_;
  };
}}
