// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <cstddef>
#include <random>
namespace Dune {
namespace HPDG {

  // fills a vector with normal distributed random values
  template<class V>
  void fillVectorRandomly(V& vector, unsigned int seed = 1887) {

    std::mt19937 mt;
    mt.seed(seed);

    std::normal_distribution<> gauss{0,1};

    for(std::size_t i = 0; i < vector.size(); i++) {
      for(std::size_t j = 0; j < vector[i].N(); j++) {
        vector[i][j]=gauss(mt);
      }
    }
  }
}
}

