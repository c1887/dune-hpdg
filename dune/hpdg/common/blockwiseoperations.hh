#ifndef DUNE_HPDG_BLOCKWISE_OPERATIONS_HH
#define DUNE_HPDG_BLOCKWISE_OPERATIONS_HH

#include <cstddef>

namespace Dune {
  namespace HPDG {
    template<class Block, class Vector>
    Vector scaleByTransposedBlock(const Block& block, const Vector& vector) {
      // some size tests
      assert(vector.size()>0);
      assert(block.N()==vector[0].size());

      auto scaledVector = vector;
      for (std::size_t i=0; i< vector.size(); i++)
        block.mtv(vector[i], scaledVector[i]);

      return scaledVector;
    }

    template<class Block, class V, class CV>
    void scaleByTransposedBlock(const Block& block, const V& vector, CV& scaledVector){
      // some size tests
      assert(vector.size()>0);
      assert(block.N()==vector[0].size());

      scaledVector.resize(vector.size());
      scaledVector=0;
      for (std::size_t i=0; i< vector.size(); i++)
        block.mtv(vector[i], scaledVector[i]);
    }

    template<class Block, class V, class CV>
    void scaleByBlock(const Block& block, const V& vector, CV& scaledVector){
      // some size tests
      assert(vector.size()>0);
      assert(block.M()==vector[0].size());

      scaledVector.resize(vector.size());
      for (std::size_t i=0; i< vector.size(); i++)
        block.mv(vector[i], scaledVector[i]);
    }

    template<class Block, class Vector>
    Vector scaleByBlock(const Block& block, const Vector& vector) {
      // some size tests
      assert(block.M()==Vector::block_type::dimension);

      auto scaledVector = vector;
      for (std::size_t i=0; i< vector.size(); i++)
        block.mv(vector[i], scaledVector[i]);

      return scaledVector;
    }

  }
}

#endif
