#ifndef DUNE_HPDG_ESTIMATORS_UTILITY_HH
#define DUNE_HPDG_ESTIMATORS_UTILITY_HH
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#if HAVE_MPI
#include <mpi.h>
#endif
namespace Dune {
namespace HPDG {
  /** \brief Returns the value at a given quantile of a vector
   *
   *  This might be inaccurate if the vector has many equal elements.
   */
  template<typename K>
  K quantile(const std::vector<K>& vec, double quant) {
    auto vec_sorted = vec;
    std::sort(std::begin(vec_sorted), std::end(vec_sorted));

    // the following is not 100% correct, but it serves it's purpose for us
    auto quantile_value = vec_sorted[(size_t) (quant*vec_sorted.size())];

    return quantile_value;
  }

  /** \brief Returns a value v such that such the sum over all
   * elements >=v in vec will be at least a given fraction of the total sum.
   *
   * All elements in vec are expected to be nonnegative.
   */
  template<typename K>
  K fraction(const std::vector<K>& vec, double frac) {
    auto vec_sorted = vec;
    std::sort(std::begin(vec_sorted), std::end(vec_sorted));

    auto fullSum = std::accumulate(std::begin(vec_sorted), std::end(vec_sorted), K(0));

    auto thresh = frac*fullSum;

    if (thresh==0.0)
      return K(0.0);

    for(const auto& val : vec_sorted) {
      if(val<0)
        DUNE_THROW(Dune::Exception, "Negative value while calculating fraction of values");

      fullSum -= val;
      if (fullSum <= thresh)
        return val;
    }

    // logically, we should never end here
    // However, compiler doesn't end a function that does not end with a return
    assert(false);
    return K(0.0);
  }

  /** \brief Marks an element e for refinement and gurantees that the level difference
   * between two neighboring elements is not greater than a given value.
   *
   * \warning Note that this might have bad complexity, I never checked.
   */
  template<class G, class E>
  void marker(G& grid, const E& e, int max_diff) {

    // mark the element e
    grid.mark(1, e);

    // check if neighbors are too coarse, i.e. there level difference is too big. If so, refine them too
    auto level = e.level()+1; // +1 from the refinement above
    for (auto& is : intersections(grid.leafGridView(), e)) {
      if (!is.neighbor())
        continue;
      auto outside = is.outside();
      auto outsidelevel = outside.level() + grid.getMark(outside);
      if (level - outsidelevel > max_diff) {
        if (not grid.getMark(outside))
          marker(grid, outside, max_diff);
      }
    }
  }
#if HAVE_MPI
  template<typename K, typename G>
  auto globalFraction (const std::vector<K>& errors, const G& grid, double frac) {
    double border;
    int rank = grid.comm().rank();
    int n_procs = grid.comm().size();

    int local_size = errors.size();

    std::vector<double> e;
    // gather all errors on root

    std::vector<int> sizes(n_procs);
    std::vector<int> disps(n_procs,0);
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, grid.comm());
    if(rank==0) {
      for(int i=1; i < n_procs; i++) {
        disps[i] = disps[i-1]+sizes[i-1];
      }
      auto full_size = std::accumulate(sizes.begin(), sizes.end(), 0);
      e.resize(full_size);
    }

    MPI_Gatherv(errors.data(), local_size, MPI_DOUBLE,
        e.data(), sizes.data(), disps.data(), MPI_DOUBLE, 0, grid.comm());
    /*
    if(rank==0) {
      for(const auto& val : e ) {
        if(val <0.)
          DUNE_THROW(Dune::Exception, "omgwtf");
      }
    }
    */

    if(rank==0)
      border = fraction(e, frac);
    MPI_Bcast(&border, 1, MPI_DOUBLE, 0, grid.comm());

    return border;
  }
#endif

}} // end namespace Dune::HPDG

#endif//DUNE_HPDG_ESTIMATORS_UTILITY_HH
