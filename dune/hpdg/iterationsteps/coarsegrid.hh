#pragma once
#ifdef HAVE_MPI
#include <dune/hpdg/iterationsteps/parmgsetup.hh>
#include <dune/hpdg/parallel/communicationhpdg.hh>
#include <dune/parmg/std/numeric.hh>
#include <dune/istl/matrixindexset.hh>
#include <mpi.h>
namespace Dune {
namespace HPDG {
namespace MultigridSetup {
  class Rank0Collector {
    public:
    Rank0Collector(const MPI_Comm comm, const std::vector<int>& dofs, const std::vector<int>& owner):
      comm_(comm)
    {
      int size;
      MPI_Comm_rank(comm_, &rank);
      MPI_Comm_size(comm_, &size);

      if (rank == 0) {
        all_counts.resize(size);
        all_displs.resize(size+1);
      }

      local_count = dofs.size();
      MPI_Gather(
        &local_count, 1, MPI_INT,
        all_counts.data(), 1, MPI_INT,
        0, comm_
        );

      ParMG::Impl::exclusive_scan(all_counts.begin(), all_counts.end(), all_displs.begin(), 0);

      if (rank == 0) {
        all_displs.back() = all_displs[size-1]+all_counts.back(); // also set the last entry of the partial sums
        globalSize = all_displs.back();
        all_globalDof.resize(globalSize);
        all_owned.resize(globalSize);
      }

      MPI_Gatherv(
        (void*) dofs.data(), dofs.size(), MPI_INT,
        all_globalDof.data(), all_counts.data(), all_displs.data(), MPI_INT,
        0, comm_
        );

      auto owned = std::vector<char>(owner.size());

      for(std::size_t i = 0; i < owner.size(); i++) {
        owned[i] = owner[i]==rank;
      }

      // compute the global size by only summing owned elements
      int localOwnedSize = std::accumulate(owned.begin(), owned.end(), 0);
      MPI_Reduce(&localOwnedSize, &ownedSize_, 1, MPI_INT, MPI_SUM, 0, comm_);


      MPI_Gatherv(
        (void*)owned.data(), owned.size(), MPI_CHAR,
        all_owned.data(), all_counts.data(), all_displs.data(), MPI_CHAR,
        0, comm_
        );
    }

    template<typename V> // TODO: V is always DynamicBlockVector
    void restrictVector(const V& localInput, V& globalOutput) {
      // assume all blocks have same size:
      auto blocksize = localInput[0].size();
      if (rank==0) {
        globalOutput.setSize(ownedSize_);

        for(std::size_t i = 0; i < ownedSize_; i++) {
          globalOutput.blockRows(i) = blocksize;
        }
        globalOutput.update();
      }
      //auto* out = &(globalOutput[0][0]);
      auto* in = &(localInput[0][0]);

      auto tmp = std::vector<double>();
      if (rank==0)
        tmp.resize(globalSize * blocksize);

      auto cts = all_counts;
      for(auto& c : cts)
        c*=blocksize;

      auto disps = all_displs;
      for(auto& d : disps)
        d*=blocksize;

      MPI_Gatherv(
          in, localInput.size()*blocksize, MPI_DOUBLE,
          tmp.data(), cts.data(), disps.data(), MPI_DOUBLE, 0, comm_
          );

      // now we have all the values in tmp. Distribute them back into globalOutput:
      if (rank==0) {
        for(std::size_t i = 0; i < tmp.size(); i++) {
          auto block = i / blocksize;
          if(all_owned[block]) {
            auto local = i%blocksize;
            globalOutput[all_globalDof[block]][local]=tmp[i];
          }
        }
      }

    }

    /** Scatters a global vector on rank 0 back to the processes */
    template<typename V> // TODO: V is always DynamicBlockVector
    void scatterVector(const V& global, V& local) {
      auto blocksize = local[0].size();
      // copy global vector to all processes and let them figure out which entries to take
      int n_global = global.size();
      MPI_Bcast(&n_global, 1, MPI_INT, 0, comm_);

      V tmp(n_global);
      for(std::size_t i = 0; i < tmp.size(); i++) {
        tmp.blockRows(i)=blocksize;
      }
      tmp.update();

      if(rank==0) {
        tmp=global;
      }

      void* buffer = &(tmp[0][0]);

      // send data to all processes
      MPI_Bcast(buffer, n_global*blocksize, MPI_DOUBLE,
          0, comm_);

      auto dofs = std::vector<int>(local_count);
      MPI_Scatterv(all_globalDof.data(), all_counts.data(), all_displs.data(), MPI_INT,
          dofs.data(), local_count, MPI_INT, 0, comm_);

      for(std::size_t i = 0; i < local_count; ++i) {
        const auto idx = dofs[i];
        local[i] = tmp[idx];
      }
    }

    /** Collects the local matrix patches from all processes and creates
     * a global matrix index set on process 0 (aka "root").
     */
    template<typename Matrix>
    void restrictMatrix(const Matrix& localMatrix, Matrix& matrix) {
      MatrixIndexSet idxSet;
      if (rank==0) {
        idxSet.resize(ownedSize_, ownedSize_);
      }

      // step 0a: count non zeros per row
      std::vector<int> local_nnzPerRow(localMatrix.N());
      for(std::size_t i = 0; i < local_nnzPerRow.size(); i++) {
        local_nnzPerRow[i] = localMatrix.getrowsize(i);
      }

      // step 0b: collect on root
      std::vector<int> all_nnzByRow;
      {
        if (rank == 0)
          all_nnzByRow.resize(globalSize);
        MPI_Gatherv(local_nnzPerRow.data(), local_nnzPerRow.size(), MPI_INT,
                    all_nnzByRow.data(), all_counts.data(), all_displs.data(), MPI_INT,
                    0, comm_);
      }

      // step 1: collect column indices and send them to root
      std::vector<int> all_colIndices;
      if(rank==0) {
        auto size = std::accumulate(all_nnzByRow.begin(), all_nnzByRow.end(), 0);
        all_colIndices.resize(size);
      }

      int local_size = std::accumulate(local_nnzPerRow.begin(), local_nnzPerRow.end(), 0);
      std::vector<int> local_colIndices(local_size);

      {
        std::size_t j =0;
        for(std::size_t i = 0; i < localMatrix.N(); i++) {
          for(auto it = localMatrix[i].begin(); it != localMatrix[i].end(); ++it)
            local_colIndices[j++] = it.index();
        }
      }

      // sum the number of column indices on all processes
      int n_proc=0;
      MPI_Comm_size(comm_, &n_proc);
      if(rank==0) {
        int n_allColIndices = std::accumulate(all_nnzByRow.begin(), all_nnzByRow.end(), 0);
        all_colIndices.resize(n_allColIndices);
      }


      {
        // gather how many nnz we have in total per rank
        std::vector<int> all_nnzByRank;
        std::vector<int> all_nnzByRank_displ;

        if (rank == 0) {
          all_nnzByRank.resize(n_proc);
          for (std::size_t i = 0; i < n_proc; ++i)
            all_nnzByRank[i] = std::accumulate(
                std::next(all_nnzByRow.begin(), all_displs[i]),
                std::next(all_nnzByRow.begin(), all_displs[i+1]),
                0
                );
          all_nnzByRank_displ.resize(n_proc);
          ParMG::Impl::exclusive_scan(
              all_nnzByRank.begin(),
              all_nnzByRank.end(),
              all_nnzByRank_displ.begin(),
              0
              );
        }

        // gather the column indices
        MPI_Gatherv(local_colIndices.data(), local_colIndices.size(), MPI_INT,
            all_colIndices.data(), all_nnzByRank.data(), all_nnzByRank_displ.data(), MPI_INT,
            0, comm_);
      }

      // step 2: put together the index set
      if(rank==0) {

        int idx = 0; // indexing var for the all_colIndices vector
        assert(all_counts.size()==n_proc);
        for(std::size_t p = 0; p < n_proc; p++) {
          for(std::size_t i = 0; i < all_counts[p]; i++) { // matrix row
            auto all_i = all_displs[p]+i;
            for(std::size_t j = 0; j < all_nnzByRow.at(all_i); j++) { // matrix col
              if (all_owned[all_i]) { // only take data from the owners
                idxSet.add(all_globalDof[all_i], all_globalDof.at(all_displs[p]+all_colIndices[idx]));
              }
              idx++;
            }
          }
        }
        assert(idx == all_colIndices.size()); // check if we reached the end of the indices
      }

      int blocksize = localMatrix[0][0].N(); // if this block does not exist, we're in trouble anyway

      // setup matrix:
      if(rank==0)
      {
        idxSet.exportIdx(matrix);
        matrix.finishIdx();
        for(std::size_t i = 0; i < matrix.N(); i++) {
          matrix.blockRows(i)=blocksize;
        }
        matrix.setSquare();
        matrix.update();
      }

      int blocks = 0;
      for(std::size_t i = 0; i < localMatrix.N(); i++) {
        blocks+=localMatrix.getrowsize(i);
      }

      auto data_size = blocks*blocksize*blocksize;
      assert(data_size == localMatrix.dataSize());

      // step 0b: gather local matrix sizes to root
      std::vector<int> all_dataSizes(n_proc);
      std::vector<int> all_dataSizes_displs(n_proc);

      MPI_Gather(
          &data_size, 1, MPI_INT,
          all_dataSizes.data(), 1, MPI_INT,
          0, comm_);
      ParMG::Impl::exclusive_scan(all_dataSizes.begin(), all_dataSizes.end(), all_dataSizes_displs.begin(), 0);


#if 0
      // we (again) need the nnz per row
      std::vector<int> local_nnzPerRow(localMatrix.N());
      for(std::size_t i = 0; i < local_nnzPerRow.size(); i++) {
        local_nnzPerRow[i] = localMatrix.getrowsize(i);
      }
      std::vector<int> all_nnzByRow;
      {
        if (rank == 0)
          all_nnzByRow.resize(globalSize);
        MPI_Gatherv(local_nnzPerRow.data(), local_nnzPerRow.size(), MPI_INT,
                    all_nnzByRow.data(), all_counts.data(), all_displs.data(), MPI_INT,
                    0, comm_);
      }
#endif

      // step 1: gather data
      std::vector<double> all_data{};
      if (rank==0) {
        auto full_size = std::accumulate(all_dataSizes.begin(), all_dataSizes.end(), 0);
        all_data.resize(full_size);
      }

      {
        //auto* local_data = &(localMatrix[0][0][0][0]); // TODO can one actually assume this?
        //auto* local_data = localMatrix.data();

        // just extracting the local data does not seem to work reliably. Therefore, we copy the data into
        // a buffer block by block...
        // // TODO Check if this is still true after the bigger bug is fixed.
        auto tmp = std::vector<double>(data_size);
        std::size_t local_idx=0;
        for(const auto& row: localMatrix) {
          for(const auto& block: row) {
            for(const auto& block_row : block) {
              for(const auto& entry : block_row) {
                tmp[local_idx++] = entry;
              }
            }
          }
        }

        MPI_Gatherv(tmp.data(), data_size, MPI_DOUBLE,
            all_data.data(), all_dataSizes.data(), all_dataSizes_displs.data(), MPI_DOUBLE,
            0, comm_);

      }

      // step 2: pack data into matrix
      if(rank==0) {
        std::size_t idx = 0;
        for (int p = 0; p < n_proc; ++p) {
          for(std::size_t i = 0; i < all_counts[p]; i++) { // matrix row
            auto all_i = all_displs[p]+i;

            if (not all_owned[all_i]) {
              idx+=all_nnzByRow[all_i]*blocksize*blocksize;
              continue;
            }

            // now, we're at a owned matrix row. Let us fill it!
            // TODO: We're assuming here that the local_i < local_j implies global(i)<global(j)
            // which might not be true
            auto& ai = matrix[all_globalDof[all_i]];
            for(std::size_t local_j = 0; local_j < all_nnzByRow[all_i]; ++local_j) {
              auto global_j = all_globalDof[all_displs[p]+all_colIndices[idx/(blocksize*blocksize)]];
              auto& aij = ai[global_j];

              // fill block
              for (auto& bk: aij) {
                for (auto& bkl : bk) {
                  bkl=all_data[idx++];
                }
              }

            }
          }
        }
        assert(idx == all_data.size());
      }
    }
    private:
      MPI_Comm comm_;
      int rank;
      int ownedSize_=0;

      int local_count;

      std::vector<int> all_counts;
      std::vector<int> all_displs;
      std::vector<int> all_globalDof;
      std::vector<char> all_owned;
      std::size_t globalSize;
  };

}
}
}
#endif
