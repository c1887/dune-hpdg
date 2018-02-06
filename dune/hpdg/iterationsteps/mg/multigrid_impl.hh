namespace Dune {
namespace HPDG {

namespace Impl {

template<typename Vector>
void resizeAndZero(Vector& x, const Vector& reference)
{
  x=reference;
  x = 0;
}

} /* namespace Impl */

template<typename Vector>
void Multigrid<Vector>::apply(Vector& x, Vector& b) const
{
  const unsigned levels = levelOperations_.size();
  const unsigned fine = levels - 1;

  State state;
  auto& tmp1 = state.tmp1;

  state.x.resize(levels);
  state.r.resize(levels);

  Impl::resizeAndZero(state.x[fine], x);

  // compute residual: r = b - Ax
  auto& r = state.r[fine];

  Impl::resizeAndZero(tmp1, x);
  levelOperations_[fine].apply(tmp1, x);
  // levelOperations_[fine].collect(tmp1);
  r = std::move(b);
  r -= tmp1;

  applyLevel(state, fine);

  //x = std::move(state.x[fine]);

#if 0
  levelOperations_[fine].restrictToMaster(r);
  serial([&](int rank, int) {
      std::cout << "I[" << rank << "]: residual (post, restricted) = " << r.two_norm() << "\n";
    });
#endif

  x += state.x[fine];
  b = std::move(r);
}

template<typename Vector>
void Multigrid<Vector>::levelOperations(const std::vector<LevelOperations>& levelOperations)
{
  levelOperations_ = levelOperations;
}

template<typename Vector>
void Multigrid<Vector>::coarseSolver(const CoarseSolver<Vector>& coarseSolver)
{
  coarseSolver_ = coarseSolver;
}

template<typename Vector>
void Multigrid<Vector>::applySmoother(State& state,
                                      const LevelOperations& op,
                                      const Smoother<Vector>& smoother,
                                      unsigned steps,
                                      Vector& x,
                                      Vector& r) const
{
  Impl::resizeAndZero(state.tmp1, x);
  Impl::resizeAndZero(state.tmp2, x); // tmp2=0 should not be required

  for (unsigned i = 0; i < steps; ++i) {
    smoother(state.tmp1, r);
  }
  x += state.tmp1;
  op.apply(state.tmp2, state.tmp1);
  r -= state.tmp2;
}

template<typename Vector>
void Multigrid<Vector>::applyLevel(State& state, unsigned level) const
{
  auto& x = state.x[level];
  auto& r = state.r[level];

  const auto& op = levelOperations_[level];

  //std::cout << "LEVEL = " << level << "\n";
  if (level == 0) {
    coarseSolver_(x, r);
    return;
  }

  // Pre-smoothing
  applySmoother(state, op, op.preSmoother, op.preSmootherSteps, x, r);

  // Coarse correction
  {
    op.restrict(state.r[level-1], r);
    Impl::resizeAndZero(state.x[level-1], state.r[level-1]);
    for (unsigned i = 0; i < mu_; ++i) {
      applyLevel(state, level-1);
    }
    op.prolong(state.tmp1, state.x[level-1]);
    x += state.tmp1;
    Impl::resizeAndZero(state.tmp2, r);
    op.apply(state.tmp2, state.tmp1);
    r -= state.tmp2;
  }

  // Post-smoothing
  applySmoother(state, op, op.postSmoother, op.postSmootherSteps, x, r);
}

} /* namespace HPDG */
} /* namespace Dune */
