#ifndef DUNE_HPDG_NORMS_ENERGY_NORM_HH
#define DUNE_HPDG_NORMS_ENERGY_NORM_HH

#include <cmath>
#include <dune/common/exceptions.hh>
#include <dune/solvers/norms/norm.hh>

namespace Dune {
namespace HPDG {

    /** \brief Vector norm induced by linear operator
     *
     *  \f$\Vert u \Vert_A = (u, Au)^{1/2}\f$
     *
     * As alternative to providing the EnergyNorm directly with a matrix
     * one can also provide it with a LinearIterationStep. In this case
     * the matrix for the linear problem associated with the LinearIterationStep
     * is used. This is necessary because you sometimes do not
     * know the matrix in advance. This is, for example, the case
     * for the coarse level matrices constructed by a multilevel solver.
     *
     * Be careful: This matrix is not the one representing the preconditioner
     * induced by the LinearIterationStep.
     *
     *  \todo Elaborate documentation.
     */
    template<class MatrixType, class V>
    class EnergyNorm : public Norm<V>
    {
    public:
        typedef V VectorType;

        /** \brief The type used for the result */
        typedef typename VectorType::field_type field_type;

        EnergyNorm(const double tol=1e-10 ) : matrix_(NULL), tol_(tol) {}

        EnergyNorm(const MatrixType& matrix, const double tol=1e-10)
            : matrix_(&matrix), tol_(tol)
        {}

        void setMatrix(const MatrixType* matrix) {
            matrix_ = matrix;
        }

        //! Compute the norm of the difference of two vectors
        field_type diff(const VectorType& f1, const VectorType& f2) const {
            if (matrix_ == nullptr)
                DUNE_THROW(Dune::Exception, "You have not supplied a matrix to the EnergyNorm");

            VectorType tmp_f = f1;
            tmp_f -= f2;
            return (*this)(tmp_f);
        }

        //! Compute the norm of the given vector
        field_type operator()(const VectorType& f) const
        {
            return std::sqrt(normSquared(f));
        }

        // \brief Compute the square of the norm of the given vector
        virtual field_type normSquared(const VectorType& f) const
        {
            //const field_type ret = Dune::MatrixVector::Axy(*matrix_, f, f);
            auto dummy = f;
            matrix_->mv(f, dummy);
            auto ret = dummy*f;

            if (ret < 0)
            {
                if (ret < -tol_)
                    DUNE_THROW(Dune::RangeError, "Supplied linear operator is not positive (semi-)definite: (u,Au) = " << ret);
                return 0.0;
            }

            return ret;
        }

    protected:

        const MatrixType* matrix_;

        const double tol_;

    };

} /* namespace HPDG */
} /* namespace Dune */
#endif
