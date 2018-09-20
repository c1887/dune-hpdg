// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/localfunctions/lagrange/qk/qklocalcoefficients.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
#include <dune/hpdg/common/mmmatrix.hh>
#include "localoperator.hh"

namespace Dune {
namespace Fufem {
namespace MatrixFree {

  /** computes (∇φ_j, ∇u) for all j on a given element.
   * This works via sumfactorization and is only valid if
   * Basis is a Dune::HPDG based Gauss-Lobatto DG-Basis!
   * In particular, it relies on several implementation details
   *
   * TODO:
   *
   * - Generalize to dimensons != 2
   *
   * - Right now, this is in an extremely messy state
   */
  template<class V, class GV, class Basis>
  class SumFactIPDGOperator : public LocalOperator<V, GV> {
    using Base = LocalOperator<V, GV>;
    using LV = typename Basis::LocalView;
    using Field = typename V::field_type;
    using IndexPair = std::array<int, 2>; // The first index describes the degree of the basis function and the second one describes _the order_ of the quadrature rule!
    using LocalMatrix = Dune::Matrix<Dune::FieldMatrix<double, 1, 1>>;
    using BV = Dune::BlockVector<Dune::FieldVector<double,1>>;

    static constexpr int dim = GV::dimension;
    static_assert(dim==2, "Sumfactorized IPDG only for dim=2 currently");
    using FV = Dune::FieldVector<double, dim>;

    public:

      SumFactIPDGOperator(const Basis& b, double penalty, bool dirichlet=true) :
        basis_(b),
        penalty_(penalty),
        dirichlet_(dirichlet),
        localView_(basis_.localView()),
        outerView_(basis_.localView()),
        mapper_(basis_.gridView(), mcmgElementLayout()) {}

      void bind(const typename Base::Entity& e)
      {
        localView_.bind(e);

        localVector_.resize(localView_.size());
        for (auto& entry: localVector_)
          entry=0;

        // get order:
        localDegree_ = basis_.preBasis().degree(e);

        int order = 2*localDegree_ - 1;
        auto idxPair = IndexPair{{localDegree_, order}}; // first basis degree, 2nd order of evaluated quadrature

        // check if order is already in cache:
        {
          auto it = cache_.find(idxPair);
          if (it != cache_.end()) {
            matrixPair_ = &(it->second);
            rule_ = &(rules_[order]);
            return;
          }
        }

        // compute the Lagrange polynomials and derivatives at the quad points
        setupMatrixPair(idxPair);

        matrixPair_ = &(cache_[idxPair]);

        rule_ = &(rules_[order]);

      }

      void compute() {
        computeBulk();
        computeFace();
      }

      void write(double factor) {
        // if the DG hack was used in compute, return here:
        // return;

        if (factor!=1.0)
          for (auto& entry: localVector_)
            entry*=factor;
        if (factor == 0.0)
          return;

        auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
        for (size_t localRow=0; localRow<localView_.size(); ++localRow)
        {
          auto& rowEntry = outputBackend(localView_.index(localRow));
          rowEntry += localVector_[localRow];
        }

      }

    private:

      void computeBulk() {
        // compute needed order:
        //auto degree = localDegree_ +1; // if order p, we want p+1 nodes


        const auto& geometry = localView_.element().geometry();

        // TODO: If grid is very simple, one can compute jac and gamma once here.

        // find the first entry in input vector. Because we know DG indices are contiguous and
        // the local 0 idx is also the lowest global index, we can perform this little
        // hack to circumvent using a buffer and several calls do localView.index(foo).
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        const auto* coeffs = &(inputBackend(localView_.index(0))); // using DG structure here


        // precompute the derivatives of the local function at each (tensor-product) quad point
        auto dx_u = Dune::HPDG::BtUL((*matrixPair_)[0], coeffs, (*matrixPair_)[1]); // matrix contains \partial_x u(xi_{i0, i1})
        auto dy_u = Dune::HPDG::BtUL((*matrixPair_)[1], coeffs, (*matrixPair_)[0]); // matrix contains \partial_y u(xi_{i0, i1})

        // i0 and i1 are indices wrt to the RULE, not the basis
        auto computeX= [&](auto i0, auto i1, const auto& jacT) {
          FV gradU {dx_u[i0][i1], dy_u[i0][i1]};
          FV dummy;
          jacT.mv(gradU, dummy);
          jacT.mtv(dummy, gradU);
          return gradU;
        };

        auto i_length = rule_->size(); // number of quadrature points

        std::array<Dune::Matrix<Dune::FieldMatrix<double,1,1>>, dim> innerValues; // TODO: document exactly what's in here
        for (auto& iv: innerValues)
          iv.setSize(i_length, i_length);

        // pre-compute inner values
        for(std::size_t i1 = 0; i1 < i_length; i1++) {
          FV pos{0, (*rule_).at(i1).position()};
          for(std::size_t i0 = 0; i0 < i_length; i0++) {
            pos[0]=(*rule_).at(i0).position();
            // compute jacobian
            const auto& jac = geometry.jacobianInverseTransposed(pos);
            auto gamma = geometry.integrationElement(pos);

            auto x_values = computeX(i0,i1,jac);
            x_values *= gamma*(*rule_)[i0].weight()*(*rule_)[i1].weight();
            for (size_t r=0; r<dim; r++) {
              innerValues[r][i0][i1]=x_values[r];
            }
          }
        }

        // compute all the integrals for the given dimension at once
        for(size_t r=0; r <dim; r++) {
          const auto& matrix0 = (*matrixPair_)[r==0 ? 1 : 0];
          const auto& matrix1 = (*matrixPair_)[r==1 ? 1 : 0];

          Dune::HPDG::CplusAXtBt(matrix1, innerValues[r], matrix0, localVector_.data());

          // This here is a nice use of the common DG hack. However, be aware that this would
          // probably not be thread-safe!
          //auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
          //auto* out = &(outputBackend(localView_.index(0)));
          //Dune::HPDG::CplusAXtBt(matrix1, innerValues[r], matrix0, out);
        }
      }

      void computeFace() {

        const auto& gv = basis_.gridView();

        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        auto* coeffs = &(inputBackend(localView_.index(0)));

        for (const auto& is: intersections(gv, localView_.element())) {
          if (!is.neighbor()) // this is also for processor boundaries valid
          {
            if(dirichlet_ && is.boundary()) {
              // TODO Do something for Dirichlet data
              computeDirichletBoundaryEdge(is);
            }
            else
              continue;
          }
          else {
            /*
             * Selbst wenn das hier so funktioniert, wird das nicht für nicht-konforme Gitter
             * laufen. Grund ist, dass bisher immer davon ausgegangen wird, dass die Intersection dem ganzen
             * Face des Elements entspricht. Wenn das nicht der Fall ist, geht sicher was schief.
             *
             * Man muss also leider l(innerGeo.global(xi)) und so weiter neu ausrechnen :(.
             * Aber auch nur, wenn is.conforming() == false.
             */

            // Skip handled edges (we visit each intersection twice, but we want to do the work only once).
            if(mapper_.index(is.inside()) < mapper_.index(is.outside()))
              continue;

            auto innerIdx = is.indexInInside();
            auto outerIdx = is.indexInOutside();

            // step 0.: bind outer variables to outer element
            // This computes outer matrix pair and also readjust the inner matrix pair if needed.
            outerBind(is.outside());

            decltype(nonConformingMatrices(is)) nc_matrices;
            auto* inner_edgePair = matrixPair_;
            auto* outer_edgePair = outerMatrixPair_;

            if(not is.conforming())
            {
              // compute 1d values at nonconforming edge
              nc_matrices = nonConformingMatrices(is);

              // set the corresponding pointers
              inner_edgePair = &((*nc_matrices)[0]);
              outer_edgePair = &((*nc_matrices)[1]);
            }

            const auto* outsideCoeffs = &(inputBackend(outerView_.index(0)));

            // penalty= sigma p^2 / |e|
            auto penalty = penalty_*std::pow(std::max(localDegree_, outerLocalDegree_), 2)
              /is.geometry().volume();
            auto outerNormal = is.centerUnitOuterNormal();

            // prepare some values for inner and outer u
            BV u_diff(rule_->size());
            BV du_sum(rule_->size());

            auto inner_X = LocalMatrix(2, rule_->size());
            auto outer_X = LocalMatrix(2, rule_->size());

            auto inner_coeffs = coefficientsOnEdge(coeffs, localDegree_, innerIdx);
            (*inner_edgePair)[0].mtv(inner_coeffs, u_diff); // CONF

            auto outer_coeffs = coefficientsOnEdge(outsideCoeffs, outerLocalDegree_, outerIdx);
            (*outer_edgePair)[0].mmtv(outer_coeffs, u_diff); // CONF

            // prepare derivatives
            auto dx_i = computeDerivatives(coeffs, *matrixPair_, *inner_edgePair, innerIdx, 0); // CONF
            auto dy_i = computeDerivatives(coeffs, *matrixPair_, *inner_edgePair, innerIdx, 1); // CONF
            auto dx_o = computeDerivatives(outsideCoeffs, *outerMatrixPair_, *outer_edgePair, outerIdx, 0); // CONF
            auto dy_o = computeDerivatives(outsideCoeffs, *outerMatrixPair_, *outer_edgePair, outerIdx, 1); // CONF

            // multiply with integration weight and Jacobian determinant (TODO: Maybe do this later)
            for(std::size_t i = 0; i < rule_->size(); i++) {
              auto pos = (*rule_).at(i).position();

              const auto& jac_i = is.inside().geometry().jacobianInverseTransposed(is.geometryInInside().global(pos));
              const auto& jac_o = is.outside().geometry().jacobianInverseTransposed(is.geometryInOutside().global(pos));
              auto gamma = is.geometry().integrationElement(pos);

              auto factor =(*rule_)[i].weight()*gamma;
              u_diff[i]*=factor;

              auto gradU_i = FV{dx_i[i], dy_i[i]};
              auto gradU_o = FV{dx_o[i], dy_o[i]};
              auto dummy = gradU_i;
              jac_i.mv(gradU_i, dummy);
              jac_o.umv(gradU_o, dummy);
              du_sum[i] = (dummy*outerNormal)*factor*0.5;

              FV tmp_i;
              jac_i.mtv(outerNormal, tmp_i);
              FV tmp_o;
              jac_o.mtv(outerNormal, tmp_o);

              for (int r = 0; r < dim; r++) {
                inner_X[r][i]=-0.5*tmp_i[r];
                outer_X[r][i]=-0.5*tmp_o[r];
              }

            }
            computeDPhi(localVector_, *matrixPair_, *inner_edgePair, inner_X, u_diff, innerIdx);
            computeDPhi(outerLocalVector_, *outerMatrixPair_, *outer_edgePair, outer_X, u_diff, outerIdx);
            // inner
            //if(false)
            {
              BV tmp(localDegree_+1,0);
              // - {du/dn}[phi]
              (*inner_edgePair)[0].mmv(du_sum, tmp);
              //tmp*=-1;
              addOnEdge(localVector_, tmp, innerIdx); // phi lives on inner here
            }
            // outer
            //if(false)
            {
              BV tmp(outerLocalDegree_+1, 0);
              // - {du/dn}[phi]
              //
              // Since phi is negative for the outer side, the minus cancel out. TODO wirklich?
              (*outer_edgePair)[0].umv(du_sum, tmp);
              addOnEdge(outerLocalVector_, tmp, outerIdx);
            }


            // Penalty term sigma/|e| [u][phi_i]
            //if(false)
            {
              u_diff *= penalty;
              {
                BV tmp(localDegree_+1);
                (*inner_edgePair)[0].mv(u_diff, tmp);
                addOnEdge(localVector_, tmp, innerIdx);
              }

              BV tmp(outerLocalDegree_+1,0);
              (*outer_edgePair)[0].mmv(u_diff, tmp);
              addOnEdge(outerLocalVector_, tmp, outerIdx);
            }

            auto outputBackend = Fufem::istlVectorBackend(*(this->output_));
            for (size_t localRow=0; localRow<outerView_.size(); ++localRow)
            {
              auto& rowEntry = outputBackend(outerView_.index(localRow));
              rowEntry += outerLocalVector_[localRow];
            }
          }
        }
      }

      template<class IS>
      void computeDirichletBoundaryEdge(const IS& is) {
        auto inputBackend = Fufem::istlVectorBackend<const Field>(*(this->input_));
        auto* coeffs = &(inputBackend(localView_.index(0)));

        auto innerIdx = is.indexInInside();

        // penalty= sigma p^2 / |e|
        auto penalty = penalty_*std::pow(localDegree_, 2)
          /is.geometry().volume();
        auto outerNormal = is.centerUnitOuterNormal();

        // prepare some values for inner and outer u
        BV u_diff(rule_->size());
        BV du_sum(rule_->size());

        auto inner_X = LocalMatrix(2, rule_->size());

        auto inner_coeffs = coefficientsOnEdge(coeffs, localDegree_, innerIdx);
        (*matrixPair_)[0].mtv(inner_coeffs, u_diff);

        // prepare derivatives
        auto dx_i = computeDerivatives(coeffs, *matrixPair_, *matrixPair_, innerIdx, 0);
        auto dy_i = computeDerivatives(coeffs, *matrixPair_, *matrixPair_, innerIdx, 1);

        // multiply with integration weight and Jacobian determinant (TODO: Maybe do this later)
        for(std::size_t i = 0; i < rule_->size(); i++) {
          auto pos = (*rule_).at(i).position();

          const auto& jac_i = is.inside().geometry().jacobianInverseTransposed(is.geometryInInside().global(pos));
          auto gamma = is.geometry().integrationElement(pos);

          auto factor =(*rule_)[i].weight()*gamma;
          u_diff[i]*=factor;

          auto gradU_i = FV{dx_i[i], dy_i[i]};
          auto dummy = gradU_i;
          jac_i.mv(gradU_i, dummy);
          du_sum[i] = (dummy*outerNormal)*factor;

          FV tmp_i;
          jac_i.mtv(outerNormal, tmp_i);

          for (int r = 0; r < dim; r++) {
            inner_X[r][i]=-tmp_i[r];
          }

        }

        computeDPhi(localVector_, *matrixPair_, *matrixPair_, inner_X, u_diff, innerIdx);

        {
          BV tmp(localDegree_+1,0);
          // - {du/dn}[phi]
          (*matrixPair_)[0].mmv(du_sum, tmp);
          addOnEdge(localVector_, tmp, innerIdx); // phi lives on inner here
        }

        // Penalty term sigma/|e| [u][phi_i]
        {
          u_diff *= penalty;
          BV tmp(localDegree_+1);
          (*matrixPair_)[0].mv(u_diff, tmp);
          addOnEdge(localVector_, tmp, innerIdx);
        }
      }

      void setupMatrixPair(IndexPair idx) {

        const auto& basis_degree = idx[0];
        const auto& quad_order = idx[1];
        // get quadrature rule:
        int order = 2*basis_degree -1;
        auto gauss_lobatto = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, order, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        rules_[quad_order] = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, quad_order+1, Dune::QuadratureType::GaussLobatto); // TODO Welche Ordnung braucht man wirklich?
        auto& rule = rules_[quad_order];

        // sort node points (they're also ordered for the basis)
        std::sort(gauss_lobatto.begin(), gauss_lobatto.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        // sort quadrature points
        std::sort(rule.begin(), rule.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        assert(gauss_lobatto.size() == basis_degree+1);

        // save all derivatives of 1-d basis functions at the quad points, i.e.
        // matrixPair_ij = l_i' (xi_j)
        LocalMatrix lp(basis_degree+1, rules_[quad_order].size());
        // same for the function values
        LocalMatrix l(basis_degree+1, lp.M());
        for (std::size_t i = 0; i < lp.N(); i++) {
          for (std::size_t j = 0; j < lp.M(); j++) {
            lp[i][j]=lagrangePrime(rule[j].position(),i, gauss_lobatto);
            l[i][j]=lagrange(rule[j].position(),i, gauss_lobatto);
          }
        }
        cache_[idx]=std::array<LocalMatrix, 2>{{std::move(l), std::move(lp)}};
      }

      template<class X, class Q>
      inline double lagrangePrime(const X& x, size_t i, const Q& quad) const {
        double result = 0.;

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            double prod= 1.0/(quad[i].position()-quad[j].position());
            for (size_t l=0; l<quad.size(); l++)
              if (l!=i && l!=j)
                prod *= (x-quad[l].position())/(quad[i].position()-quad[l].position());
            result += prod;
          }
        return result;
      }

      template<class X, class Q>
      inline double lagrange(const X& x, size_t i, const Q& quad) const {
        double result = 1.;

        auto xi= quad[i].position();

        for (size_t j=0; j<quad.size(); j++)
          if (j!=i)
          {
            result*= (x-quad[j].position())/(xi-quad[j].position());
          }
        return result;
      }

      void outerBind(const typename Base::Entity& e)
      {
        outerView_.bind(e);

        outerLocalVector_.resize(outerView_.size());
        for (auto& entry: outerLocalVector_)
          entry=0;

        // get order:
        outerLocalDegree_ = basis_.preBasis().degree(e);

        // set up correct idx pair:
        // 0. Fall:
        // outerDeg = localDeg, alles schick, kein Problem:
        if (outerLocalDegree_ == localDegree_) { // everything's uniform, just set the pointers.
          outerMatrixPair_ = matrixPair_;
          return ;
        }
        int outerOrder = 2*outerLocalDegree_ - 1;
        // 1. Fall:
        // outerDeg > local(inner) degree,
        // dann brauchen wir (outer, outer) für außen
        // und (inner, outer) für innen
        IndexPair outside;
        IndexPair inner;
        if (outerLocalDegree_ > localDegree_) {
          outside={outerLocalDegree_, outerOrder};
          inner= {localDegree_, outerOrder};

          // check if inner is already in cache, compute if necessary
          {
            auto it = cache_.find(inner);
            if (it != cache_.end()) {
              matrixPair_ = &(it->second);
            }
            else {
              setupMatrixPair(inner);
              matrixPair_ = &(cache_[inner]);
            }
            rule_ = &(rules_[outerOrder]);
          }
        }

        // 2. Fall,
        // inner degree > outer degree
        // (outer, inner_order) für außen
        else  {
          inner = {localDegree_, 2*localDegree_ -1}; // This one should be available yet, as it was used for the bulk terms already.
          {
            auto it = cache_.find(inner);
            if (it != cache_.end()) {
              matrixPair_ = &(it->second);
            }
            else {
              setupMatrixPair(inner);
              matrixPair_ = &(cache_[inner]);
            }
            rule_ = &(rules_[2*localDegree_-1]);
          }

          // more interestingly, set the outer quad order to the one of the inner:
          outside = {outerLocalDegree_, 2*localDegree_-1};
        }

        // set (and if needed, calculate) outer matrix pair and rule
        {
          auto it = cache_.find(outside);
          if (it != cache_.end()) {
            outerMatrixPair_= &(it->second);
          }
          else {
            setupMatrixPair(outside);
            outerMatrixPair_ = &(cache_[outside]);
          }
          //outerRule_ = &(rules_[outside]);
        }

      }
      template<class IS>
      auto nonConformingMatrices(const IS& is)
      {
        // basis nodes:
        auto gl_i = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, 2*localDegree_-1, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        auto gl_o = Dune::QuadratureRules<typename GV::Grid::ctype,1>::rule(Dune::GeometryType::cube, 2*outerLocalDegree_-1, Dune::QuadratureType::GaussLobatto); // these are the GL lagrange nodes
        assert(gl_i.size() == localDegree_+1);
        assert(gl_o.size() == outerLocalDegree_+1);

        const auto& rule = *rule_;

        // sort node points (they're also ordered for the basis)
        std::sort(gl_i.begin(), gl_i.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });
        std::sort(gl_o.begin(), gl_o.end(), [](auto&& a, auto&& b) {
          return a.position() < b.position(); });

        // compute inner and outer matrix pair for the noncorming edge.
        // step 0.: compute new quad notes:
        auto inner_quad = std::vector<typename GV::Grid::ctype>(rule.size());
        auto outer_quad = std::vector<typename GV::Grid::ctype>(rule.size());
        for (size_t i = 0; i < rule.size(); i++) {
          inner_quad[i] = is.geometryInInside().global(rule[i].position())[is.indexInInside() < 2]; // we only take the coordinate we need, which will be the first for edge number 2 and 3, and the second for 0 and 1
          outer_quad[i] = is.geometryInOutside().global(rule[i].position())[is.indexInOutside() < 2];
        }
        auto ret = std::make_unique<std::array<std::array<LocalMatrix, 2>,2>>();

        auto& innerPair = (*ret)[0];
        auto& outerPair = (*ret)[1];

        // resize:
        innerPair[0].setSize(gl_i.size(), inner_quad.size());
        innerPair[1].setSize(gl_i.size(), inner_quad.size());

        outerPair[0].setSize(gl_o.size(), outer_quad.size());
        outerPair[1].setSize(gl_o.size(), outer_quad.size());


        // fill cache
        for (std::size_t j = 0; j < rule.size(); j++) {
          for (std::size_t i = 0; i < gl_i.size(); i++) {
            innerPair[1][i][j]=lagrangePrime(inner_quad[j],i, gl_i);
            innerPair[0][i][j]=lagrange(inner_quad[j],i, gl_i);
          }
          for (std::size_t i = 0; i < gl_o.size(); i++) {
            outerPair[1][i][j]=lagrangePrime(outer_quad[j],i, gl_o);
            outerPair[0][i][j]=lagrange(outer_quad[j],i, gl_o);
          }
        }

        return ret;
      }

      // TODO: Hier rauszukopieren ist auch dämlich... Da kann man das RowOrColumnWindow nutzen.
      // Nur dass ein nackter double coefficients (statt FV) sich nicht gut mit der ISTL Arithmetik verträgt.
      template<class U>
      auto coefficientsOnEdge(const U* coefficients, int degree, int edgeNumber) const {
        auto vector = BV(degree+1);
        switch(edgeNumber) {
          case 2: // we're on the left edge. The x-coordinate is \delta_0,i ane hence we look at the first row of localVector
            for (size_t k = 0; k < vector.size(); k++) {
              vector[k] = coefficients[k];
            }
            break;
          case 3: // right edge, coordinate is \delta_{p, i}, hence last row
            for (size_t k = 0; k < vector.size(); k++) {
              vector[k]=coefficients[(degree+1)*degree + k];
            }
            break;
          case 0: // lower edge, we look at the first col of localVector
            for (size_t k = 0; k < vector.size(); k++)
              vector[k]=coefficients[k*(degree+1)];
            break;
          case 1: // upper edge, last col
            for (size_t k = 0; k < vector.size(); k++)
              vector[k]=coefficients[k*(degree+1)+degree];
            break;
          default:
            DUNE_THROW(Dune::Exception, "Something went wrong here");
        }
        return vector;
      }

      // TODO document the logic here
      void addOnEdge(std::vector<typename V::field_type>& buffer, const BV& values, int edgeNumber) {
        auto degree = values.size()-1;

        // TODO: RoC
        switch(edgeNumber) {
          case 2:
            for (size_t k = 0; k < degree+1; k++) {
              buffer[k] += values[k];
            }
            break;
          case 3:
            for (size_t k = 0; k < degree+1; k++) {
              buffer.at((degree+1)*degree + k) += values[k]; // TODO. replace .at() by []
            }
            break;
          case 0:
            for (size_t k = 0; k < degree+1; k++)
              buffer.at(k*(degree+1)) +=values[k];
            break;
          case 1:
            for (size_t k = 0; k < degree+1; k++)
              buffer.at(k*(degree+1)+degree) +=values[k];
            break;
          default:
            DUNE_THROW(Dune::Exception, "Something went wrong here");
        }
      }

      template<class MP, class MP2, class M, class VV>
      void computeDPhi(std::vector<typename V::field_type>& buffer, const MP& matrixPair, const MP2& edgeMatrixPair, const M& STn, const VV& u, int edgeNumber) {
        auto degree = matrixPair[0].N()-1;
        auto quad_size = matrixPair[0].M();
        switch(edgeNumber) {
          case 0:// (0,q)
            for (size_t i = 0; i < degree +1; i++) {
              auto ith_col = HPDG::RowOrColumnWindow<typename V::field_type>(buffer.data(), degree+1, degree+1, i, HPDG::RowOrColumnWindow<typename V::field_type>::RowOrColumn::COL);
              for (size_t j = 0; j < degree +1; j++) {
                double front = 0;
                double back = 0;
                const auto& m0j = edgeMatrixPair[0][j];
                const auto& m1j = edgeMatrixPair[1][j];
                for (size_t q =0; q< u.size(); q++) {
                  front+= m0j[q]*STn[0][q]*u[q];
                  back+= m1j[q]*STn[1][q]*u[q];
                }
                ith_col[j] += matrixPair[1][i][0] * front + matrixPair[0][i][0] * back;
              }
            }
            return;
          case 1:
            for (size_t i = 0; i < degree +1; i++) {
              auto ith_col = HPDG::RowOrColumnWindow<typename V::field_type>(buffer.data(), degree+1, degree+1, i, HPDG::RowOrColumnWindow<typename V::field_type>::RowOrColumn::COL);
              for (size_t j = 0; j < degree +1; j++) {
                double front = 0;
                double back = 0;
                const auto& m0j = edgeMatrixPair[0][j];
                const auto& m1j = edgeMatrixPair[1][j];
                for (size_t q =0; q< u.size(); q++) {
                  front+= m0j[q]*STn[0][q]*u[q];
                  back+= m1j[q]*STn[1][q]*u[q];
                }
                ith_col[j] += matrixPair[1][i][quad_size-1] * front + matrixPair[0][i][quad_size -1] * back;
              }
            }
            return;
          case 2:
            for (size_t j = 0; j < degree +1; j++) {
              auto jth_row = HPDG::RowOrColumnWindow<typename V::field_type>(buffer.data(), degree+1, degree+1, j, HPDG::RowOrColumnWindow<typename V::field_type>::RowOrColumn::ROW);
              for (size_t i = 0; i < degree +1; i++) {
                double front = 0;
                double back = 0;
                const auto& m0i = edgeMatrixPair[0][i];
                const auto& m1i = edgeMatrixPair[1][i];
                for (size_t q =0; q< u.size(); q++) {
                  front+= m0i[q]*STn[1][q]*u[q];
                  back+= m1i[q]*STn[0][q]*u[q];
                }
                jth_row[i] += matrixPair[0][j][0] * back + matrixPair[1][j][0] * front;
              }
            }
            return;
          case 3:
            for (size_t j = 0; j < degree +1; j++) {
              auto jth_row = HPDG::RowOrColumnWindow<typename V::field_type>(buffer.data(), degree+1, degree+1, j, HPDG::RowOrColumnWindow<typename V::field_type>::RowOrColumn::ROW);
              for (size_t i = 0; i < degree +1; i++) {
                double front = 0;
                double back = 0;
                const auto& m0i = edgeMatrixPair[0][i];
                const auto& m1i = edgeMatrixPair[1][i];
                for (size_t q =0; q< u.size(); q++) {
                  front+= m0i[q]*STn[1][q]*u[q];
                  back+= m1i[q]*STn[0][q]*u[q];
                }
                jth_row[i] += matrixPair[0][j][quad_size-1] * back + matrixPair[1][j][quad_size-1] * front;
              }
            }
            return;
          default:
            DUNE_THROW(Dune::Exception, "Something went wrong here");
        }
      }

      template<class M>
      HPDG::RowOrColumnWindow<const typename M::block_type> coefficientWindow(const M& matrix, int edgeNumber) {
        using RoC = HPDG::RowOrColumnWindow<const typename M::block_type>;
        switch(edgeNumber) {
          case 0:
            return HPDG::rowWindow(matrix,0);
          case 1:
            return HPDG::rowWindow(matrix,matrix.N()-1);
          case 2:
            return HPDG::columnWindow(matrix, 0);
          case 3:
            return HPDG::columnWindow(matrix, matrix.M() -1);
          default:
            DUNE_THROW(Dune::Exception, "Something went wrong here");
        }
        // should never happen:
        assert(false);
        return RoC(nullptr, 0,0,0, RoC::RowOrColumn::ROW);
      }

      template<class U, class MP, class MP2>
      auto computeDerivatives(const U* coefficients, const MP& matrixPair, const MP2& edgeMatrixPair, const int edgeNumber, int direction) const {
        assert(direction == 0 or direction == 1);
        auto quad_size = matrixPair[0].M();
        BV dx(quad_size);
        switch(edgeNumber) {
          case 0:
            // (0, x)
            if (direction==0) {
              auto col0 = Dune::HPDG::columnWindow(matrixPair[1], 0);
              BV tmp(col0.size(),0); // we know that the coefficient "matrix" is quadratic
              Dune::HPDG::umv(coefficients, col0, tmp);
              edgeMatrixPair[0].mtv(tmp, dx);
            }
            else {
              auto coeffs = coefficientsOnEdge(coefficients, matrixPair[0].N()-1, edgeNumber);
              edgeMatrixPair[1].mtv(coeffs, dx);
            }
            return dx;
          case 1:
            // (1, x)
            if (direction==0) {
              auto col_last = Dune::HPDG::columnWindow(matrixPair[1], quad_size-1); // last column
              BV tmp(col_last.size(), 0);
              Dune::HPDG::umv(coefficients, col_last, tmp);
              edgeMatrixPair[0].mtv(tmp, dx);
            }
            else {
              auto coeffs = coefficientsOnEdge(coefficients, matrixPair[0].N()-1, edgeNumber);
              edgeMatrixPair[1].mtv(coeffs, dx);
            }
            return dx;
          case 2:
            // (x, 0)
            if (direction==0) {
              auto coeffs = coefficientsOnEdge(coefficients, matrixPair[0].N()-1, edgeNumber);
              edgeMatrixPair[1].mtv(coeffs, dx);
            }
            else {
              auto col0 = Dune::HPDG::columnWindow(matrixPair[1], 0); // first column
              BV tmp(col0.size(),0);
              Dune::HPDG::umtv(coefficients, col0, tmp);
              edgeMatrixPair[0].mtv(tmp, dx);
            }
            return dx;
          case 3:
            // (x, 1)
            if (direction==0) {
              auto coeffs = coefficientsOnEdge(coefficients, matrixPair[0].N()-1, edgeNumber);
              edgeMatrixPair[1].mtv(coeffs, dx);
            }
            else {
              auto col_last = Dune::HPDG::columnWindow(matrixPair[1], quad_size-1); // last column
              BV tmp(col_last.size(),0);
              Dune::HPDG::umtv(coefficients, col_last, tmp);
              edgeMatrixPair[0].mtv(tmp, dx);
            }
            return dx;
          default:
            DUNE_THROW(Dune::Exception, "Something went wrong here");
        }
      }

      // members:
      const Basis& basis_;
      double penalty_;
      bool dirichlet_;
      LV localView_;
      LV outerView_;
      std::map<IndexPair, std::array<LocalMatrix, 2>> cache_; // contains all lagrange Polynomials at all quadrature points and all derivatives of said polynomials at all quad points
      std::map<int, Dune::QuadratureRule<typename GV::Grid::ctype, 1>> rules_;
      std::vector<typename V::field_type> localVector_; // contiguous memory buffer
      std::vector<typename V::field_type> outerLocalVector_; // contiguous memory buffer
      int localDegree_;
      int outerLocalDegree_;
      const typename decltype(cache_)::mapped_type* matrixPair_; // Current matrix pair
      const typename decltype(cache_)::mapped_type* outerMatrixPair_; // Current matrix pair
      Dune::QuadratureRule<typename GV::Grid::ctype,1>* rule_;
      Dune::MultipleCodimMultipleGeomTypeMapper<typename Basis::GridView> mapper_;
  };
}
}
}
