/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "qp_stabilizer_internal.hpp"

#include "symbolic/sx/sx_tools.hpp"
#include "symbolic/fx/sx_function.hpp"

using namespace std;
namespace CasADi {

  QPStabilizerInternal::QPStabilizerInternal(const std::vector<Sparsity> &st) : StabilizedQPSolverInternal(st) {
    addOption("qp_solver",         OT_QPSOLVER,   GenericType(), "The QP solver used to solve the stabilized QPs.");
    addOption("qp_solver_options", OT_DICTIONARY, GenericType(), "Options to be passed to the QP solver instance");
  }

  QPStabilizerInternal::~QPStabilizerInternal(){ 
  }

  void QPStabilizerInternal::deepCopyMembers(std::map<SharedObjectNode*,SharedObject>& already_copied){
    StabilizedQPSolverInternal::deepCopyMembers(already_copied);
    qp_solver_ = deepcopy(qp_solver_,already_copied);
  }

  void QPStabilizerInternal::init(){
    // Initialize the base classes
    StabilizedQPSolverInternal::init();

    // Form augmented QP
    Sparsity H_sparsity_qp = blkdiag(st_[QP_STRUCT_H],sp_diag(nc_));
    Sparsity A_sparsity_qp = horzcat(st_[QP_STRUCT_A],sp_diag(nc_));
    QPSolverCreator qp_solver_creator = getOption("qp_solver");
    qp_solver_ = qp_solver_creator(qpStruct("h",H_sparsity_qp,"a",A_sparsity_qp));

    // Pass options if provided
    if(hasSetOption("qp_solver_options")){
      Dictionary qp_solver_options = getOption("qp_solver_options");
      qp_solver_.setOption(qp_solver_options);
    } 
    
    // Initialize the QP solver
    qp_solver_.init();
  }

  void QPStabilizerInternal::evaluate() {
    double muR = input(STABILIZED_QP_SOLVER_MUR).at(0);
    std::vector<double>& muE = input(STABILIZED_QP_SOLVER_MUE).data();
    std::vector<double>& mu = input(STABILIZED_QP_SOLVER_MU).data();

    // Construct stabilized H
    DMatrix& H_qp = qp_solver_.input(QP_SOLVER_H);
    std::copy(input(STABILIZED_QP_SOLVER_H).begin(),input(STABILIZED_QP_SOLVER_H).end(),H_qp.begin());
    std::fill(H_qp.begin()+input(STABILIZED_QP_SOLVER_H).size(),H_qp.end(),muR);

    // Linear constraints
    if (nc_>0) {
      // Upper and lower bounds
      qp_solver_.setInput(input(STABILIZED_QP_SOLVER_LBA),QP_SOLVER_LBA);
      qp_solver_.setInput(input(STABILIZED_QP_SOLVER_UBA),QP_SOLVER_UBA);
      
      // Matrix term in the linear constraint
      DMatrix& A_qp = qp_solver_.input(QP_SOLVER_A);
      DMatrix& A = input(STABILIZED_QP_SOLVER_A);
      std::copy(A.begin(),A.end(),A_qp.begin());
      std::fill(A_qp.begin()+A.size(),A_qp.end(),-muR);
            
      // Add constant to linear inequality 
      for(int i=0; i<mu.size(); ++i) {
        double extra = muR*(mu[i]-muE[i]);
        qp_solver_.input(QP_SOLVER_LBA).at(i) += extra;
        qp_solver_.input(QP_SOLVER_UBA).at(i) += extra;
      }
    }

    // Bounds on x
    std::copy(input(STABILIZED_QP_SOLVER_LBX).begin(),input(STABILIZED_QP_SOLVER_LBX).end(),qp_solver_.input(QP_SOLVER_LBX).begin());
    std::copy(input(STABILIZED_QP_SOLVER_UBX).begin(),input(STABILIZED_QP_SOLVER_UBX).end(),qp_solver_.input(QP_SOLVER_UBX).begin());
    std::fill(qp_solver_.input(QP_SOLVER_LBX).begin()+n_,qp_solver_.input(QP_SOLVER_LBX).end(),-numeric_limits<double>::infinity());
    std::fill(qp_solver_.input(QP_SOLVER_UBX).begin()+n_,qp_solver_.input(QP_SOLVER_UBX).end(),numeric_limits<double>::infinity());
    
    // Gradient term in the objective
    DMatrix &g = input(STABILIZED_QP_SOLVER_G);
    std::copy(g.begin(),g.end(),qp_solver_.input(QP_SOLVER_G).begin());
    for (int i=0;i<nc_;++i) {
      qp_solver_.input(QP_SOLVER_G).at(g.size()+i) = muR * mu[i];
    }

    // Hot-starting if possible
    std::copy(input(STABILIZED_QP_SOLVER_X0).begin(),input(STABILIZED_QP_SOLVER_X0).end(),qp_solver_.input(QP_SOLVER_X0).begin());

    // Solve the QP
    qp_solver_.evaluate();

    // Pass the stats
    stats_["qp_solver_stats"] = qp_solver_.getStats();
  
    // Get the optimal solution
    std::copy(qp_solver_.output(QP_SOLVER_X).begin(),qp_solver_.output(QP_SOLVER_X).begin()+n_,output(QP_SOLVER_X).begin());
    std::copy(qp_solver_.output(QP_SOLVER_LAM_X).begin(),qp_solver_.output(QP_SOLVER_LAM_X).begin()+n_,output(QP_SOLVER_LAM_X).begin());
    std::copy(qp_solver_.output(QP_SOLVER_LAM_A).begin(),qp_solver_.output(QP_SOLVER_LAM_A).begin()+nc_,output(QP_SOLVER_LAM_A).begin());
  }

  void QPStabilizerInternal::generateNativeCode(std::ostream &file) const {
    qp_solver_.generateNativeCode(file);
  }

} // namespace CasADi

