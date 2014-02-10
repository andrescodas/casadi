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

#include "nlp_implicit_internal.hpp"

#include "symbolic/mx/mx_tools.hpp"
#include "symbolic/fx/mx_function.hpp"

using namespace std;
namespace CasADi {
  
  NLPImplicitInternal::NLPImplicitInternal(const FX& f, const FX& jac, const LinearSolver& linsol) : ImplicitFunctionInternal(f,jac,linsol) {
    addOption("nlp_solver",               OT_NLPSOLVER,  GenericType(), "The NLPSolver used to solve the implicit system.");
    addOption("nlp_solver_options",       OT_DICTIONARY, GenericType(), "Options to be passed to the NLPSolver");
  }

  NLPImplicitInternal::~NLPImplicitInternal(){ 
  }

  void NLPImplicitInternal::deepCopyMembers(std::map<SharedObjectNode*,SharedObject>& already_copied){
    ImplicitFunctionInternal::deepCopyMembers(already_copied);
    nlp_solver_ = deepcopy(nlp_solver_,already_copied);
  }

  void NLPImplicitInternal::solveNonLinear() {

    // Equality nonlinear constraints
    nlp_solver_.input(NLP_SOLVER_LBG).set(0.);
    nlp_solver_.input(NLP_SOLVER_UBG).set(0.);
    
    // Simple constraints
    vector<double>& lbx = nlp_solver_.input(NLP_SOLVER_LBX).data();
    vector<double>& ubx = nlp_solver_.input(NLP_SOLVER_UBX).data();
    for(int k=0; k<u_c_.size(); ++k){
      lbx[k] = u_c_[k] <= 0 ? -std::numeric_limits<double>::infinity() : 0;
      ubx[k] = u_c_[k] >= 0 ?  std::numeric_limits<double>::infinity() : 0;
    }
    
    // Pass initial guess
    nlp_solver_.input(NLP_SOLVER_X0).set(output(iout_));
  
    // Add auxiliary inputs
    vector<double>::iterator nlp_p = nlp_solver_.input(NLP_SOLVER_P).begin();
    for(int i=0, k=0; i<getNumInputs(); ++i){
      if(i!=iin_){
        std::copy(input(i).begin(),input(i).end(),nlp_p);
        nlp_p += input(i).size();
      }
    }
  
    // Solve the NLP
    nlp_solver_.evaluate();
    stats_["nlp_solver_stats"] = nlp_solver_.getStats();

    // Get the implicit variable
    output(iout_).set(nlp_solver_.output(NLP_SOLVER_X));

    // Evaluate auxilary outputs, if necessary
    if(getNumOutputs()>0){
      f_.setInput(output(iout_),iin_);
      for(int i=0; i<getNumInputs(); ++i)
        if(i!=iin_) f_.setInput(input(i),i);
      f_.evaluate();
      for(int i=0; i<getNumOutputs(); ++i){
        if(i!=iout_) f_.getOutput(output(i),i);
      }
    }
  }

  void NLPImplicitInternal::init(){

    // Call the base class initializer
    ImplicitFunctionInternal::init();

    // Free variable in the NLP
    MX u = msymQQQ("u",input(iin_).sparsity());
    
    // So that we can pass it on to createParent
    std::vector<CCSSparsity> sps;
    for(int i=0; i<getNumInputs(); ++i)
      if(i!=iin_) sps.push_back(input(i).sparsity());
    std::pair<MX,std::vector<MX> > mypair = createParent(sps);

    // u groups all parameters in an MX
    MX p = mypair.first;
    std::vector< MX > inputs(mypair.second);
    
    // Dummy NLP objective
    MX nlp_f = 0;

    // NLP constraints
    std::vector< MX > args_call(getNumInputs());
    args_call[iin_] = u;
    for(int i=0, i2=0; i<getNumInputs(); ++i)
      if(i!=iin_) args_call[i] = mypair.second[i2++];
    MX nlp_g = f_.call(args_call).at(iout_);

    // We're going to use two-argument objective and constraints to allow the use of parameters
    MXFunction nlp(nlpIn("x",u,"p",p),nlpOut("f",nlp_f,"g",nlp_g));
  
    // Create an nlpsolver instance
    NLPSolverCreator nlp_solvercreator = getOption("nlp_solver");
    nlp_solver_ = nlp_solvercreator(nlp);
    if(hasSetOption("nlp_solver_options")){
      nlp_solver_.setOption(getOption("nlp_solver_options"));
    }
  
    // Initialize the NLP solver
    nlp_solver_.init();
  }

} // namespace CasADi

