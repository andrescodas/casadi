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

#include "nlp_solver_internal.hpp"
#include "mx_function.hpp"
#include "sx_function.hpp"
#include "../sx/sx_tools.hpp"
#include "../mx/mx_tools.hpp"
#include "../fx/fx_tools.hpp"

INPUTSCHEME(NLPSolverInput)
OUTPUTSCHEME(NLPSolverOutput)

using namespace std;
namespace CasADi{

  NLPSolverInternal::NLPSolverInternal(const FX& nlp) : nlp_(nlp){

    // Set default options
    setOption("name","unnamed NLP solver"); // name of the function

    // Options available in all NLP solvers
    addOption("expand",             OT_BOOLEAN,  false,          "Expand the NLP function in terms of scalar operations, i.e. MX->SX");
    addOption("hess_lag",            OT_FX,       GenericType(),  "Function for calculating the Hessian of the Lagrangian (autogenerated by default)"); 
    addOption("grad_lag",           OT_FX,       GenericType(),  "Function for calculating the gradient of the Lagrangian (autogenerated by default)"); 
    addOption("jac_g",              OT_FX,       GenericType(),  "Function for calculating the Jacobian of the constraints (autogenerated by default)"); 
    addOption("grad_f",             OT_FX,       GenericType(),  "Function for calculating the gradient of the objective (autogenerated by default)"); 
    addOption("iteration_callback", OT_CALLBACK, GenericType(),  "A function that will be called at each iteration with the solver as input. Check documentation of Callback.");
    addOption("iteration_callback_step", OT_INTEGER,         1,  "Only call the callback function every few iterations.");
    addOption("iteration_callback_ignore_errors", OT_BOOLEAN, false, "If set to true, errors thrown by iteration_callback will be ignored.");
    addOption("ignore_check_vec",   OT_BOOLEAN,  false,          "If set to true, the input shape of F will not be checked.");
    addOption("warn_initial_bounds",OT_BOOLEAN,  false,          "Warn if the initial guess does not satisfy LBX and UBX");

    // Legacy options, will go away. See #566.
    addOption("expand_f",           OT_BOOLEAN,  GenericType(),  "Expand the objective function in terms of scalar operations, i.e. MX->SX. Deprecated, use \"expand\" instead.");
    addOption("expand_g",           OT_BOOLEAN,  GenericType(),  "Expand the constraint function in terms of scalar operations, i.e. MX->SX. Deprecated, use \"expand\" instead.");
    addOption("generate_hessian",   OT_BOOLEAN,  GenericType(),  "Deprecated option. Generate an exact Hessian of the Lagrangian if not supplied.");
    addOption("generate_jacobian",  OT_BOOLEAN,  GenericType(),  "Deprecated option. Generate an exact Jacobian of the constraints if not supplied.");
    addOption("generate_gradient",  OT_BOOLEAN,  GenericType(),  "Deprecated option. Generate a function for calculating the gradient of the objective.");
    addOption("parametric",         OT_BOOLEAN,  GenericType(),  "Deprecated option. Expect F, G, H, J to have an additional input argument appended at the end, denoting fixed parameters.");
    addOption("gauss_newton",       OT_BOOLEAN,  GenericType(),  "Deprecated option. Use Gauss Newton Hessian approximation");      

    // Enable string notation for IO
    input_.scheme = SCHEME_NLPSolverInput;
    output_.scheme = SCHEME_NLPSolverOutput;
    
    // Make the ref object a non-refence counted pointer to this (as reference counting would prevent deletion of the object)
    ref_.assignNodeNoCount(this);
    
  }

  NLPSolverInternal::~NLPSolverInternal(){
    // Explicitly remove the pointer to this (as the counter would otherwise be decreased)
    ref_.assignNodeNoCount(0);
  }

  void NLPSolverInternal::init(){
    // Initialize the NLP
    nlp_.init(false);
    casadi_assert_message(nlp_.getNumInputs()==NL_NUM_IN, "The NLP function must have exactly two input");
    casadi_assert_message(nlp_.getNumOutputs()==NL_NUM_OUT, "The NLP function must have exactly two outputs");
    
    // Sparsity patterns
    const CCSSparsity& x_sparsity = nlp_.input(NL_X).sparsity();
    const CCSSparsity& p_sparsity = nlp_.input(NL_P).sparsity();
    const CCSSparsity& g_sparsity = nlp_.output(NL_G).sparsity();

    // Get dimensions
    nx_ = x_sparsity.size();
    np_ = p_sparsity.size();
    ng_ = g_sparsity.size();
    
    // Allocate space for inputs
    setNumInputs(NLP_SOLVER_NUM_IN);
    input(NLP_SOLVER_X0)       =  DMatrix::zeros(x_sparsity);
    input(NLP_SOLVER_LBX)      = -DMatrix::infQQQ(x_sparsity);
    input(NLP_SOLVER_UBX)      =  DMatrix::infQQQ(x_sparsity);
    input(NLP_SOLVER_LBG)      = -DMatrix::infQQQ(g_sparsity);
    input(NLP_SOLVER_UBG)      =  DMatrix::infQQQ(g_sparsity);
    input(NLP_SOLVER_LAM_X0)   =  DMatrix::zeros(x_sparsity);
    input(NLP_SOLVER_LAM_G0)   =  DMatrix::zeros(g_sparsity);
    input(NLP_SOLVER_P)        =  DMatrix::zeros(p_sparsity);
  
    // Allocate space for outputs
    setNumOutputs(NLP_SOLVER_NUM_OUT);
    output(NLP_SOLVER_X)       = DMatrix::zeros(x_sparsity);
    output(NLP_SOLVER_F)       = DMatrix::zeros(1);
    output(NLP_SOLVER_LAM_X)   = DMatrix::zeros(x_sparsity);
    output(NLP_SOLVER_LAM_G)   = DMatrix::zeros(g_sparsity);
    output(NLP_SOLVER_LAM_P)   = DMatrix::zeros(p_sparsity);
    output(NLP_SOLVER_G)       = DMatrix::zeros(g_sparsity);
  
    // Call the initialization method of the base class
    FXInternal::init();
    
    // Deprecation warnings
    casadi_assert_warning(!hasSetOption("expand_f"),"Option \"expand_f\" ignored (deprecated). Use \"expand\" instead.");
    casadi_assert_warning(!hasSetOption("expand_g"),"Option \"expand_g\" ignored (deprecated). Use \"expand\" instead.");
    casadi_assert_warning(!hasSetOption("generate_hessian"),"Option \"generate_hessian\" ignored (deprecated).");
    casadi_assert_warning(!hasSetOption("generate_jacobian"),"Option \"generate_jacobian\" ignored (deprecated).");
    casadi_assert_warning(!hasSetOption("generate_gradient"),"Option \"generate_gradient\" ignored (deprecated).");
    casadi_assert_warning(!hasSetOption("parametric"),"Option \"parametric\" ignored (deprecated).");
    casadi_assert_warning(!hasSetOption("gauss_newton"),"Option \"gauss_newton\" ignored (deprecated).");
    
    // Find out if we are to expand the NLP in terms of scalar operations
    bool expand = getOption("expand");
    if(expand){
      log("Expanding NLP in scalar operations");
      
      // Cast to MXFunction
      MXFunction nlp_mx = shared_cast<MXFunction>(nlp_);
      if(nlp_mx.isNull()){
	casadi_warning("Cannot expand NLP as it is not an MXFunction");
      } else {
        nlp_ = SXFunction(nlp_mx);
        nlp_.copyOptions(nlp_mx, true);
        nlp_.init();
      }
    }
  
    if (hasSetOption("iteration_callback")) {
      callback_ = getOption("iteration_callback");
    }
  
    callback_step_ = getOption("iteration_callback_step");
  }

  void NLPSolverInternal::checkInitialBounds() { 
    if(bool(getOption("warn_initial_bounds"))){
      bool violated = false;
      for (int k=0;k<input(NLP_SOLVER_X0).size();++k) {
        if (input(NLP_SOLVER_X0).at(k)>input(NLP_SOLVER_UBX).at(k)) {
          violated = true;
        }
        if (input(NLP_SOLVER_X0).at(k)<input(NLP_SOLVER_LBX).at(k)) {
          violated = true;
        }
      }
      if (violated) casadi_warning("NLPSolver: The initial guess does not satisfy LBX and UBX. Option 'warn_initial_bounds' controls this warning.");
    }
  }
   

  void NLPSolverInternal::reportConstraints(std::ostream &stream) { 
  
    stream << "Reporting NLP constraints" << endl;
    CasADi::reportConstraints(stream,output(NLP_SOLVER_X),input(NLP_SOLVER_LBX),input(NLP_SOLVER_UBX), "decision bounds");
    double tol = 1e-8;
    if (hasOption("constr_viol_tol")) tol = getOption("constr_viol_tol");
    CasADi::reportConstraints(stream,output(NLP_SOLVER_G),input(NLP_SOLVER_LBG),input(NLP_SOLVER_UBG), "constraints",getOption("constr_viol_tol"));
  }

  FX& NLPSolverInternal::gradF(){
    if(gradF_.isNull()){
      gradF_ = getGradF();
    }
    return gradF_;
  }

  FX NLPSolverInternal::getGradF(){
    FX gradF;
    if(hasSetOption("grad_f")){
      gradF = getOption("grad_f");
    } else {
      log("Generating objective gradient");
      gradF = nlp_.gradient(NL_X,NL_F);
      log("Gradient function generated");
    }
    gradF.setOption("name","grad_f");
    gradF.init(false);
    casadi_assert_message(gradF.getNumInputs()==GRADF_NUM_IN, "Wrong number of inputs to the gradient function. Note: The gradient signature was changed in #544");
    casadi_assert_message(gradF.getNumOutputs()==GRADF_NUM_OUT, "Wrong number of outputs to the gradient function. Note: The gradient signature was changed in #544");
    gradF.setInputScheme(SCHEME_GradFInput);
    gradF.setOutputScheme(SCHEME_GradFOutput);
    log("Objective gradient function initialized");
    return gradF;
  }

  FX& NLPSolverInternal::jacG(){
    if(jacG_.isNull()){
      jacG_ = getJacG();
    }
    return jacG_;
  }

  FX NLPSolverInternal::getJacG(){
    FX jacG;
    
    // Return null if no constraints
    if(ng_==0) return jacG;
    
    if(hasSetOption("jac_g")){
      jacG = getOption("jac_g");
    } else {
      log("Generating constraint Jacobian");
      jacG = nlp_.jacobian(NL_X,NL_G);
      log("Jacobian function generated");
    }
    jacG.setOption("name","jac_g");
    jacG.init(false);
    casadi_assert_message(jacG.getNumInputs()==JACG_NUM_IN, "Wrong number of inputs to the Jacobian function. Note: The Jacobian signature was changed in #544");
    casadi_assert_message(jacG.getNumOutputs()==JACG_NUM_OUT, "Wrong number of outputs to the Jacobian function. Note: The Jacobian signature was changed in #544");
    jacG.setInputScheme(SCHEME_JacGInput);
    jacG.setOutputScheme(SCHEME_JacGOutput);
    log("Jacobian function initialized");
    return jacG;
  }

  FX& NLPSolverInternal::gradLag(){
    if(gradLag_.isNull()){
      gradLag_ = getGradLag();
    }
    return gradLag_;
  }

  FX NLPSolverInternal::getGradLag(){
    FX gradLag;
    if(hasSetOption("grad_lag")){
      gradLag = getOption("grad_lag");
    } else {
      log("Generating/retrieving Lagrangian gradient function");
      gradLag = nlp_.derivative(0,1);
      log("Gradient function generated");
    }
    gradLag.setOption("name","grad_lag");
    gradLag.init(false);
    log("Gradient function initialized");
    return gradLag;
  }

  FX& NLPSolverInternal::hessLag(){
    if(hessLag_.isNull()){
      hessLag_ = getHessLag();
    }
    return hessLag_;
  }

  FX NLPSolverInternal::getHessLag(){
    FX hessLag;
    if(hasSetOption("hess_lag")){
      hessLag = getOption("hess_lag");
    } else {
      FX& gradLag = this->gradLag();
      log("Generating Hessian of the Lagrangian");
      hessLag = gradLag.jacobian(NL_X,NL_NUM_OUT+NL_X,false,true);
      log("Hessian function generated");
    }
    hessLag.setOption("name","hess_lag");
    hessLag.init(false);
    casadi_assert_message(hessLag.getNumInputs()==HESSLAG_NUM_IN, "Wrong number of inputs to the Hessian function. Note: The Lagrangian Hessian signature was changed in #544");
    casadi_assert_message(hessLag.getNumOutputs()==HESSLAG_NUM_OUT, "Wrong number of outputs to the Hessian function. Note: The Lagrangian Hessian signature was changed in #544");
    hessLag.setInputScheme(SCHEME_HessLagInput);
    hessLag.setOutputScheme(SCHEME_HessLagOutput);
    log("Hessian function initialized");
    return hessLag;
  }

  CCSSparsity& NLPSolverInternal::spHessLag(){
    if(spHessLag_.isNull()){
      spHessLag_ = getSpHessLag();
    }
    return spHessLag_;
  }

  CCSSparsity NLPSolverInternal::getSpHessLag(){
    CCSSparsity spHessLag;
    if(false /*hasSetOption("hess_lag_sparsity")*/){ // NOTE: No such option yet, need support for GenericType(CCSSparsity)
      //spHessLag = getOption("hess_lag_sparsity");
    } else {
      FX& gradLag = this->gradLag();
      log("Generating Hessian of the Lagrangian sparsity pattern");
      spHessLag = gradLag.jacSparsity(NL_X,NL_NUM_OUT+NL_X,false,true);
      log("Hessian sparsity pattern generated");
    }
    return spHessLag;
  }
  
  void NLPSolverInternal::checkInputs() const {
    for (int i=0;i<input(NLP_SOLVER_LBX).size();++i) {
      casadi_assert_message(input(NLP_SOLVER_LBX).at(i)<=input(NLP_SOLVER_UBX).at(i),"LBX[i] <= UBX[i] was violated for i=" << i << ". Got LBX[i]=" << input(NLP_SOLVER_LBX).at(i) << " and UBX[i]=" << input(NLP_SOLVER_UBX).at(i));
    }
    for (int i=0;i<input(NLP_SOLVER_LBG).size();++i) {
      casadi_assert_message(input(NLP_SOLVER_LBG).at(i)<=input(NLP_SOLVER_UBG).at(i),"LBG[i] <= UBG[i] was violated for i=" << i << ". Got LBG[i]=" << input(NLP_SOLVER_LBG).at(i) << " and UBG[i]=" << input(NLP_SOLVER_UBG).at(i));
    }
  }

} // namespace CasADi
