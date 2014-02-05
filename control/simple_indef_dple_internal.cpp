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

#include "simple_indef_dple_internal.hpp"
#include <cassert>
#include "../symbolic/stl_vector_tools.hpp"
#include "../symbolic/matrix/matrix_tools.hpp"
#include "../symbolic/mx/mx_tools.hpp"
#include "../symbolic/sx/sx_tools.hpp"
#include "../symbolic/fx/mx_function.hpp"
#include "../symbolic/fx/sx_function.hpp"

#include <numeric>

INPUTSCHEME(DPLEInput)
OUTPUTSCHEME(DPLEOutput)

using namespace std;
namespace CasADi{

  SimpleIndefDpleInternal::SimpleIndefDpleInternal(const std::vector< CCSSparsity > & A, const std::vector< CCSSparsity > &V) : DpleInternal(A,V) {
  
    // set default options
    setOption("name","unnamed_simple_indef_dple_solver"); // name of the function 

    addOption("linear_solver",            OT_LINEARSOLVER, GenericType(), "User-defined linear solver class. Needed for sensitivities.");
    addOption("linear_solver_options",    OT_DICTIONARY,   GenericType(), "Options to be passed to the linear solver.");
    
  }

  SimpleIndefDpleInternal::~SimpleIndefDpleInternal(){ 

  }

  void SimpleIndefDpleInternal::init(){
  
    DpleInternal::init();

    casadi_assert_message(!pos_def_,"pos_def option set to True: Solver only handles the indefinite case.");
    casadi_assert_message(const_dim_,"const_dim option set to False: Solver only handles the True case.");
    
    n_ = A_[0].size2();
    
    
    MX As = msym("A",K_*n_,n_);
    MX Vs = msym("V",K_*n_,n_);
    
    std::vector< MX > Vss = horzsplit(Vs,n_);
    std::vector< MX > Ass = horzsplit(As,n_);
    
    for (int k=0;k<K_;++k) {
      Vss[k]=(Vss[k]+trans(Vss[k]))/2;
    }
    
    std::vector< MX > AA_list(K_);
    for (int k=0;k<K_;++k) {
      AA_list[k] = kron(Ass[k],Ass[k]);
    }
    
    MX AA = blkdiag(AA_list);
    
    MX A_total = DMatrix::eye(n_*n_*K_)-horzcat(AA(range(K_*n_*n_-n_*n_,K_*n_*n_),range(K_*n_*n_)),AA(range(K_*n_*n_-n_*n_),range(K_*n_*n_)));

    std::vector<MX> Vss_shift;
    Vss_shift.push_back(Vss.back());
    Vss_shift.insert(Vss_shift.end(),Vss.begin(),Vss.begin()+K_-1);
    
    MX Pf = solve(A_total,vec(horzcat(Vss_shift)),getOption("linear_solver"));
          
    MX P = reshape(Pf,K_*n_,n_);
    
    std::vector<MX> v_in;
    v_in.push_back(As);
    v_in.push_back(Vs);
    f_ = MXFunction(v_in,P);
    f_.setInputScheme(SCHEME_DPLEInput);
    f_.setOutputScheme(SCHEME_DPLEOutput);
    f_.init();
  }
  
  
  
  void SimpleIndefDpleInternal::evaluate(){
    for (int i=0;i<getNumInputs();++i) {
      std::copy(input(i).begin(),input(i).end(),f_.input(i).begin());
    }
    f_.evaluate();
    for (int i=0;i<getNumOutputs();++i) {
      std::copy(f_.output(i).begin(),f_.output(i).end(),output(i).begin());
    }
  }
  
  FX SimpleIndefDpleInternal::getDerivative(int nfwd, int nadj) {
    return f_.derivative(nfwd,nadj);
  }


  void SimpleIndefDpleInternal::deepCopyMembers(std::map<SharedObjectNode*,SharedObject>& already_copied){
    DpleInternal::deepCopyMembers(already_copied);
  }
  
  SimpleIndefDpleInternal* SimpleIndefDpleInternal::clone() const{
    // Return a deep copy
    SimpleIndefDpleInternal* node = new SimpleIndefDpleInternal(A_,V_);
    node->setOption(dictionary());
    return node;
  }


} // namespace CasADi


