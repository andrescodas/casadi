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

#include "nullspace_internal.hpp"
#include "fx_internal.hpp"
#include "../stl_vector_tools.hpp"
#include "sx_function.hpp"
#include "../sx/sx_tools.hpp"
#include "../mx/mx_tools.hpp"
#include "fx_tools.hpp"
#include <utility>
#include <string>

using namespace std;
namespace CasADi{

  
  NullspaceInternal::NullspaceInternal(const Sparsity& A_sp) : A_sp_(A_sp) {
    addOption("dense",        OT_BOOLEAN,       true, "Indicates that dense matrices can be assumed");
  
  }
  
  NullspaceInternal::~NullspaceInternal(){
  }


  void NullspaceInternal::init(){

    dense_ = getOption("dense");
    
    casadi_assert(!dense_ || A_sp_.dense());
    
    m_ = A_sp_.size1();
    n_ = A_sp_.size2();
    
    casadi_assert(m_<=n_);
    
    Sparsity dense;
    if (dense_) {
      dense = sp_dense(n_,n_-m_);
    }
  
    FXInternal::init();
    
    setNumInputs(1);
    input(0)  = DMatrix::zeros(A_sp_);
    
    setNumOutputs(1);
    if (dense_) {
      output(0)  = DMatrix::zeros(dense);
    }
  
  }

  void NullspaceInternal::evaluate(){
    throw CasadiException("NullspaceInternal::evaluate: Not implemented");
  }


} // namespace CasADi


