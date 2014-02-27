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

#include "custom_function_internal.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

namespace CasADi{

using namespace std;

CustomFunction::CustomFunction(){
}

CustomFunction::CustomFunction(const CustomEvaluate &c_fcn,const vector<Sparsity> &inputscheme,const  vector<Sparsity> &outputscheme) {
  assignNode(new CustomFunctionInternal(c_fcn,inputscheme,outputscheme));
}

CustomFunction::CustomFunction(const CustomEvaluate &c_fcn,const IOSchemeVector< Sparsity > &inputscheme,const  vector<Sparsity> &outputscheme) {
  assignNode(new CustomFunctionInternal(c_fcn,inputscheme,outputscheme));
  setInputScheme(inputscheme.scheme);
}

CustomFunction::CustomFunction(const CustomEvaluate &c_fcn,const vector<Sparsity> &inputscheme,const  IOSchemeVector< Sparsity > &outputscheme) {
  assignNode(new CustomFunctionInternal(c_fcn,inputscheme,outputscheme));
  setOutputScheme(outputscheme.scheme);
}

CustomFunction::CustomFunction(const CustomEvaluate &c_fcn,const IOSchemeVector< Sparsity > &inputscheme,const  IOSchemeVector< Sparsity > &outputscheme) {
  assignNode(new CustomFunctionInternal(c_fcn,inputscheme,outputscheme));
  setInputScheme(inputscheme.scheme);
  setOutputScheme(outputscheme.scheme);
}

CustomFunction::CustomFunction(const CustomEvaluate &c_fcn) {
  assignNode(new CustomFunctionInternal(c_fcn,vector<Sparsity>(),vector<Sparsity>()));
}

CustomFunctionInternal* CustomFunction::operator->(){
  return static_cast<CustomFunctionInternal*>(FX::operator->());
}

const CustomFunctionInternal* CustomFunction::operator->() const{
   return static_cast<const CustomFunctionInternal*>(FX::operator->()); 
}
  
bool CustomFunction::checkNode() const{
  return dynamic_cast<const CustomFunctionInternal*>(get())!=0;
}



} // namespace CasADi

