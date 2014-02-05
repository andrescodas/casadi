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

#include "symbolic_mx.hpp"
#include "../stl_vector_tools.hpp"

using namespace std;

namespace CasADi{

SymbolicMX::SymbolicMX(const std::string& name, int n, int m) : name_(name) {
  setSparsity(CCSSparsity(n,m,true));
}

SymbolicMX::SymbolicMX(const std::string& name, const CCSSparsity & sp) : name_(name) {
  setSparsity(sp);
}

SymbolicMX* SymbolicMX::clone() const{
  return new SymbolicMX(*this);
}

void SymbolicMX::printPart(std::ostream &stream, int part) const{
  stream << name_;
}

void SymbolicMX::evaluateD(const DMatrixPtrV& input, DMatrixPtrV& output, std::vector<int>& itmp, std::vector<double>& rtmp){
}

void SymbolicMX::evaluateSX(const SXMatrixPtrV& input, SXMatrixPtrV& output, std::vector<int>& itmp, std::vector<SX>& rtmp){
}

void SymbolicMX::evaluateMX(const MXPtrV& input, MXPtrV& output, const MXPtrVV& fwdSeed, MXPtrVV& fwdSens, const MXPtrVV& adjSeed, MXPtrVV& adjSens, bool output_given){
}

const std::string& SymbolicMX::getName() const{
  return name_;
}

void SymbolicMX::propagateSparsity(DMatrixPtrV& input, DMatrixPtrV& output, bool fwd){
  bvec_t *outputd = get_bvec_t(output[0]->data());
  fill_n(outputd,output[0]->size(),0);
}


} // namespace CasADi

