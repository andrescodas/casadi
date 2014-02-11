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

#ifndef MULTIPLICATION_IMPL_HPP
#define MULTIPLICATION_IMPL_HPP

#include "multiplication.hpp"
#include "../matrix/matrix_tools.hpp"
#include "mx_tools.hpp"
#include "../stl_vector_tools.hpp"
#include "../fx/fx_internal.hpp"

using namespace std;

namespace CasADi{

  template<bool TrX, bool TrY>
  Multiplication<TrX,TrY>::Multiplication(const MX& z, const MX& x, const MX& y){
    casadi_assert_message(x.size1() == y.size1(),"Multiplication::Multiplication: dimension mismatch. Attempting to multiply " << x.dimString() << " with " << y.dimString());
    setDependencies(z,x,y);
    setSparsity(z.sparsity());
    casadi_assert_message(!TrX || TrY, "Illegal combination");
    casadi_assert_message(!TrX, "Not implemented");
    casadi_assert_message(TrY,"Not implemented");
  }

  template<bool TrX, bool TrY>
  void Multiplication<TrX,TrY>::printPart(std::ostream &stream, int part) const{
    if(part==0){
      stream << "(";
    } else if(part==1){
      stream << "+mul(";
    } else if(part==2){
      if(TrX) stream << "'";
      stream << ",";
    } else {
      if(TrY) stream << "'";
      stream << "))";
    }
  }

  template<bool TrX, bool TrY>
  void Multiplication<TrX,TrY>::evaluateD(const DMatrixPtrV& input, DMatrixPtrV& output, std::vector<int>& itmp, std::vector<double>& rtmp){
    evaluateGen<double,DMatrixPtrV,DMatrixPtrVV>(input,output,itmp,rtmp);
  }

  template<bool TrX, bool TrY>
  void Multiplication<TrX,TrY>::evaluateSX(const SXMatrixPtrV& input, SXMatrixPtrV& output, std::vector<int>& itmp, std::vector<SX>& rtmp){
    evaluateGen<SX,SXMatrixPtrV,SXMatrixPtrVV>(input,output,itmp,rtmp);
  }

  template<bool TrX, bool TrY>
  template<typename T, typename MatV, typename MatVV>
  void Multiplication<TrX,TrY>::evaluateGen(const MatV& input, MatV& output, std::vector<int>& itmp, std::vector<T>& rtmp){
    if(input[0]!=output[0]){
      copy(input[0]->begin(),input[0]->end(),output[0]->begin());
    }
    Matrix<T>::mul_no_alloc_tnQQQ(*input[2],*input[1],*output[0]);
  }

  template<bool TrX, bool TrY>
  void Multiplication<TrX,TrY>::evaluateMX(const MXPtrV& input, MXPtrV& output, const MXPtrVV& fwdSeed, MXPtrVV& fwdSens, const MXPtrVV& adjSeed, MXPtrVV& adjSens, bool output_given){
    if(!output_given)
      *output[0] = *input[0] + mul(tr<TrX>(*input[1]),tr<TrY>(*input[2]),(*input[0]).sparsity());

    // Forward sensitivities
    int nfwd = fwdSens.size();
    for(int d=0; d<nfwd; ++d){
      *fwdSens[d][0] = *fwdSeed[d][0] + mul(tr<TrX>(*fwdSeed[d][1]),tr<TrY>(*input[2]),(*input[0]).sparsity()) + mul(tr<TrX>(*input[1]),tr<TrY>(*fwdSeed[d][2]),(*input[0]).sparsity());
    }
  
    // Adjoint sensitivities
    int nadj = adjSeed.size();
    for(int d=0; d<nadj; ++d){
      *adjSens[d][1] += tr<TrX>(mul(*adjSeed[d][0],tr<!TrY>(*input[2]),tr<TrX>(*input[1]).sparsity()));
      *adjSens[d][2] += tr<TrY>(mul(tr<!TrX>(*input[1]),*adjSeed[d][0],tr<TrY>(*input[2]).sparsity()));
      if(adjSeed[d][0]!=adjSens[d][0]){
        *adjSens[d][0] += *adjSeed[d][0];
        *adjSeed[d][0] = MX();
      }
    }
  }

  template<bool TrX, bool TrY>
  void Multiplication<TrX,TrY>::propagateSparsity(DMatrixPtrV& input, DMatrixPtrV& output, bool fwd){
    bvec_t *zd = get_bvec_t(input[0]->data());
    bvec_t *rd = get_bvec_t(output[0]->data());
    const size_t n = this->size();
    if(fwd){
      if(zd!=rd) copy(zd,zd+n,rd);
      DMatrix::mul_sparsity<true>(*input[1],*input[2],*input[0]);
    } else {
      DMatrix::mul_sparsity<false>(*input[1],*input[2],*output[0]);
      if(zd!=rd){
        for(int i=0; i<n; ++i){
          zd[i] |= rd[i];
          rd[i] = bvec_t(0);
        }
      }
    }
  }

  template<bool TrX, bool TrY>
  void Multiplication<TrX,TrY>::generateOperation(std::ostream &stream, const std::vector<std::string>& arg, const std::vector<std::string>& res, CodeGenerator& gen) const{
    // Check if inplace
    bool inplace = arg.at(0).compare(res.front())==0;

    // Copy first argument if not inplace
    if(!inplace){      
      stream << "  for(i=0; i<" << this->size() << "; ++i) " << res.front() << "[i]=" << arg.at(0) << "[i];" << endl;
    }

    // Perform sparse matrix multiplication
    gen.addAuxiliary(CodeGenerator::AUX_MM_NT_SPARSE);
    stream << "  casadi_mm_nt_sparse(";
    stream << arg.at(1) << ",s" << gen.getSparsity(dep(1).sparsity()) << ",";
    stream << arg.at(2) << ",s" << gen.getSparsity(dep(2).sparsity()) << ",";
    stream << res.front() << ",s" << gen.getSparsity(sparsity()) << ");" << endl;
  }

  template<bool TrX, bool TrY>
  void DenseMultiplication<TrX,TrY>::generateOperation(std::ostream &stream, const std::vector<std::string>& arg, const std::vector<std::string>& res, CodeGenerator& gen) const{
    // Check if inplace
    bool inplace = arg.at(0).compare(res.front())==0;

    // Copy first argument if not inplace
    if(!inplace){      
      stream << "  for(i=0; i<" << this->size() << "; ++i) " << res.front() << "[i]=" << arg.at(0) << "[i];" << endl;
    }

    int ncol_x = this->dep(1).size2();
    int nrow_x = this->dep(1).size1();
    int ncol_y = this->dep(2).size2();
    stream << "  for(i=0, rr=" << res.front() <<"; i<" << ncol_x << "; ++i)";
    stream << " for(j=0; j<" << ncol_y << "; ++j, ++rr)";
    stream << " for(k=0, ss=" << arg.at(1) << "+i*" << nrow_x << ", tt=" << arg.at(2) << "+j*" << nrow_x << "; k<" << nrow_x << "; ++k)";
    stream << " *rr += *ss++**tt++;" << endl;
  }

} // namespace CasADi

#endif // MULTIPLICATION_IMPL_HPP

