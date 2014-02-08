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

#ifndef SUBMATRIX_HPP
#define SUBMATRIX_HPP

namespace CasADi{


  /** SubMatrix class for Matrix 
      SubMatrix is the return type for operator() of the Matrix class, it allows access to the value as well as changing the parent object
      \author Joel Andersson 
      \date 2011
  */

  /// submatrix
  template<typename M, typename I, typename J>
  class SubMatrix : public M{
  public:
    /// Constructor
    SubMatrix(M& mat, const I& i, const J& j) : M(mat.sub(i,j)), mat_(mat), i_(i), j_(j){}

    //@{
    /// Methods that modify a part of the parent obejct (A(i,j) = ?, A(i,j) += ?, etc.)
    const M& operator=(const SubMatrix<M,I,J> &y);
    const M& operator=(const M &y);
    M operator+=(const M &y);
    M operator-=(const M &y);
    M operator*=(const M &y);
    M operator/=(const M &y);
    //@}
    
  private:
    /// A reference to the matrix that is allowed to be modified
    M& mat_;
    
    /// The element of the matrix that is allowed to be modified
    I i_;
    J j_;
  };

  // Implementation
  template<typename M, typename I, typename J>
  const M& SubMatrix<M,I,J>::operator=(const SubMatrix<M,I,J> &y) { 
    mat_.setSub(y,i_,j_); 
    return y;
  }

  // Implementation
  template<typename M, typename I, typename J>
  const M& SubMatrix<M,I,J>::operator=(const M &y) { 
    mat_.setSub(y,i_,j_); 
    return y;
  }

  template<typename M, typename I, typename J>
  M SubMatrix<M,I,J>::operator+=(const M &y){ 
    M s = *this+y;
    mat_.setSub(s,i_,j_); 
    return s;
  }

  template<typename M, typename I, typename J>
  M SubMatrix<M,I,J>::operator-=(const M &y){ 
    M s = *this-y;
    mat_.setSub(s,i_,j_); 
    return s;
  }

  template<typename M, typename I, typename J>
  M SubMatrix<M,I,J>::operator*=(const M &y){ 
    M s = *this*y;
    mat_.setSub(s,i_,j_); 
    return s;
  }

  template<typename M, typename I, typename J>
  M SubMatrix<M,I,J>::operator/=(const M &y){ 
    M s = *this/y;
    mat_.setSub(s,i_,j_); 
    return s;
  }


} // namespace CasADi


#endif // SUBMATRIX_HPP
