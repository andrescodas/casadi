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

#ifndef GENERIC_MATRIX_HPP
#define GENERIC_MATRIX_HPP

#include "slice.hpp"
#include "submatrix.hpp"
#include "nonzeros.hpp"
#include "ccs_sparsity.hpp"
#include "sparsity_tools.hpp"
#include "../casadi_math.hpp"

namespace CasADi{

  /** Sparsity format for getting and setting inputs and outputs */
  enum SparsityType{SPARSE,SPARSESYM,DENSE,DENSESYM,DENSETRANS};

  /** \brief Matrix base class
      This is a common base class for MX and Matrix<>, introducing a uniform syntax and implementing
      common functionality using the curiously recurring template pattern (CRTP) idiom.\n
  
      The class is designed with the idea that "everything is a matrix", that is, also scalars and vectors.\n
      This philosophy makes it easy to use and to interface in particularily with Python and Matlab/Octave.\n
  
      The syntax tries to stay as close as possible to the ublas syntax  when it comes to vector/matrix operations.\n

      Index starts with 0.\n
      Index vec happens as follows: (rr,cc) -> k = rr+cc*size1()\n
      Vectors are column vectors.\n
  
      The storage format is Compressed Column Storage (CCS), similar to that used for sparse matrices in Matlab, \n
      but unlike this format, we do allow for elements to be structurally non-zero but numerically zero.\n
  
      The sparsity pattern, which is reference counted and cached, can be accessed with CCSSparsity& sparsity()\n
  
      \author Joel Andersson 
      \date 2012    
  */
  template<typename MatType>
  class GenericMatrix{
  public:
    
    /** \brief Get the number of (structural) non-zero elements */
    int size() const;

    /** \brief Get the number of non-zeros in the lower triangular half */
    int sizeL() const;

    /** \brief Get the number of non-zeros in the upper triangular half */
    int sizeU() const;

    /** \brief Get get the number of non-zeros on the diagonal */
    int sizeD() const;
    
    /** \brief Get the number of elements */
    int numel() const;

    /** \brief Get the first dimension (i.e. number of rows) */
    int size1() const;

    /** \brief Get the second dimension (i.e. number of columns) */
    int size2() const;

    /** \brief Get the number if non-zeros for a given sparsity pattern */
    int size(SparsityType sp) const;
    
    /** \brief Get string representation of dimensions.
        The representation is (nrow x ncol = numel | size)
    */
    std::string dimString() const;
    
#ifndef SWIG  
    /** \brief  Get the shape */
    std::pair<int,int> shape() const;
#endif

    /** \brief  Check if the matrix expression is empty, i.e. one of its dimensions is 0 */
    bool empty() const;
    
    /** \brief  Check if the matrix expression is null, i.e. its dimensios are 0-by-0 */
    bool null() const;
    
    /** \brief  Check if the matrix expression is dense */
    bool dense() const;
    
    /** \brief  Check if the matrix expression is scalar */
    bool scalar(bool scalar_and_dense=false) const;

    /** \brief  Check if the matrix expression is square */
    bool square() const;

    /** \brief  Check if the matrix is a vector (i.e. size2()==1) */
    bool vector() const;

    /** \brief Get the sparsity pattern */
    const CCSSparsity& sparsity() const;

    /** \brief Access the sparsity, make a copy if there are multiple references to it */
    CCSSparsity& sparsityRef();

#ifndef SWIG
    /** \brief  Get vector nonzero or slice of nonzeros */
    template<typename K>
    const MatType operator[](const K& k) const{ return static_cast<const MatType*>(this)->getNZ(k); }

    /** \brief  Access vector nonzero or slice of nonzeros */
    template<typename K>
    NonZeros<MatType,K> operator[](const K& k){ return NonZeros<MatType,K>(static_cast<MatType&>(*this),k); }

    /** \brief  Get vector element or slice */
    template<typename RR>
    const MatType operator()(const RR& rr) const{ return static_cast<const MatType*>(this)->sub(rr,0);}

    /** \brief  Get Sparsity slice */
    const MatType operator()(const CCSSparsity& sp) const{ return static_cast<const MatType*>(this)->sub(sp); }
    
    /** \brief  Get Matrix element or slice */
    template<typename RR, typename CC>
    const MatType operator()(const RR& rr, const CC& cc) const{ return static_cast<const MatType*>(this)->sub(rr,cc); }

    /** \brief  Access vector element or slice */
    template<typename RR>
    SubMatrix<MatType,RR,int> operator()(const RR& rr){ return SubMatrix<MatType,RR,int>(static_cast<MatType&>(*this),rr,0); }

    /** \brief  Access Sparsity slice */
    SubMatrix<MatType,CCSSparsity,int> operator()(const CCSSparsity& sp){ return SubMatrix<MatType,CCSSparsity,int>(static_cast<MatType&>(*this),sp,0); }
      
    /** \brief  Access Matrix element or slice */
    template<typename RR, typename CC>
    SubMatrix<MatType,RR,CC> operator()(const RR& rr, const CC& cc){ return SubMatrix<MatType,RR,CC>(static_cast<MatType&>(*this),rr,cc); }
#endif // SWIG

    /** \brief Create an n-by-m matrix with symbolic variables */
    static MatType sym(const std::string& name, int nrow=1, int ncol=1);

    /** \brief Create a vector of length p with with matrices with symbolic variables of given sparsity */
    static std::vector<MatType > sym(const std::string& name, const CCSSparsity& sp, int p);

    /** \brief Create a vector of length p with nrow-by-ncol matrices with symbolic variables */
    static std::vector<MatType > sym(const std::string& name, int nrow, int ncol, int p);

    /** \brief Create an matrix with symbolic variables, given a sparsity pattern */
    static MatType sym(const std::string& name, const CCSSparsity& sp);
    
    /** \brief Matrix-matrix multiplication.
     * Attempts to identify quick returns on matrix-level and 
     * delegates to MatType::mul_full if no such quick returns are found.
     */
    MatType mul_smart(const MatType& y, const CCSSparsity& sp_z) const;
    
  };

#ifndef SWIG
  // Implementations

  template<typename MatType>
  const CCSSparsity& GenericMatrix<MatType>::sparsity() const{
    return static_cast<const MatType*>(this)->sparsity();
  }

  template<typename MatType>
  CCSSparsity& GenericMatrix<MatType>::sparsityRef(){
    return static_cast<MatType*>(this)->sparsityRef();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::size() const{
    return sparsity().size();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::sizeL() const{
    return sparsity().sizeL();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::sizeU() const{
    return sparsity().sizeU();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::sizeD() const{
    return sparsity().sizeD();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::numel() const{
    return sparsity().numel();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::size1() const{
    return sparsity().size1();
  }

  template<typename MatType>
  int GenericMatrix<MatType>::size2() const{
    return sparsity().size2();
  }

  template<typename MatType>
  std::pair<int,int> GenericMatrix<MatType>::shape() const{
    return sparsity().shape();
  }

  template<typename MatType>
  std::string GenericMatrix<MatType>::dimString() const {
    return sparsity().dimString();
  }

  template<typename MatType>
  bool GenericMatrix<MatType>::empty() const{
    return numel()==0;
  }

  template<typename MatType>
  bool GenericMatrix<MatType>::null() const{
    return size2()==0 && size1()==0;
  }

  template<typename MatType>
  bool GenericMatrix<MatType>::dense() const{
    return numel()==size();
  }

  template<typename MatType>
  bool GenericMatrix<MatType>::scalar(bool scalar_and_dense) const{
    return sparsity().scalar(scalar_and_dense);
  }

  template<typename MatType>
  bool GenericMatrix<MatType>::square() const{
    return sparsity().square();
  }

  template<typename MatType>
  bool GenericMatrix<MatType>::vector() const{
    return sparsity().vector();
  }

  template<typename MatType>
  MatType GenericMatrix<MatType>::mul_smart(const MatType& y, const CCSSparsity &sp_z) const {
    const MatType& x = *static_cast<const MatType*>(this);
  
    if (!(x.scalar() || y.scalar())) {
      casadi_assert_message(size2()==y.size1(),"Matrix product with incompatible dimensions. Lhs is " << dimString() << " and rhs is " << y.dimString() << ".");
    }
  
    // Check if we can simplify the product
    if(isIdentity(x)){
      return y;
    } else if(isIdentity(y)){
      return x;
    } else if(isZero(x) || isZero(y)){
      // See if one of the arguments can be used as result
      if(y.size()==0 && x.size2()==x.size1()) {
        return y;
      } else if(x.size()==0 && y.size2()==y.size1()) {
        return x;
      } else {
        if (y.size()==0 || x.size()==0 || x.empty() || y.empty()) {
          return MatType::sparse(x.size1(),y.size2());
        } else {
          return MatType::zeros(x.size1(),y.size2());
        }
      }
    } else if(x.scalar() || y.scalar()){
      return x*y;
    } else {
      return x.mul_full(y,sp_z);
    }
  }

  template<typename MatType>
  int GenericMatrix<MatType>::size(SparsityType sp) const{
    if(sp==SPARSE){
      return size();
    } else if(sp==SPARSESYM){
      return sizeU();
    } else if(sp==DENSE){
      return numel();
    } else if(sp==DENSESYM){
      return (numel()+size2())/2;
    } else {
      throw CasadiException("Matrix<T>::size(Sparsity): unknown sparsity");
    }
  }


#endif // SWIG

  template<typename MatType>
  MatType GenericMatrix<MatType>::sym(const std::string& name, int nrow, int ncol){ return sym(name,sp_dense(nrow,ncol));}

  template<typename MatType>
  std::vector<MatType> GenericMatrix<MatType>::sym(const std::string& name, const CCSSparsity& sp, int p){
    std::vector<MatType> ret(p);
    std::stringstream ss;
    for(int k=0; k<p; ++k){
      ss.str("");
      ss << name << k;
      ret[k] = sym(ss.str(),sp);
    }
    return ret;
  }

  template<typename MatType>
  std::vector<MatType > GenericMatrix<MatType>::sym(const std::string& name, int nrow, int ncol, int p){ return sym(name,sp_dense(nrow,ncol),p);}

  template<typename MatType>
  MatType GenericMatrix<MatType>::sym(const std::string& name, const CCSSparsity& sp){ throw CasadiException("\"sym\" not defined for instantiation");}

} // namespace CasADi

#endif // GENERIC_MATRIX_HPP

