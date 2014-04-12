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

#include "sx/sx_element.hpp"
#include "matrix/matrix.hpp"
#include "mx/mx.hpp"

#include "casadi_limits.hpp"

#include "weak_ref.hpp"
#include <iostream>

#include "function/schemes_helpers.hpp"

using namespace std;
namespace casadi{

  INSTANTIATE_SUBMATRIX(Matrix<SXElement>)
  INSTANTIATE_NONZEROS(Matrix<SXElement>)

  template class GenericMatrix< Matrix<SXElement> >;
  template class Matrix< SXElement >;

  INSTANTIATE_SUBMATRIX(MX)
  INSTANTIATE_NONZEROS(MX)

  INSTANTIATE_SUBMATRIX(Matrix<int>)
  INSTANTIATE_SUBMATRIX(Matrix<double>)
  INSTANTIATE_NONZEROS(Matrix<int>)
  INSTANTIATE_NONZEROS(Matrix<double>)

  template class GenericMatrix< Matrix<double> >;
  template class GenericMatrix< Matrix<int> >;

  template class Matrix<double>;
  template class Matrix<int>;

  template<class T>
  const T casadi_limits<T>::zero = T(0);

  template<class T>
  const T casadi_limits<T>::one = 1;

  template<class T>
  const T casadi_limits<T>::two = 2;

  template<class T>
  const T casadi_limits<T>::minus_one = -1;

  template class casadi_limits<double>;
  template class casadi_limits<int>;

  // Becuase we use Matrix<Sparsity>, Matrix<WeakRef> ...
  template const Sparsity  casadi_limits<Sparsity>::zero;
  template const WeakRef   casadi_limits<WeakRef>::zero;
  template void Matrix<WeakRef>::repr(std::ostream &) const;
  template void Matrix<WeakRef>::print(std::ostream &) const;
  template<> Matrix<WeakRef>::Matrix(const Matrix<WeakRef>&);


  INSTANTIATE_IOSCHEME_HELPERS(SX)
  INSTANTIATE_IOSCHEME_HELPERS(MX)
  INSTANTIATE_IOSCHEME_HELPERS(Sparsity)

} // namespace casadi

namespace std {
  #ifndef _MSC_VER
  template struct std::numeric_limits<casadi::SXElement>;
  #endif

} // namespace std
