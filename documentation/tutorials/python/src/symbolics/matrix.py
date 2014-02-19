#
#     This file is part of CasADi.
# 
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
# 
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
# 
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
# 
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
# 
# 
#! CasADi tutorial 1
#! ==================
#! This tutorial file explains the use of CasADi's Matrix<T> in a python context.
#! Matrix<T> is a general class for sparse matrices. We inspect it with the help of Matrix<double>
#! Let's start with the import statements to load CasADi.
from casadi import *
from numpy import *
#! Contructors & printing
#! --------------------------------------
#! The python name for Matrix<double> is DMatrix
a = DMatrix(3,4)
print a
#! The string representation shows only the structural non-zero entries. In this case there are none.
#! Let's make a DMatrix with some structural non-zero entries.
w = DMatrix(3,4,[1,2,1],[0,2,2,3],[3,2.3,8])
print w
#! Internally, the Matrix<> class uses a Compressed Row Format which containts the offset to the first nonzero on each row ...
print "row offsets: ", w.rowind()
#! ... the columns for each nonzero ...
print "columns: ", w.col()
#! ... and the nonzero data entries:
print "nonzeros: ", w.data()
#! Conversion
#! --------------
#! DMatrix can easily be converted into other data formats
print list(w.data())
print tuple(w.data())
print w.toArray()
print array(w)
print w.toMatrix()
print matrix(w)
print w.toCsr_matrix()

