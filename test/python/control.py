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
from casadi import *
import casadi as c
from numpy import *
import unittest
from types import *
from helpers import *
import random

dplesolvers = []
try:
  dplesolvers.append((PsdIndefDpleSolver,{"linear_solver": CSparse}))
except:
  pass
  
  
try:
  dplesolvers.append((SimpleIndefDpleSolver,{"linear_solver": CSparse}))
except:
  pass
  
  
print dplesolvers

class ControlTests(casadiTestCase):
  
  @memory_heavy()
  def test_dple_small(self):
    
    for Solver, options in dplesolvers:
      for K in [1,2,3,4]:
        for n in [2,3]:
          numpy.random.seed(1)
          print (n,K)
          A_ = [DMatrix(numpy.random.random((n,n))) for i in range(K)]
          
          V_ = [mul(v.T,v) for v in [DMatrix(numpy.random.random((n,n))) for i in range(K)]]
          
          
          solver = Solver([sp_dense(n,n) for i in range(K)],[sp_dense(n,n) for i in range(K)])
          solver.setOption(options)
          solver.setOption("ad_mode","forward")
          solver.init()
          solver.setInput(horzcat(A_),DPLE_A)
          solver.setInput(horzcat(V_),DPLE_V)
          
          As = msym("A",n,K*n)
          Vs = msym("V",n,K*n)
          
          Vss = horzcat([(i+i.T)/2 for i in horzsplit(Vs,n) ])
          
          
          AA = blkdiag([c.kron(i,i) for i in horzsplit(As,n)])

          A_total = DMatrix.eye(n*n*K) - horzcat([AA[-n*n:,:],AA[:-n*n,:]])
          
          
          Pf = solve(A_total.T,vec(horzcat([Vss[-n:,:],Vss[:-n,:]])).T,CSparse).T
          P = Pf.reshape((K*n,n))
          #P = (P+P.T)/2
          
          refsol = MXFunction([As,Vs],[P])
          refsol.init()
          
          refsol.setInput(horzcat(A_),DPLE_A)
          refsol.setInput(horzcat(V_),DPLE_V)
          
          solver.evaluate()
          X = list(horzsplit(solver.output(),n))
          
          a0 = (mul([blkdiag(A_).T,blkdiag(X),blkdiag(A_)])+blkdiag(V_))
          
          def sigma(a):
            return a[1:] + [a[0]]
            
          a1 = blkdiag(sigma(X))

          self.checkarray(a0,a1)

          
          self.checkfx(solver,refsol,sens_der=False,hessian=False,evals=False)
  
  @memory_heavy()
  def test_dple_large(self):
    
    for Solver, options in dplesolvers:
      if "Simple" in str(Solver): continue
      for K in [1,2,3,4,5]:
        for n in [2,3,4,8,16,32]:
          numpy.random.seed(1)
          print (n,K)
          A_ = [DMatrix(numpy.random.random((n,n))) for i in range(K)]
          
          V_ = [mul(v.T,v) for v in [DMatrix(numpy.random.random((n,n))) for i in range(K)]]
          
          
          solver = Solver([sp_dense(n,n) for i in range(K)],[sp_dense(n,n) for i in range(K)])
          solver.setOption(options)
          solver.init()
          solver.setInput(horzcat(A_),DPLE_A)
          solver.setInput(horzcat(V_),DPLE_V)
          
          solver.evaluate()
          X = list(horzsplit(solver.output(),n))
          
          a0 = (mul([blkdiag(A_).T,blkdiag(X),blkdiag(A_)])+blkdiag(V_))
          
          def sigma(a):
            return a[1:] + [a[0]]
            
          a1 = blkdiag(sigma(X))

          self.checkarray(a0,a1,digits=7)
      
      

if __name__ == '__main__':
    unittest.main()
