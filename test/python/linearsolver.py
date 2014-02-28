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

lsolvers = []
try:
  lsolvers.append((CSparse,{}))
except:
  pass
  
try:
  lsolvers.append((LapackLUDense,{}))
except:
  pass
  
try:
  lsolvers.append((LapackQRDense,{}))
except:
  pass
  
#try:
#  lsolvers.append((SymbolicQR,{}))
#except:
#  pass

nsolvers = []
  
def nullspacewrapper(sp):
  a = ssym("a",sp)
  f = SXFunction([a],[nullspace(a)])
  f.init()
  return f
  
nsolvers.append((nullspacewrapper,{}))
  
print lsolvers

class LinearSolverTests(casadiTestCase):

  def test_nullspace(self):
  
    for A in  [
                  DMatrix([[1,1.3],[2,5],[1,0.5],[1.8,1.7]]),
                  DMatrix([[1,1.3],[2,5],[1,0.5]]),
                  DMatrix([[1,1.3],[2,5],[1,0.5],[0.2,0.3],[-0.3,0.7]]),
                  DMatrix([[1,0],[0,0],[0,1],[0,0]]),
                  DMatrix([[1.3,0,0.4,1],[0.2,0.1,11,0],[0,1,0,0],[0.7,0.9,0,0],[1.1,0.99,0,0]])
              ]:
      n ,m = A.shape
      for Solver, options in nsolvers:
        solver = Solver(A.T.sparsity())
        solver.setOption(options)
        solver.init()
        solver.setInput(A.T)

        solver.evaluate()
        
        self.checkarray(mul(A.T,solver.output()),DMatrix.zeros(m,n-m))
        self.checkarray(mul(solver.output().T,solver.output()),DMatrix.eye(n-m))
        
        solver.setOption("ad_mode","forward")
        solver.init()
        
        Jf = solver.jacobian()
        Jf.init()

        solver.setOption("ad_mode","reverse")
        solver.init()
        
        Jb = solver.jacobian()
        Jb.init()
        
        Jf.setInput(A.T)
        Jb.setInput(A.T)
        
        Jf.evaluate()
        Jb.evaluate()

        self.checkarray(Jf.output(),Jb.output())
        self.checkarray(Jf.output(1),Jb.output(1))
        
        d = solver.derivative(1,0)
        d.init()
        
        r = numpy.random.rand(*A.shape)
        
        d.setInput(A.T,0)
        d.setInput(r.T,1)
        
        d.evaluate()
        
        exact = d.getOutput(1)
        
        solver.setInput(A.T,0)
        solver.evaluate()
        nom = solver.getOutput()
        
        eps = 1e-6
        solver.setInput((A+eps*r).T,0)
        solver.evaluate()
        pert = solver.getOutput()
        
        fd = (pert-nom)/eps
        
        #print exact, fd
        
        #print numpy.linalg.svd(horzcat([exact, fd]).T)[1]
        
        #print "fd:", mul(fd.T,fd), numpy.linalg.eig(mul(fd.T,fd))[0]
        #print "exact:", mul(exact.T,exact), numpy.linalg.eig(mul(exact.T,exact))[0]
        #print "fd:", mul(fd,fd.T), numpy.linalg.eig(mul(fd,fd.T))[0]
        #print "exact:", mul(exact,exact.T), numpy.linalg.eig(mul(exact,exact.T))[0]
        
        V = numpy.random.rand(A.shape[0]-A.shape[1],A.shape[0]-A.shape[1])
        V = V+V.T
        print V
        #V = DMatrix.eye(A.shape[0]-A.shape[1])
        a = mul([nom,V,fd.T])+mul([fd,V,nom.T])
        b = mul([nom,V,exact.T])+mul([exact,V,nom.T])
        
        print "here:", a-b
        
        #self.checkarray(a,b,digits=5)
    
        V = numpy.random.rand(A.shape[0],A.shape[0])
        V = V+V.T
        V = DMatrix.eye(A.shape[0])
        a = mul([nom.T,V,fd])+mul([fd.T,V,nom])
        b = mul([nom.T,V,exact])+mul([exact.T,V,nom])
        
        self.checkarray(a,b,digits=5)
  
  def test_simple_solve(self):
    A_ = DMatrix([[3,7],[1,2]])
    b_ = DMatrix([1,0.5])
    
    A = msym("A",A_.sparsity())
    b = msym("b",b_.sparsity())
    
    for Solver, options in lsolvers:
      print Solver.creator
      C = solve(A,b,Solver,options)
      
      f = MXFunction([A,b],[C])
      f.init()
      f.setInput(A_,0)
      f.setInput(b_,1)
      f.evaluate()
      
      self.checkarray(f.output(),DMatrix([1.5,-0.5]))
      self.checkarray(mul(A_,f.output()),b_)

  def test_pseudo_inverse(self):
    numpy.random.seed(0)
    A_ = DMatrix(numpy.random.rand(4,6))
    
    A = msym("A",A_.sparsity())
    As = ssym("A",A_.sparsity())
    
    for Solver, options in lsolvers:
      print Solver.creator
      B = pinv(A,Solver,options)
      
      f = MXFunction([A],[B])
      f.init()
      f.setInput(A_,0)
      f.evaluate()
      
      self.checkarray(mul(A_,f.output()),DMatrix.eye(4))
      
      f = SXFunction([As],[pinv(As)])
      f.init()
      f.setInput(A_,0)
      f.evaluate()
      
      self.checkarray(mul(A_,f.output()),DMatrix.eye(4))
      
      trans(solve(mul(A,trans(A)),A,Solver,options))
      pinv(A_,Solver,options)
      
      #self.checkarray(mul(A_,pinv(A_,Solver,options)),DMatrix.eye(4))
      
    A_ = DMatrix(numpy.random.rand(3,5))
    
    A = msym("A",A_.sparsity())
    As = ssym("A",A_.sparsity())
    
    for Solver, options in lsolvers:
      print Solver.creator
      B = pinv(A,Solver,options)
      
      f = MXFunction([A],[B])
      f.init()
      f.setInput(A_,0)
      f.evaluate()
      
      self.checkarray(mul(A_,f.output()),DMatrix.eye(3)) 
      
      f = SXFunction([As],[pinv(As)])
      f.init()
      f.setInput(A_,0)
      f.evaluate()
      
      self.checkarray(mul(A_,f.output()),DMatrix.eye(3))
      
      #self.checkarray(mul(pinv(A_,Solver,options),A_),DMatrix.eye(3))
      
  def test_simple_solve_dmatrix(self):
    A = DMatrix([[3,7],[1,2]])
    b = DMatrix([1,0.5])
    for Solver, options in lsolvers:
      print Solver.creator
      C = solve(A,b,Solver,options)
      
      self.checkarray(C,DMatrix([1.5,-0.5]))
      self.checkarray(mul(A,DMatrix([1.5,-0.5])),b)
    
  def test_simple_trans(self):
    A = DMatrix([[3,1],[7,2]])
    for Solver, options in lsolvers:
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      solver.setInput(A,"A")

      solver.prepare()
      
      b = DMatrix([1,0.5])
      solver.setInput(b,"B")
      
      solver.solve(True)
      
      res = DMatrix([1.5,-0.5])
      self.checkarray(solver.output("X"),res)
      #   result' = A\b'               Ax = b

  def test_simple(self):
    A = DMatrix([[3,1],[7,2]])
    for Solver, options in lsolvers:
      print Solver
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      solver.setInput(A,"A")

      solver.prepare()
      
      b = DMatrix([1,0.5])
      solver.setInput(b,"B")
      
      solver.solve()
      
      res = DMatrix([-1.5,5.5])
      self.checkarray(solver.output("X"),res)
      #   result' = A'\b'             Ax = b

  def test_simple_fx_direct(self):
    A_ = DMatrix([[3,7],[1,2]]).T
    A = msym("A",A_.sparsity())
    b_ = DMatrix([1,0.5])
    b = msym("b",b_.sparsity())
    
    for Solver, options in lsolvers:
      print Solver
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      solver.setInput(A_,"A")
      solver.setInput(b_,"B")
      
      A_0 = A[0,0]
      A_1 = A[0,1]
      A_2 = A[1,0]
      A_3 = A[1,1]
      
      b_0 = b[0]
      b_1 = b[1]
      
      solution = MXFunction(linsolIn(A=A,B=b),[vertcat([(((A_3/((A_0*A_3)-(A_2*A_1)))*b_0)+(((-A_1)/((A_0*A_3)-(A_2*A_1)))*b_1)),((((-A_2)/((A_0*A_3)-(A_2*A_1)))*b_0)+((A_0/((A_0*A_3)-(A_2*A_1)))*b_1))])])
      solution.init()
      
      solution.setInput(A_,"A")
      solution.setInput(b_,"B")
      
      self.checkfx(solver,solution,fwd=False,adj=False,jacobian=False,evals=False)
       
  def test_simple_fx_indirect(self):
    A_ = DMatrix([[3,1],[7,2]])
    A = msym("A",A_.sparsity())
    b_ = DMatrix([1,0.5])
    b = msym("b",b_.sparsity())
    
    for Solver, options in lsolvers:
      print Solver
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      solver.setInput(A_,"A")
      solver.setInput(b_,"B")
      
      relay = MXFunction(linsolIn(A=A,B=b),solver.call(linsolIn(A=A,B=b)))
      relay.init()

      relay.setInput(A_,"A")
      relay.setInput(b_,"B")
      
      A_0 = A[0,0]
      A_1 = A[0,1]
      A_2 = A[1,0]
      A_3 = A[1,1]
      
      b_0 = b[0]
      b_1 = b[1]
      
      solution = MXFunction(linsolIn(A=A,B=b),[vertcat([(((A_3/((A_0*A_3)-(A_2*A_1)))*b_0)+(((-A_1)/((A_0*A_3)-(A_2*A_1)))*b_1)),((((-A_2)/((A_0*A_3)-(A_2*A_1)))*b_0)+((A_0/((A_0*A_3)-(A_2*A_1)))*b_1))])])
      solution.init()
      
      solution.setInput(A_,"A")
      solution.setInput(b_,"B")
      
      self.checkfx(solver,solution,fwd=False,adj=False,jacobian=False,evals=False)

  def test_simple_solve_node(self):
    A_ = DMatrix([[3,1],[7,2]])
    A = msym("A",A_.sparsity())
    b_ = DMatrix([1,0.5])
    b = msym("b",b_.sparsity())
    for Solver, options in lsolvers:
      print Solver
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      for tr in [True, False]:
        x = solver.solve(A,b,tr)
        f = MXFunction([A,b],[x])
        f.init()
        f.setInput(A_,0)
        f.setInput(b_,1)
        f.evaluate()

        if tr:
          A_0 = A[0,0]
          A_1 = A[1,0]
          A_2 = A[0,1]
          A_3 = A[1,1]
        else:
          A_0 = A[0,0]
          A_1 = A[0,1]
          A_2 = A[1,0]
          A_3 = A[1,1]
          
        b_0 = b[0]
        b_1 = b[1]
        
        solution = MXFunction([A,b],[vertcat([(((A_3/((A_0*A_3)-(A_2*A_1)))*b_0)+(((-A_1)/((A_0*A_3)-(A_2*A_1)))*b_1)),((((-A_2)/((A_0*A_3)-(A_2*A_1)))*b_0)+((A_0/((A_0*A_3)-(A_2*A_1)))*b_1))])])
        solution.init()
        
        solution.setInput(A_,0)
        solution.setInput(b_,1)
        
        self.checkfx(f,solution)


  def test_simple_solve_node_sparseA(self):
    A_ = DMatrix([[3,0],[7,2]])
    makeSparse(A_)
    A = msym("A",A_.sparsity())
    print A.size(), A_.size()
    b_ = DMatrix([1,0.5])
    b = msym("b",b_.sparsity())
    for Solver, options in lsolvers:
      print Solver
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      for tr in [True, False]:
        x = solver.solve(A,b,tr)
        f = MXFunction([A,b],[x])
        f.init()
        f.setInput(A_,0)
        f.setInput(b_,1)
        f.evaluate()

        if tr:
          A_0 = A[0,0]
          A_1 = A[1,0]
          A_2 = A[0,1]
          A_3 = A[1,1]
        else:
          A_0 = A[0,0]
          A_1 = A[0,1]
          A_2 = A[1,0]
          A_3 = A[1,1]
          
        b_0 = b[0]
        b_1 = b[1]
        
        solution = MXFunction([A,b],[vertcat([(((A_3/((A_0*A_3)-(A_2*A_1)))*b_0)+(((-A_1)/((A_0*A_3)-(A_2*A_1)))*b_1)),((((-A_2)/((A_0*A_3)-(A_2*A_1)))*b_0)+((A_0/((A_0*A_3)-(A_2*A_1)))*b_1))])])
        solution.init()
        
        solution.setInput(A_,0)
        solution.setInput(b_,1)
        
        self.checkfx(f,solution,sens_der=False,digits_sens=7)

  def test_simple_solve_node_sparseB(self):
    A_ = DMatrix([[3,1],[7,2]])
    A = msym("A",A_.sparsity())
    b_ = DMatrix([1,0])
    makeSparse(b_)
    b = msym("b",b_.sparsity())
    for Solver, options in lsolvers:
      print Solver
      solver = Solver(A.sparsity())
      solver.setOption(options)
      solver.init()
      for tr in [True, False]:
        x = solver.solve(A,b,tr)
        f = MXFunction([A,b],[x])
        f.init()
        f.setInput(A_,0)
        f.setInput(b_,1)
        f.evaluate()

        if tr:
          A_0 = A[0,0]
          A_1 = A[1,0]
          A_2 = A[0,1]
          A_3 = A[1,1]
        else:
          A_0 = A[0,0]
          A_1 = A[0,1]
          A_2 = A[1,0]
          A_3 = A[1,1]
          
        b_0 = b[0,0]
        b_1 = b[1,0]
        
        solution = MXFunction([A,b],[vertcat([(((A_3/((A_0*A_3)-(A_2*A_1)))*b_0)+(((-A_1)/((A_0*A_3)-(A_2*A_1)))*b_1)),((((-A_2)/((A_0*A_3)-(A_2*A_1)))*b_0)+((A_0/((A_0*A_3)-(A_2*A_1)))*b_1))])])
        solution.init()
        
        solution.setInput(A_,0)
        solution.setInput(b_,1)
        
        self.checkfx(f,solution,digits_sens=7)

  @requires("CSparseCholesky")
  def test_cholesky(self):
    numpy.random.seed(0)
    n = 10
    L = self.randDMatrix(n,n,sparsity=0.2) +  1.5*c.diag(range(1,n+1))
    L = L[sp_tril(n)]
    M = mul(L,L.T)
    b = self.randDMatrix(n,1)
    
    M.sparsity().spy()

    S = CSparseCholesky(M.sparsity())
    
    S.init()
    S.setInput(M)
    S.prepare()
    
    self.checkarray(M,M.T)
    
    C = S.getFactorization()
    self.checkarray(mul(C,C.T),M)
    self.checkarray(C,L)
    
    print C
    
    S.getFactorizationSparsity().spy()

    C = solve(M,b,CSparseCholesky)
    self.checkarray(mul(M,C),b)
    

  @requires("CSparseCholesky")
  def test_cholesky2(self):
    numpy.random.seed(0)
    n = 10
    L = c.diag(range(1,n+1))
    M = mul(L,L.T)

    print L
    S = CSparseCholesky(M.sparsity())
    

    S.init()
    S.getFactorizationSparsity().spy()
    S.setInput(M)
    S.prepare()

    C = S.getFactorization()
    self.checkarray(mul(C,C.T),M)
    
  def test_large_sparse(self):
    numpy.random.seed(1)
    n = 10
    A = self.randDMatrix(n,n,sparsity=0.5)
    b = self.randDMatrix(n,3,sparsity=0.5)
    
    As = msym("A",A.sparsity())
    bs = msym("B",b.sparsity())
    for Solver, options in lsolvers:
      print Solver.creator
      C = solve(A,b,Solver,options)
      
      self.checkarray(mul(A,C),b)
      
      f = MXFunction([As,bs],[solve(As,bs,Solver,options)])
      f.init()
      f.setInput(A,0)
      f.setInput(b,1)
      f.evaluate()
      
      self.checkarray(mul(A,f.output()),b)
      
  @known_bug()
  def test_large_sparse(self):
    numpy.random.seed(1)
    n = 10
    A = self.randDMatrix(n,n,sparsity=0.5)
    b = self.randDMatrix(n,3)
    
    As = msym("A",A.sparsity())
    bs = msym("B",b.sparsity())
    for Solver, options in lsolvers:
      print Solver.creator
      C = solve(A,b,Solver,options)
      
      self.checkarray(mul(A,C),b)
      
      for As_,A_ in [(full(As),full(A)),(full(As).T,full(A).T),(full(As.T),full(A.T)),(As.T,A.T)]:
        f = MXFunction([As,bs],[solve(As_,bs,Solver,options)])
        f.init()
        f.setInput(A_,0)
        f.setInput(b,1)
        f.evaluate()

        self.checkarray(mul(A,f.output()),b)
      
if __name__ == '__main__':
    unittest.main()
