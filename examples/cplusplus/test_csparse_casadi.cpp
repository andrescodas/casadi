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

/** 
File superlu.c from the CSparse example collection
Joel Andersson, K.U. Leuven, 2010
*/

#include "symbolic/casadi.hpp"
#include "interfaces/csparse/csparse.hpp"

using namespace CasADi;
using namespace std;

int main(int argc, char *argv[])
{

  /* Initialize matrix A. */
  int ncol = 5, nrow = 5;
  int nnz = 12;
  
  vector<int> colind(ncol+1);
  vector<int> row(nnz);
  
  // Sparsity pattern
  row[0] = 0; row[1] = 1; row[2] = 4; row[3] = 1;
  row[4] = 2; row[5] = 4; row[6] = 0; row[7] = 2;
  row[8] = 0; row[9] = 3; row[10]= 3; row[11]= 4;
  colind[0] = 0; colind[1] = 3; colind[2] = 6; colind[3] = 8; colind[4] = 10; colind[5] = 12;
  CCSSparsity spA = CCSSparsity::QQQ(nrow,ncol,colind,row);
  
  // Create a solver instance
  CSparse linear_solver(spA);
    
  // Initialize
  linear_solver.init();

  // Pass Non-zero elements
  double   s, u, p, e, r, l;
  s = 19.0; u = 21.0; p = 16.0; e = 5.0; r = 18.0; l = 12.0;

  vector<double> val(nnz);
  val[0] = s; val[1] = l; val[2] = l; val[3] = u; val[4] = l; val[5] = l;
  val[6] = u; val[7] = p; val[8] = u; val[9] = e; val[10]= u; val[11]= r;
  
  // Right hand side
  vector<double> rhs(ncol,1.0);
  
  // Transpose?
  bool tr = false;

  // Solve
  linear_solver.setInput(val,"A");
  linear_solver.setInput(rhs,"B");
  linear_solver.prepare();
  linear_solver.solve(tr);
  
  // Print the solution
  cout << "solution = " << linear_solver.output("X") << endl;

  // Embed in an MX graph
  MX A = msym("A",spA);
  MX B = msym("B",1,ncol);
  MX X = linear_solver.solve(A,B,tr);
  MXFunction F(linsolIn("A",A,"B",B),linsolOut("X",X));
  F.init();

  // Solve
  F.setInput(val,"A");
  F.setInput(rhs,"B");
  F.evaluate();
  
  // Print the solution
  cout << "solution (embedded) = " << F.output("X") << endl;
  
  // Preturb the linear solver
  double t = 0.01;
  DMatrix x_unpreturbed = F.output("X");
  F.input("A")(2,3)   += 1*t;
  F.input("B")(0,2)   += 2*t;
  F.evaluate();
  cout << "solution (fd) = " << (F.output("X")-x_unpreturbed)/t << endl;

  // Jacobian
  FX J = F.jacobian("B","X");  
  J.init();
  J.setInput(val,"A");
  J.setInput(rhs,"B");
  J.evaluate();
  cout << "solution (dx/db) = " << J.output() << endl;
  DMatrix J_analytic = inv(J.input("A"));
  if(!tr) J_analytic = trans(J_analytic);
  cout << "analytic solution (dx/db) = " << J_analytic << endl;
}
