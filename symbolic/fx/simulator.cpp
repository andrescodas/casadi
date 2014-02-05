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

#include "simulator.hpp"
#include "simulator_internal.hpp"

using namespace std;
namespace CasADi{

Simulator::Simulator(){
}

Simulator::Simulator(const Integrator& integrator, const FX& output_fcn, const vector<double>& grid){
  assignNode(new SimulatorInternal(integrator,output_fcn,grid));
}

Simulator::Simulator(const Integrator& integrator, const FX& output_fcn, const Matrix<double>& grid){
  casadi_assert_message(grid.size1()==1,"Simulator::Simulator: grid must be of a row matrix shape, but got " << grid.dimString());
  casadi_assert_message(grid.dense(),"Simulator::Simulator: grid must be dense, but got " << grid.dimString());
  assignNode(new SimulatorInternal(integrator,output_fcn,grid.data()));
}

Simulator::Simulator(const Integrator& integrator, const vector<double>& grid){
  assignNode(new SimulatorInternal(integrator,FX(),grid));
}

Simulator::Simulator(const Integrator& integrator, const Matrix<double>& grid){
  casadi_assert_message(grid.size1()==1,"Simulator::Simulator: grid must be of a row matrix shape, but got " << grid.dimString());
  casadi_assert_message(grid.dense(),"Simulator::Simulator: grid must be dense, but got " << grid.dimString());
  assignNode(new SimulatorInternal(integrator,FX(),grid.data()));
}


SimulatorInternal* Simulator::operator->(){
  return (SimulatorInternal*)(FX::operator->());
}

const SimulatorInternal* Simulator::operator->() const{
   return (const SimulatorInternal*)(FX::operator->()); 
}

bool Simulator::checkNode() const{
  return dynamic_cast<const SimulatorInternal*>(get())!=0;
}


} // namespace CasADi

