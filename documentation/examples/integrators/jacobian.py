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
#! Integrator jacobian
#! =====================
from casadi import *
from numpy import *

#! We will investigate the working of integrator jacobian with the help of the parametrically exited Duffing equation:
#!
#$ $\ddot{u}+\dot{u}-\epsilon (2 \mu \dot{u}+\alpha u^3+2 k u \cos(\Omega t))$ with $\Omega = 2 + \epsilon \sigma$.

t = ssym("t")

u = ssym("u") 
v = ssym("v") 

eps   = ssym("eps")
mu    = ssym("mu")
alpha = ssym("alpha")
k     = ssym("k")
sigma = ssym("sigma")
Omega = 2 + eps*sigma

params = vertcat([eps,mu,alpha,k,sigma])
states = vertcat([u,v])
rhs    = vertcat([v,-u-eps*(2*mu*v+alpha*u**3+2*k*u*cos(Omega*t))])

f=SXFunction(daeIn(x=states,p=params,t=t),daeOut(ode=rhs))
f.init()

integrator = CVodesIntegrator(f)

integrator.init()

#! First argument is input index, secpnd argument is output index
jac = integrator.jacobian("p", "xf")
