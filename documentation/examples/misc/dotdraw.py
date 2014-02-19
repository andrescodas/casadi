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
from casadi.tools import *

#! An SX graph
a = SX("a")
b = SX("b")

c = sin(a**5 + b)

c = c - b/ sqrt(fabs(c))
print c

dotdraw(c)

#! An SX Matrix

dotdraw(ssym("x",sp_tril(3)))

dotdraw(ssym("x",sp_tril(3))**2)

#! An MX graph
x = MX("x",sp_tril(2))
y = MX("y",sp_tril(2))

z = msym("z",4,2)

zz = x+y

dotdraw(zz)

f = MXFunction([z,y],[z+x[0],x-y])
f.setOption("name","magic")
f.init()

[z,z2] = f.call([vertcat([x,y]),zz.T])

z = z[:2,:] +x + cos(x) - sin(x) / tan(z2)

dotdraw(z)
