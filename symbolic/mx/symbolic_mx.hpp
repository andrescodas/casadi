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

#ifndef SYMBOLIC_MX_HPP
#define SYMBOLIC_MX_HPP

#include "mx_node.hpp"

namespace CasADi{
  /** \brief Represents a symbolic MX
      \author Joel Andersson 
      \date 2010
      A regular user is not supposed to work with this Node class.
      This user can call MX(name,n,m) directly.
  */
  class SymbolicMX : public MXNode{
  public:

    /** \brief  Constructors */
    explicit SymbolicMX(const std::string& name, int n=1, int m=1);

    /** \brief  Constructors */
    explicit SymbolicMX(const std::string& name, const CCSSparsity & sp);

    /// Destructor
    virtual ~SymbolicMX(){}

    /** \brief  Clone function */
    virtual SymbolicMX* clone() const;

    /** \brief  Print a part of the expression */
    virtual void printPart(std::ostream &stream, int part) const;

    /** \brief  Evaluate the function numerically */
    virtual void evaluateD(const DMatrixPtrV& input, DMatrixPtrV& output, std::vector<int>& itmp, std::vector<double>& rtmp);

    /** \brief  Evaluate the function symbolically (SX) */
    virtual void evaluateSX(const SXMatrixPtrV& input, SXMatrixPtrV& output, std::vector<int>& itmp, std::vector<SX>& rtmp);

    /** \brief  Evaluate the function symbolically (MX) */
    virtual void evaluateMX(const MXPtrV& input, MXPtrV& output, const MXPtrVV& fwdSeed, MXPtrVV& fwdSens, const MXPtrVV& adjSeed, MXPtrVV& adjSens, bool output_given);

    /** \brief  Propagate sparsity */
    virtual void propagateSparsity(DMatrixPtrV& input, DMatrixPtrV& output, bool fwd);

    /** \brief  Get the name */
    virtual const std::string& getName() const;

    /** \brief Get the operation */
    virtual int getOp() const{ return OP_PARAMETER;}

  protected:
    // Name of the varible
    std::string name_;
  };

} // namespace CasADi


#endif // SYMBOLIC_MX_HPP
