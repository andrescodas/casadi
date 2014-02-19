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

#ifndef PSD_INDEF_DPLE_SOLVER_HPP
#define PSD_INDEF_DPLE_SOLVER_HPP

#include "../../control/dple_solver.hpp"

namespace CasADi{

  /// Forward declaration of internal class
  class PsdIndefDpleInternal;

  /**  @copydoc DPLE_doc
  
       Uses Periodic Schur Decomposition (psd) and does not assume positive definiteness.
       Based on Periodic Lyapunov equations: some applications and new algorithms. Int. J. Control, vol. 67, pp. 69-87, 1997. 
  
       \author Joris gillis
      \date 2014
      
  */
  class PsdIndefDpleSolver : public DpleSolver {
  public:
    /// Default constructor
    PsdIndefDpleSolver();
    
    /** \brief  Constructor
     *  \param[in] A  List of sparsities of A_i 
     *  \param[in] V  List of sparsities of V_i 
     */
    explicit PsdIndefDpleSolver(const std::vector< CRSSparsity > & A, const std::vector< CRSSparsity > &V);
    
    /// Access functions of the node
    PsdIndefDpleInternal* operator->();

    /// Access functions of the node
    const PsdIndefDpleInternal* operator->() const;
 
    /// Check if the node is pointing to the right type of object
    virtual bool checkNode() const;
  
    /// Static creator function
    #ifdef SWIG
    %callback("%s_cb");
    #endif
    static DpleSolver creator(const std::vector< CRSSparsity > & A, const std::vector< CRSSparsity > &V){ return PsdIndefDpleSolver(A,V);}
    #ifdef SWIG
    %nocallback;
    #endif
  
  };

} // namespace CasADi

#endif // DPLE_SOLVER_HPP
