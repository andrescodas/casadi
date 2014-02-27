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

#ifndef INTEGRATOR_INTERNAL_HPP
#define INTEGRATOR_INTERNAL_HPP

#include "integrator.hpp"
#include "fx_internal.hpp"

namespace CasADi{

  /** \brief Internal storage for integrator related data

      @copydoc DAE_doc
      \author Joel Andersson 
      \date 2010
  */
  class IntegratorInternal : public FXInternal{
  public:
    /** \brief  Constructor */
    IntegratorInternal(const FX& f, const FX& g);

    /** \brief  Destructor */
    virtual ~IntegratorInternal()=0;

    /** \brief  Clone */
    virtual IntegratorInternal* clone() const=0;

    /** \brief  Deep copy data members */
    virtual void deepCopyMembers(std::map<SharedObjectNode*,SharedObject>& already_copied);
  
    /** \brief  Create a new integrator */
    virtual IntegratorInternal* create(const FX& f, const FX& g) const = 0;
  
    /** \brief  Print solver statistics */
    virtual void printStats(std::ostream &stream) const{}

    /** \brief  Reset the forward problem and bring the time back to t0 */
    virtual void reset();

    /** \brief  Reset the backward problem and take time to tf */
    virtual void resetB();

    /** \brief  Integrate forward until a specified time point */
    virtual void integrate(double t_out) = 0;

    /** \brief  Integrate backward until a specified time point */
    virtual void integrateB(double t_out) = 0;

    /** \brief  evaluate */
    virtual void evaluate();

    /** \brief  Initialize */
    virtual void init();

    /** \brief  Propagate the sparsity pattern through a set of directional derivatives forward or backward */
    virtual void spEvaluate(bool fwd);

    /// Is the class able to propate seeds through the algorithm?
    virtual bool spCanEvaluate(bool fwd){ return true;}
    
    /// Generate a function that calculates nfwd forward derivatives and nadj adjoint derivatives
    virtual FX getDerivative(int nfwd, int nadj);

    /** \brief Calculate the jacobian of output oind with respect to input iind */
    virtual FX getJacobian(int iind, int oind, bool compact, bool symmetric);

    // Helper structure
    struct AugOffset{
      std::vector<int> x,z,q,p,rx,rz,rq,rp;
    };

    /** \brief Set solver specific options to generated augmented integrators */
    virtual void setDerivativeOptions(Integrator& integrator, const AugOffset& offset);

    /// Generate a augmented DAE system with nfwd forward sensitivities and nadj adjoint sensitivities
    virtual std::pair<FX,FX> getAugmented(int nfwd, int nadj, AugOffset& offset);
  
    // Get offsets in augmented problem
    AugOffset getAugOffset(int nfwd, int nadj);

    // Create sparsity pattern of the extended Jacobian (forward problem)
    Sparsity spJacF();

    // Create sparsity pattern of the extended Jacobian (backward problem)
    Sparsity spJacG();
  
    //@{
    // Shorthands
    DMatrix&  x0(){ return input(INTEGRATOR_X0 );}
    DMatrix&   p(){ return input(INTEGRATOR_P  );}
    DMatrix&  z0(){ return input(INTEGRATOR_Z0 );}
    DMatrix& rx0(){ return input(INTEGRATOR_RX0);}
    DMatrix&  rp(){ return input(INTEGRATOR_RP );}
    DMatrix& rz0(){ return input(INTEGRATOR_RZ0);}
    DMatrix&  xf(){ return output(INTEGRATOR_XF );}
    DMatrix&  qf(){ return output(INTEGRATOR_QF );}
    DMatrix&  zf(){ return output(INTEGRATOR_ZF );}
    DMatrix& rxf(){ return output(INTEGRATOR_RXF);}
    DMatrix& rqf(){ return output(INTEGRATOR_RQF);}
    DMatrix& rzf(){ return output(INTEGRATOR_RZF);}
    //@}

    /// Number of states for the forward integration
    int nx_, nz_, nq_;
  
    /// Number of states for the backward integration
    int nrx_, nrz_, nrq_;

    /// Number of forward and backward parameters
    int np_, nrp_;

    /// Integration horizon
    double t0_, tf_;

    // Current time
    double t_;
  
    /// ODE/DAE forward integration function
    FX f_;
  
    /// ODE/DAE backward integration function, if any
    FX g_;

    /// Integrator for sparsity pattern propagation
    LinearSolver linsol_f_, linsol_g_;

    /// Options
    bool print_stats_;

  };
  
} // namespace CasADi

#endif // INTEGRATOR_INTERNAL_HPP
