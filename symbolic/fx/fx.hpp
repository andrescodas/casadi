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

#ifndef FX_HPP
#define FX_HPP

#include "io_interface.hpp"

namespace CasADi{
  
  /** Forward declaration of internal class */
  class FXInternal;
  
  /** \brief General function
      
      A general function \f$f\f$ in casadi can be multi-input, multi-output.\n
      Number of inputs:  nin    getNumInputs()\n
      Number of outputs: nout   getNumOutputs()\n
  
      We can view this function as a being composed of a (nin, nout) grid of single-input, single-output primitive functions.\n
      Each such primitive function \f$f_{i,j} \forall i \in [0,nin-1], j \in [0,nout-1]\f$ can map as \f$\mathbf{R}^{n,m}\to\mathbf{R}^{p,q}\f$, 
      in which n,m,p,q can take different values for every (i,j) pair.\n
  
      When passing input, you specify which partition i is active.     You pass the numbers flattened, as a vector of size \f$(n*m)\f$.\n
      When requesting output, you specify which partition j is active. You get the numbers flattened, as a vector of size \f$(p*q)\f$.\n
  
      To calculate jacobians, you need to have \f$(m=1,q=1)\f$.
  
      Write the jacobian as \f$J_{i,j} = \nabla f_{i,j} = \frac{\partial f_{i,j}(\flatten{x})}{\partial \flatten{x}}\f$.
  
      Using \f$\flatten{v} \in \mathbf{R}^n\f$ as a forward seed:  setFwdSeed(v,i)\n
      Retrieving \f$\flatten{s}_f \in \mathbf{R}^p\f$ from:        getFwdSens(sf,j)\n
  
      Using \f$\flatten{w} \in \mathbf{R}^p\f$ as a forward seed:  setAdjSeed(w,j)\n
      Retrieving \f$\flatten{s}_a \in \mathbf{R}^n \f$ from:        getAdjSens(sa,i)\n
  
      We have the following relationships for function mapping from a row vector to a row vector:
  
      \f$ \flatten{s}_f = \nabla f_{i,j} . \flatten{v}\f$ \n
      \f$ \flatten{s}_a = (\nabla f_{i,j})^T . \flatten{w}\f$
  
      Some quantities is these formulas must be transposed: \n 
      input  col: transpose \f$ \flatten{v} \f$ and \f$\flatten{s}_a\f$ \n
      output col: transpose \f$ \flatten{w} \f$ and \f$\flatten{s}_f\f$ \n
    
      NOTE: FX's are allowed to modify their input arguments when evaluating: implicitFunction, IDAS solver
      Futher releases may disallow this.
    
      \section Notes for developers
  
      Each function consists of 4 files:\n
      1. public class header file: imported in python\n
      2. public class implementation\n
      3. internal class header file: should only be used by derived classes\n
      4. internal class implementation\n
  
      python and c++ should be 1-to-1\n
      There should be no extra features in 1.\n
      All the functionality should exist in 1.\n
      If it means that c++ will be more "pythonic", so be it.
  
      \author Joel Andersson 
      \date 2010
  */
  class FX : public OptionsFunctionality, public IOInterface<FX>{
    
  public:
    /** \brief  default constructor */
    FX(); 

    /** \brief  Destructor */
    ~FX();
    
#ifndef SWIG
    /** \brief  Create from node */
    static FX create(FXInternal* node);
#endif // SWIG
    
    //@{
    /** \brief Access input/output scheme */
    const CasADi::IOScheme& inputScheme() const;
    const CasADi::IOScheme& outputScheme() const;
    CasADi::IOScheme& inputScheme();
    CasADi::IOScheme& outputScheme();
    //@}

    //@{
    /// Input/output structures of the function */
    const IOSchemeVector<DMatrix>& input_struct() const;
    const IOSchemeVector<DMatrix>& output_struct() const;
    IOSchemeVector<DMatrix>& input_struct();
    IOSchemeVector<DMatrix>& output_struct();
    //@}
  
    /** \brief  Get total number of nonzeros in all of the matrix-valued inputs */
    int getNumInputNonzeros() const;

    /** \brief  Get total number of nonzeros in all of the matrix-valued outputs */
    int getNumOutputNonzeros() const;

    /** \brief  Get total number of elements in all of the matrix-valued inputs */
    int getNumInputElements() const;

    /** \brief  Get total number of elements in all of the matrix-valued outputs */
    int getNumOutputElements() const;
  
    /** \brief Set input scheme */
    void setInputScheme(const CasADi::IOScheme &scheme);

    /** \brief Set output scheme */
    void setOutputScheme(const CasADi::IOScheme &scheme);

    /** \brief Get input scheme */
    CasADi::IOScheme getInputScheme() const;

    /** \brief Get output scheme */
    CasADi::IOScheme getOutputScheme() const;
    
    /** \brief  Evaluate */
    void evaluate();
  
    /// the same as evaluate()
    void solve();
    
    //@{
    /** \brief Generate a Jacobian function of output oind with respect to input iind
     * \param iind The index of the input
     * \param oind The index of the output
     *
     * The default behavior of this class is defined by the derived class.
     * If compact is set to true, only the nonzeros of the input and output expressions are considered.
     * If symmetric is set to true, the Jacobian being calculated is known to be symmetric (usually a Hessian),
     * which can be exploited by the algorithm.
     * 
     * The generated Jacobian has one more output than the calling function corresponding to the Jacobian and the same number of inputs.
     * 
     */
    FX jacobianQQQ(int iind=0, int oind=0, bool compact=false, bool symmetric=false);
    FX jacobianQQQ(const std::string& iind,  int oind=0, bool compact=false, bool symmetric=false) { return jacobianQQQ(inputSchemeEntry(iind),oind,compact,symmetric); }
    FX jacobianQQQ(int iind, const std::string& oind, bool compact=false, bool symmetric=false) { return jacobianQQQ(iind,outputSchemeEntry(oind),compact,symmetric); }
    FX jacobianQQQ(const std::string& iind, const std::string& oind, bool compact=false, bool symmetric=false) { return jacobianQQQ(inputSchemeEntry(iind),outputSchemeEntry(oind),compact,symmetric); }
    //@}

    /** Set the Jacobian function of output oind with respect to input iind
     NOTE: Does _not_ take ownership, only weak references to the Jacobians are kept internally */
    void setJacobianQQQ(const FX& jac, int iind=0, int oind=0, bool compact=false);
    
    //@{
    /** \brief Generate a gradient function of output oind with respect to input iind
     * \param iind The index of the input
     * \param oind The index of the output
     *
     * The default behavior of this class is defined by the derived class.
     * Note that the output must be scalar. In other cases, use the Jacobian instead.
     * 
     */
    FX gradient(int iind=0, int oind=0);
    FX gradient(const std::string& iind, int oind=0) { return gradient(inputSchemeEntry(iind),oind); }
    FX gradient(int iind, const std::string& oind) { return gradient(iind,outputSchemeEntry(oind)); }
    FX gradient(const std::string& iind, const std::string& oind) { return gradient(inputSchemeEntry(iind),outputSchemeEntry(oind)); }
    //@}

    //@{
    /** \brief Generate a tangent function of output oind with respect to input iind
     * \param iind The index of the input
     * \param oind The index of the output
     *
     * The default behavior of this class is defined by the derived class.
     * Note that the input must be scalar. In other cases, use the Jacobian instead.
     * 
     */
    FX tangent(int iind=0, int oind=0);
    FX tangent(const std::string& iind, int oind=0) { return tangent(inputSchemeEntry(iind),oind); }
    FX tangent(int iind, const std::string& oind) { return tangent(iind,outputSchemeEntry(oind)); }
    FX tangent(const std::string& iind, const std::string& oind) { return tangent(inputSchemeEntry(iind),outputSchemeEntry(oind)); }
    //@}
    
    //@{
    /** \brief Generate a Hessian function of output oind with respect to input iind 
     * \param iind The index of the input
     * \param oind The index of the output
     *
     * The generated Hessian has two more outputs than the calling function corresponding to the Hessian
     * and the gradients.
     * 
     */
    FX hessian(int iind=0, int oind=0);
    FX hessian(const std::string& iind, int oind=0) { return hessian(inputSchemeEntry(iind),oind); }
    FX hessian(int iind, const std::string& oind) { return hessian(iind,outputSchemeEntry(oind)); }
    FX hessian(const std::string& iind, const std::string& oind) { return hessian(inputSchemeEntry(iind),outputSchemeEntry(oind)); }
    //@}

    /** \brief Generate a Jacobian function of all the inputs elements with respect to all the output elements).
     */
    FX fullJacobianQQQ();

    /** Set the Jacobian of all the input nonzeros with respect to all output nonzeros
     NOTE: Does _not_ take ownership, only weak references to the Jacobian are kept internally */
    void setFullJacobianQQQ(const FX& jac);

#ifndef SWIG
    /** \brief  Create a function call (single input) */
    std::vector<MX> call(const MX &arg);
#endif // SWIG
  
    /** \brief  Create a function call (MX graph) */
    std::vector<MX> call(const std::vector<MX> &arg);

    /** \brief  Create a function call with directional derivatives 
     * Note: return by reference with SWIG
     */
#ifndef SWIG
    void call(const MXVector& arg, MXVector& res, 
              const MXVectorVector& fseed, MXVectorVector& fsens, 
              const MXVectorVector& aseed, MXVectorVector& asens);
#else // SWIG
    void call(const MXVector& arg, MXVector& OUTPUT, 
              const MXVectorVector& fseed, MXVectorVector& OUTPUT, 
              const MXVectorVector& aseed, MXVectorVector& OUTPUT);
#endif // SWIG
  
    /** \brief  Evaluate symbolically in parallel (matrix graph)
        paropt: Set of options to be passed to the Parallelizer
    */
    std::vector<std::vector<MX> > call(const std::vector<std::vector<MX> > &arg, const Dictionary& paropt=Dictionary());

  
    /// evaluate symbolically, SX type (overloaded)
    std::vector<SXMatrix> eval(const std::vector<SXMatrix>& arg){ return evalSX(arg);}

    /// evaluate symbolically, MX type (overloaded)
    std::vector<MX> eval(const std::vector<MX>& arg){return evalMX(arg);}
  
    /// evaluate symbolically, MX type (unambiguous)
    std::vector<MX> evalMX(const std::vector<MX>& arg);

    /// evaluate symbolically, SX type (unambiguous)
    std::vector<SXMatrix> evalSX(const std::vector<SXMatrix>& arg);
  
    /** \brief Evaluate symbolically with with directional derivatives, SX type
     * The first two arguments are the nondifferentiated inputs and results of the evaluation,
     * the next two arguments are a set of forward directional seeds and the resulting forward directional derivatives,
     * the length of the vector being the number of forward directions.
     * The next two arguments are a set of adjoint directional seeds and the resulting adjoint directional derivatives,
     * the length of the vector being the number of adjoint directions.
     */
#ifndef SWIG
    void evalSX(const SXMatrixVector& arg, SXMatrixVector& res, 
                const SXMatrixVectorVector& fseed, SXMatrixVectorVector& fsens, 
                const SXMatrixVectorVector& aseed, SXMatrixVectorVector& asens);
#else // SWIG
    void evalSX(const SXMatrixVector& arg, SXMatrixVector& OUTPUT, 
                const SXMatrixVectorVector& fseed, SXMatrixVectorVector& OUTPUT, 
                const SXMatrixVectorVector& aseed, SXMatrixVectorVector& OUTPUT);
#endif // SWIG

    /** \brief Evaluate symbolically with with directional derivatives, MX type
     * The first two arguments are the nondifferentiated inputs and results of the evaluation,
     * the next two arguments are a set of forward directional seeds and the resulting forward directional derivatives,
     * the length of the vector being the number of forward directions.
     * The next two arguments are a set of adjoint directional seeds and the resulting adjoint directional derivatives,
     * the length of the vector being the number of adjoint directions.
     */
#ifndef SWIG
    void evalMX(const MXVector& arg, MXVector& res, 
                const MXVectorVector& fseed, MXVectorVector& fsens, 
                const MXVectorVector& aseed, MXVectorVector& asens);
#else // SWIG
    void evalMX(const MXVector& arg, MXVector& OUTPUT, 
                const MXVectorVector& fseed, MXVectorVector& OUTPUT, 
                const MXVectorVector& aseed, MXVectorVector& OUTPUT);
#endif // SWIG  
              
    /** \brief Evaluate symbolically with with directional derivatives, SX type, overloaded
     * The first two arguments are the nondifferentiated inputs and results of the evaluation,
     * the next two arguments are a set of forward directional seeds and the resulting forward directional derivatives,
     * the length of the vector being the number of forward directions.
     * The next two arguments are a set of adjoint directional seeds and the resulting adjoint directional derivatives,
     * the length of the vector being the number of adjoint directions.
     */
#ifndef SWIG
    void eval(const SXMatrixVector& arg, std::vector<SXMatrix>& res, 
              const SXMatrixVectorVector& fseed, SXMatrixVectorVector& fsens, 
              const SXMatrixVectorVector& aseed, SXMatrixVectorVector& asens);
#else // SWIG
    void eval(const SXMatrixVector& arg, std::vector<SXMatrix>& OUTPUT, 
              const SXMatrixVectorVector& fseed, SXMatrixVectorVector& OUTPUT, 
              const SXMatrixVectorVector& aseed, SXMatrixVectorVector& OUTPUT);
#endif // SWIG 
    /** \brief Evaluate symbolically with with directional derivatives, MX type, overloaded
     * The first two arguments are the nondifferentiated inputs and results of the evaluation,
     * the next two arguments are a set of forward directional seeds and the resulting forward directional derivatives,
     * the length of the vector being the number of forward directions.
     * The next two arguments are a set of adjoint directional seeds and the resulting adjoint directional derivatives,
     * the length of the vector being the number of adjoint directions.
     */
#ifndef SWIG
    void eval(const MXVector& arg, MXVector& res, 
              const MXVectorVector& fseed, MXVectorVector& fsens, 
              const MXVectorVector& aseed, MXVectorVector& asens);
#else // SWIG
    void eval(const MXVector& arg, MXVector& OUTPUT, 
              const MXVectorVector& fseed, MXVectorVector& OUTPUT, 
              const MXVectorVector& aseed, MXVectorVector& OUTPUT);
#endif // SWIG 

#ifndef SWIG
    /// evaluate symbolically, single input, single output 
    SXMatrix eval(const SXMatrix& arg){ return eval(std::vector<SXMatrix>(1,arg)).at(0);}
#endif // SWIG
  
    /** \brief Get a function that calculates nfwd forward derivatives and nadj adjoint derivatives
     *         Returns a function with (1+nfwd)*n_in+nadj*n_out inputs
     *         and (1+nfwd)*n_out + nadj*n_in outputs.
     *         The first n_in inputs corresponds to nondifferentiated inputs. The next nfwd*n_in inputs
     *         corresponds to forward seeds, one direction at a time and the last nadj*n_out inputs
     *         corresponds to adjoint seeds, one direction at a time.
     *         The first n_out outputs corresponds to nondifferentiated outputs. The next nfwd*n_out outputs
     *         corresponds to forward sensitivities, one direction at a time and the last nadj*n_in outputs
     *         corresponds to  adjoint sensitivties, one direction at a time.
     *
     *         (n_in = getNumInputs(), n_out = getNumOutputs())
     *
     *        The functions returned are cached, meaning that if called multiple timed with the same value,
     *        then multiple references to the same function will be returned.
     */
    FX derivative(int nfwd, int nadj);

    /** Set a function that calculates nfwd forward dedrivatives and nadj adjoint derivatives
     NOTE: Does _not_ take ownership, only weak references to the derivatives are kept internally */
    void setDerivative(const FX& fcn, int nfwd, int nadj);

    //@{
    /// Get, if necessary generate, the sparsity of a Jacobian block
    CCSSparsity& jacSparsity(int iind=0, int oind=0, bool compact=false, bool symmetric=false);
    CCSSparsity& jacSparsity(const std::string &iind, int oind=0, bool compact=false, bool symmetric=false) { return jacSparsity(inputSchemeEntry(iind),oind,compact,symmetric); }
    CCSSparsity& jacSparsity(int iind, const std::string &oind, bool compact=false, bool symmetric=false) { return jacSparsity(iind,outputSchemeEntry(oind),compact,symmetric); }
    CCSSparsity& jacSparsity(const std::string &iind, const std::string &oind, bool compact=false, bool symmetric=false) { return jacSparsity(inputSchemeEntry(iind),outputSchemeEntry(oind),compact,symmetric); }
    //@}
    
    //@{
    /// Generate the sparsity of a Jacobian block
    void setJacSparsity(const CCSSparsity& sp, int iind, int oind, bool compact=false);
    void setJacSparsity(const CCSSparsity& sp, const std::string &iind, int oind, bool compact=false) { setJacSparsity(sp,inputSchemeEntry(iind),oind,compact); }
    void setJacSparsity(const CCSSparsity& sp, int iind, const std::string &oind, bool compact=false) { setJacSparsity(sp,iind,outputSchemeEntry(oind),compact); }
    void setJacSparsity(const CCSSparsity& sp, const std::string &iind, const std::string &oind, bool compact=false) { setJacSparsity(sp,inputSchemeEntry(iind),outputSchemeEntry(oind),compact); }
    //@}
    
    /** \brief Export / Generate C code for the function */
    void generateCode(const std::string& filename);
  
#ifndef SWIG 
    /// Construct a function that has only the k'th output
    FX operator[](int k) const;
#endif //SWIG 

    FX indexed_one_based(int k) const{ return operator[](k-1);}
    FX indexed_zero_based(int k) const{ return operator[](k);}
  
    /** \brief  Access functions of the node */
    FXInternal* operator->();

    /** \brief  Const access functions of the node */
    const FXInternal* operator->() const;

    /// Check if the node is pointing to the right type of object
    virtual bool checkNode() const;
    
    /// Get all statistics obtained at the end of the last evaluate call
    const Dictionary& getStats() const;

    /// Get a single statistic obtained at the end of the last evaluate call
    GenericType getStat(const std::string& name) const;

    /** \brief  Get a vector of symbolic variables with the same dimensions as the inputs
     * There is no guarantee that consecutive calls return identical objects
     */
    std::vector<MX> symbolicInput() const;
  
    /** \brief Get a vector of symbolic variables with the same dimensions as the inputs, SX graph
     * There is no guarantee that consecutive calls return identical objects
     */
    std::vector<SXMatrix> symbolicInputSX() const;

    /** \brief Is the class able to propate seeds through the algorithm? (for usage, see the example propagating_sparsity.cpp) */
    bool spCanEvaluate(bool fwd);

    /** \brief Reset the sparsity propagation (for usage, see the example propagating_sparsity.cpp) */
    void spInit(bool fwd);

    /** \brief Propagate the sparsity pattern through a set of directional derivatives forward or backward (for usage, see the example propagating_sparsity.cpp) */
    void spEvaluate(bool fwd);

    /** \brief Add modules to be monitored */
    void addMonitor(const std::string& mon);
  
    /** \brief Remove modules to be monitored */
    void removeMonitor(const std::string& mon);
    
    /** \brief Check if the numerical values of the supplied bounds make sense */
    void checkInputs() const;

  };
} // namespace CasADi

#ifdef SWIG
// Template instantiations
%template(FXVector)             std::vector<CasADi::FX>;
#endif // SWIG


#endif // FX_HPP
