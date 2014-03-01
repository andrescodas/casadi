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

#ifndef SPARSITY_INTERNAL_HPP
#define SPARSITY_INTERNAL_HPP

#include "sparsity.hpp"

namespace CasADi{

  class SparsityInternal : public SharedObjectNode{
  public:    
    /// Construct a sparsity pattern from vectors
    SparsityInternal(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row) : nrow_(nrow), ncol_(ncol), colind_(colind), row_(row) { sanityCheck(false); }
    
    /// Check if the dimensions and colind,row vectors are compatible
    void sanityCheck(bool complete=false) const;

    /// Get the diagonal of the matrix/create a diagonal matrix (mapping will contain the nonzero mapping)
    Sparsity diag(std::vector<int>& mapping) const;

    /// Calculate the elimination tree: See cs_etree in CSparse
    std::vector<int> eliminationTree(bool ata) const;
    
    /// Find strongly connected components: See cs_dfs in CSparse
    int depthFirstSearch(int j, int top, std::vector<int>& xi, std::vector<int>& pstack, const std::vector<int>& pinv, std::vector<bool>& marked) const;

    /// Find the strongly connected components of a square matrix: See cs_scc in CSparse
    int stronglyConnectedComponents(std::vector<int>& p, std::vector<int>& r) const;

    /// Transpose the matrix
    Sparsity transpose() const;

    /// Transpose the matrix and get the reordering of the non-zero entries, i.e. the non-zeros of the original matrix for each non-zero of the new matrix
    Sparsity transpose(std::vector<int>& mapping, bool invert_mapping=false) const;

    /// Check if the sparsity is the transpose of another
    bool isTranspose(const SparsityInternal& y) const;

    /// Breadth-first search for coarse decomposition: see cs_bfs in CSparse
    void breadthFirstSearch(int n, std::vector<int>& wi, std::vector<int>& wj, std::vector<int>& queue, const std::vector<int>& imatch, const std::vector<int>& jmatch, int mark) const;
    
    /// Collect matched cols and rows into p and q: see cs_matched in CSparse
    static void matched(int n, const std::vector<int>& wj, const std::vector<int>& imatch, std::vector<int>& p, std::vector<int>& q, std::vector<int>& cc, std::vector<int>& rr, int set, int mark);
    
    /// Collect unmatched cols into the permutation vector p : see cs_unmatched in CSparse
    static void unmatched(int m, const std::vector<int>& wi, std::vector<int>& p, std::vector<int>& rr, int set);
    
    /// return 1 if col i is in R2 : see cs_rprune in CSparse
    static int rprune (int i, int j, double aij, void *other);

    /// drop entries for which fkeep(A(i,j)) is false; return nz if OK, else -1: : see cs_fkeep in CSparse
    int drop(int (*fkeep) (int, int, double, void *), void *other);

    /// Compute the Dulmage-Mendelsohn decomposition : see cs_dmperm in CSparse
    int dulmageMendelsohn(std::vector<int>& rowperm, std::vector<int>& colperm, std::vector<int>& rowblock, std::vector<int>& colblock, std::vector<int>& coarse_rowblock, std::vector<int>& coarse_colblock, int seed) const{
      return transpose()->dulmageMendelsohnUpper(colperm,rowperm,colblock,rowblock,coarse_colblock,coarse_rowblock,seed);
    }

    /// Compute the Dulmage-Mendelsohn decomposition -- upper triangular TODO: refactor and merge with the above
    int dulmageMendelsohnUpper(std::vector<int>& rowperm, std::vector<int>& colperm, std::vector<int>& rowblock, std::vector<int>& colblock, std::vector<int>& coarse_rowblock, std::vector<int>& coarse_colblock, int seed) const;
    
    /// Compute the maximum transversal (maximum matching): see cs_maxtrans in CSparse
    void maxTransversal(std::vector<int>& imatch, std::vector<int>& jmatch, Sparsity& trans, int seed) const;
    
    /// Find an augmenting path: see cs_augment in CSparse
    void augmentingPath(int k, std::vector<int>& jmatch, int *cheap, std::vector<int>& w, int *js, int *is, int *ps) const;
    
    /// return a random permutation vector, the identity perm, or p = n-1:-1:0.  seed = -1 means p = n-1:-1:0.  seed = 0 means p = identity.  otherwise p = random permutation. See cs_randperm in CSparse
    static std::vector<int> randomPermutation(int n, int seed);

    /// Invert a permutation matrix: see cs_pinv in CSparse
    static std::vector<int> invertPermutation(const std::vector<int>& p);

    /// C = A(p,q) where p and q are permutations of 0..m-1 and 0..n-1.: see cs_permute in CSparse
    Sparsity permute(const std::vector<int>& pinv, const std::vector<int>& q, int values) const;

    /// consider A(i,j), node j in ith col subtree and return lca(jprev,j): See cs_leaf in CSparse
    static int leaf (int i, int j, const int *first, int *maxfirst, int *prevleaf, int *ancestor, int *jleaf);
    
    /// compute nnz(V) = S->lnz, S->pinv, S->leftmost, S->m2 from A and S->parent: See cs_vcount in CSparse
    int vcount(std::vector<int>& pinv, std::vector<int>& parent, std::vector<int>& leftmost, int& S_m2, double& S_lnz) const;
    
    /// post order a forest: See cs_post in CSparse
    static std::vector<int> postorder(const std::vector<int>& parent, int n);
    
    /// Depth-first search and postorder of a tree rooted at node j: See cs_tdfs in CSparse
    static int depthFirstSearchAndPostorder(int j, int k, int *head, const int *next, int *post, int *stack);

    /// row counts of LL'=A or LL'=A'A, given parent & post ordering: see init_ata in CSparse
    void init_ata(const int *post, int *w, int **head, int **next) const;
    
    /// Row counts: See cs_counts in CSparse
    std::vector<int> counts(const int *parent, const int *post, int ata) const;

    /// Approximate minimal degree, p = amd(A+A') if symmetric is true, or amd(A'A) otherwise. order 0:natural, 1:Chol, 2:LU, 3:QR. See cs_amd in CSparse
    std::vector<int> approximateMinimumDegree(int order) const;

    /// symbolic ordering and analysis for QR or LU: See cs_sqr in CSparse
    void prefactorize(int order, int qr, std::vector<int>& pinv, std::vector<int>& q, std::vector<int>& parent, std::vector<int>& cp, std::vector<int>& leftmost, int& m2, double& lnz, double& unz) const;
    
    /// clear w: cs_wclear in CSparse
    static int wclear(int mark, int lemax, int *w, int n);
    
    /// keep off-diagonal entries; drop diagonal entries: See cs_diag in CSparse
    static int diag(int i, int j, double aij, void *other);

    /// C = A*B: See cs_multiply in CSparse
    Sparsity multiply(const Sparsity& B) const;

    /// x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse: See cs_scatter in CSparse
    int scatter(int j, std::vector<int>& w, int mark, Sparsity& C, int nz) const;
    
    /// Get the column for each nonzero
    std::vector<int> getCol() const;

    /// Resize
    void resize(int nrow, int ncol);
    
    /// Reshape a sparsity, order of nonzeros remains the same
    Sparsity reshape(int nrow, int ncol) const;

    /// Number of structural non-zeros
    int size() const;

    /// Number of elements
    int numel() const;
    
    /// Check if the sparsity is empty, i.e. one of its dimensions is 0 
    bool empty() const;
    
    /// Check if the sparsity is null, i.e. dimension is 0-by-0
    bool null() const;

    /// Number of non-zeros in the lower triangular half
    int sizeL() const;

    /// Number of non-zeros in the upper triangular half
    int sizeU() const;

    /// Number of non-zeros on the diagonal
    int sizeD() const;
    
    /// Shape
    std::pair<int,int> shape() const;
    
    /// Is scalar?
    bool scalar(bool scalar_and_dense) const;
    
    /// Is dense?
    bool dense() const;
    
    /// Is diagonal?
    bool diagonal() const;

    /// Is square?
    bool square() const;

    /// Is symmetric?
    bool symmetric() const;

    /// Is lower triangular?
    bool tril() const;

    /// is upper triangular?
    bool triu() const;

    /// Get lower triangular part
    Sparsity lower(bool includeDiagonal) const;

    /// Get lower triangular part
    Sparsity upper(bool includeDiagonal) const;

    /// Get nonzeros in lower triangular part
    std::vector<int> lowerNZ() const;

    /// Get nonzeros in upper triangular part
    std::vector<int> upperNZ() const;

    /// Get the dimension as a string
    std::string dimString() const;

    //@{
    /// Sparsity pattern for a matrix-matrix product (details in public class)
    Sparsity patternProduct(const Sparsity& y, std::vector< std::vector< std::pair<int,int> > >& mapping) const;
    Sparsity patternProduct(const Sparsity& y) const;
    //@}
    
    //@{
    /// Union of two sparsity patterns
    Sparsity patternCombine(const Sparsity& y, bool f0x_is_zero, bool fx0_is_zero, std::vector<unsigned char>& mapping) const;
    Sparsity patternCombine(const Sparsity& y, bool f0x_is_zero, bool fx0_is_zero) const;

    template<bool with_mapping>
    Sparsity patternCombineGen1(const Sparsity& y, bool f0x_is_zero, bool fx0_is_zero, std::vector<unsigned char>& mapping) const;

    template<bool with_mapping, bool f0x_is_zero, bool fx0_is_zero>
    Sparsity patternCombineGen(const Sparsity& y, std::vector<unsigned char>& mapping) const;
    //@}
    
    /// Take the inverse of a sparsity pattern; flip zeros and non-zeros
    Sparsity patternInverse() const;

    /// Check if two sparsity patterns are the same
    bool isEqual(const Sparsity& y) const;

    /// Check if two sparsity patterns are the same
    bool isEqual(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row) const;

    /// Enlarge the matrix along the first dimension (i.e. insert rows)
    void enlargeRows(int nrow, const std::vector<int>& jj);

    /// Enlarge the matrix along the second dimension (i.e. insert columns)
    void enlargeColumns(int ncol, const std::vector<int>& ii);
    
    /// Make a patten dense
    Sparsity makeDense(std::vector<int>& mapping) const;

    /// Erase rows and/or columns - does bounds checking
    std::vector<int> erase(const std::vector<int>& jj, const std::vector<int>& ii);

    /// Append another sparsity patten vertically (vectors only)
    void append(const SparsityInternal& sp);

    /// Append another sparsity patten horizontally
    void appendColumns(const SparsityInternal& sp);

    /// Reserve space
    void reserve(int nnz, int ncol);
    
    /** \brief Get a submatrix
    * Does bounds checking
    * ii and jj are not required to be monotonous
    */
    Sparsity sub(const std::vector<int>& jj, const std::vector<int>& ii, std::vector<int>& mapping) const;

    /// Get the index of an existing non-zero element
    int getNZ(int rr, int cc) const;
    
    /// Get a set of non-zero element - does bounds checking
    std::vector<int> getNZ(const std::vector<int>& rr, const std::vector<int>& cc) const;

    /// Get the nonzero index for a set of elements (see descripion in public class)
    void getNZInplace(std::vector<int>& indices) const;
    
    /// Does the rows appear sequentially on each col
    bool rowsSequential(bool strictly) const;
    
    /// Remove duplicate entries: The same indices will be removed from the mapping vector, which must have the same length as the number of nonzeros
    void removeDuplicates(std::vector<int>& mapping);
  
    /// Get element index for each nonzero
    void getElements(std::vector<int>& loc, bool col_major) const;
    
    /// Hash the sparsity pattern
    std::size_t hash() const;

    /// Clone
    virtual SparsityInternal* clone() const{ return new SparsityInternal(*this); }

    /// Print representation
    virtual void repr(std::ostream &stream) const;

    /// Print description
    virtual void print(std::ostream &stream) const;

    /// Number of rows
    int nrow_;

    /// Number of cols
    int ncol_;

    /// vector of length n+1 containing the index of the last non-zero element up till each col 
    std::vector<int> colind_;

    /// vector of length nnz containing the rows for all the indices of the non-zero elements
    std::vector<int> row_;
        
    /// Perform a unidirectional coloring: A greedy distance-2 coloring algorithm (Algorithm 3.1 in A. H. GEBREMEDHIN, F. MANNE, A. POTHEN) 
    Sparsity unidirectionalColoring(const Sparsity& AT, int cutoff) const;

    /// Perform a star coloring of a symmetric matrix: A greedy distance-2 coloring algorithm (Algorithm 4.1 in A. H. GEBREMEDHIN, F. MANNE, A. POTHEN)
    Sparsity starColoring(int ordering, int cutoff) const;
    
    /// Perform a star coloring of a symmetric matrix: An improved distance-2 coloring algorithm (Algorithm 4.1 in A. H. GEBREMEDHIN, A. TARAFDAR, F. MANNE, A. POTHEN)
    Sparsity starColoring2(int ordering, int cutoff) const;

    /// Order the columns by decreasing degree
    std::vector<int> largestFirstOrdering() const;

    /// Permute rows and/or columns
    Sparsity pmult(const std::vector<int>& p, bool permute_rows=true, bool permute_cols=true, bool invert_permutation=false) const;

    /** \brief Print a textual representation of sparsity */
    void spy(std::ostream &stream) const;
    
    /// Generate a script for Matlab or Octave which visualizes the sparsity using the spy command
    void spyMatlab(const std::string& mfile) const;

 private: 
    /// Time complexity: O(ii.size()*jj.size())
    Sparsity sub1(const std::vector<int>& jj, const std::vector<int>& ii, std::vector<int>& mapping) const;
    /// Time complexity: O(ii.size()*(nnz per column))
    Sparsity sub2(const std::vector<int>& jj, const std::vector<int>& ii, std::vector<int>& mapping) const;
};

} // namespace CasADi

#endif // SPARSITY_INTERNAL_HPP
