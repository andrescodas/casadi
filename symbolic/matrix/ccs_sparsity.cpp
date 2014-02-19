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

#include "ccs_sparsity_internal.hpp"
#include "sparsity_tools.hpp"
#include "../matrix/matrix.hpp"
#include "../stl_vector_tools.hpp"
#include <climits>

using namespace std;

namespace CasADi{

  // Singletons
  class EmptySparsity : public CCSSparsity{  
  public:
    EmptySparsity(){
      std::vector<int> colind(1,0), row;
      assignNode(new CCSSparsityInternal(0,0,colind,row));
    }
  };

  class ScalarSparsity : public CCSSparsity{  
  public:
    ScalarSparsity(){
      std::vector<int> colind(2),row(1,0);
      colind[0] = 0;
      colind[1] = 1;
      assignNode(new CCSSparsityInternal(1,1,colind,row));
    }
  };

  class ScalarSparseSparsity : public CCSSparsity{  
  public:
    ScalarSparseSparsity(){
      std::vector<int> colind(2,0), row;
      assignNode(new CCSSparsityInternal(1,1,colind,row));
    }
  };
  
  CCSSparsity::CCSSparsity(int dummy){
    casadi_assert(dummy==0);
  }
  
  CCSSparsity CCSSparsity::create(CCSSparsityInternal *node){
    CCSSparsity ret;
    ret.assignNode(node);
    return ret;
  }

  CCSSparsity::CCSSparsity(int nrow, int ncol, bool dense){
    std::vector<int> row, colind(ncol+1,0);
    if(dense){
      row.resize(ncol*nrow);
      colind.resize(ncol+1);
      for(int i=0; i<ncol+1; ++i)
        colind[i] = i*nrow;
      for(int i=0; i<ncol; ++i)
        for(int j=0; j<nrow; ++j)
          row[j+i*nrow] = j;
    }
 
    assignCached(nrow, ncol, colind, row);
  }

  CCSSparsity::CCSSparsity(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row){
    assignCached(nrow, ncol, colind, row);
  }

  void CCSSparsity::reCache(){
    assignCached(size1(),size2(),colind(),row());
  }
 
  CCSSparsityInternal* CCSSparsity::operator->(){
    makeUnique();
    return static_cast<CCSSparsityInternal*>(SharedObject::operator->());
  }

  const CCSSparsityInternal* CCSSparsity::operator->() const{
    return static_cast<const CCSSparsityInternal*>(SharedObject::operator->());
  }
  
  bool CCSSparsity::checkNode() const{
    return dynamic_cast<const CCSSparsityInternal*>(get())!=0;
  }

  int CCSSparsity::size1() const{
    return (*this)->nrow_;
  }

  int CCSSparsity::size2() const{
    return (*this)->ncol_;
  }
    
  int CCSSparsity::numel() const{
    return (*this)->numel();
  }

  bool CCSSparsity::empty() const{
    return (*this)->empty();
  }
  
  bool CCSSparsity::null() const{
    return (*this)->null();
  }
    
  int CCSSparsity::size() const{
    return (*this)->size();
  }
    
  std::pair<int,int> CCSSparsity::shape() const{
    return (*this)->shape();
  }
    
  const std::vector<int>& CCSSparsity::row() const{
    return (*this)->row_;
  }
    
  const std::vector<int>& CCSSparsity::colind() const{
    return (*this)->colind_;
  }
    
  std::vector<int>& CCSSparsity::rowRef(){
    makeUnique();
    return (*this)->row_;
  }
    
  std::vector<int>& CCSSparsity::colindRef(){
    makeUnique();
    return (*this)->colind_;
  }
    
  int CCSSparsity::row(int el) const{
    return row().at(el);
  }
    
  int CCSSparsity::colind(int col) const{
    return colind().at(col);
  }

  void CCSSparsity::sanityCheck(bool complete) const { 
    (*this)->sanityCheck(complete);
  }
    
  void CCSSparsity::resize(int nrow, int ncol){
    makeUnique();
    (*this)->resize(nrow,ncol);
  }

  int CCSSparsity::getNZ(int j, int i){
    casadi_assert_message(i<size2() && j<size1(),"Indices out of bounds");

    if (i<0) i += size2();
    if (j<0) j += size1();
  
    // Quick return if matrix is dense
    if(numel()==size())
      return j+i*size1();
  
    // Quick return if we are adding an element to the end
    if(colind(i)==size() || (colind(i+1)==size() && row().back()<j)){
      std::vector<int>& rowv = rowRef();
      std::vector<int>& colindv = colindRef();
      rowv.push_back(j);
      for(int ii=i; ii<size2(); ++ii){
        colindv[ii+1]++;
      }
      return rowv.size()-1;
    }

    // go to the place where the element should be
    int ind;
    for(ind=colind(i); ind<colind(i+1); ++ind){ // better: loop from the back to the front
      if(row(ind) == j){
        return ind; // element exists
      } else if(row(ind) > j)
        break;                // break at the place where the element should be added
    }
  
    // Make sure that there no other objects are affected
    makeUnique();
  
    // insert the element
    rowRef().insert(rowRef().begin()+ind,j);
    for(int col=i+1; col<size2()+1; ++col)
      colindRef()[col]++;
  
    // Return the location of the new element
    return ind;
  }

  bool CCSSparsity::hasNZ(int j, int i) const {
    return (*this)->getNZ(j,i)!=-1;
  }


  int CCSSparsity::getNZ(int j, int i) const{
    return (*this)->getNZ(j,i);
  }

  CCSSparsity CCSSparsity::reshape(int nrow, int ncol) const{
    return (*this)->reshape(nrow,ncol);
  }

  std::vector<int> CCSSparsity::getNZ(const std::vector<int>& jj, const std::vector<int>& ii) const{
    return (*this)->getNZ(jj,ii);
  }

  bool CCSSparsity::scalar(bool scalar_and_dense) const{
    return (*this)->scalar(scalar_and_dense);
  }

  bool CCSSparsity::dense() const{
    return (*this)->dense();
  }

  bool CCSSparsity::diagonal() const{
    return (*this)->diagonal();
  }

  bool CCSSparsity::square() const{
    return (*this)->square();
  }

  bool CCSSparsity::symmetric() const{
    return (*this)->symmetric();
  }

  bool CCSSparsity::tril() const{
    return (*this)->tril();
  }

  bool CCSSparsity::triu() const{
    return (*this)->triu();
  }

  CCSSparsity CCSSparsity::sub(const std::vector<int>& jj, const std::vector<int>& ii, std::vector<int>& mapping) const{
    return (*this)->sub(jj,ii,mapping);
  }

  std::vector<int> CCSSparsity::erase(const std::vector<int>& jj, const std::vector<int>& ii){
    makeUnique();
    return (*this)->erase(jj,ii);
  }

  int CCSSparsity::sizeL() const{
    return (*this)->sizeL();
  }

  int CCSSparsity::sizeU() const{
    return (*this)->sizeU();
  }

  int CCSSparsity::sizeD() const{
    return (*this)->sizeD();
  }

  std::vector<int> CCSSparsity::getCol() const{
    return (*this)->getCol();
  }

  void CCSSparsity::getSparsityCCS(std::vector<int>& colind, std::vector<int>& row) const{
    colind = this->colind();
    row = this->row();
  }

  void CCSSparsity::getSparsityCRS(std::vector<int>& rowind, std::vector<int>& col) const {
    transpose().getSparsityCCS(rowind,col);
  }
    

  void CCSSparsity::getSparsity(std::vector<int>& row, std::vector<int>& col) const{
    row = this->row();
    col = this->getCol();
  }

  CCSSparsity CCSSparsity::transpose(std::vector<int>& mapping, bool invert_mapping) const{
    return (*this)->transpose(mapping,invert_mapping);
  }

  CCSSparsity CCSSparsity::transpose() const{
    return (*this)->transpose();
  }

  CCSSparsity CCSSparsity::patternCombine(const CCSSparsity& y, bool f0x_is_zero, bool fx0_is_zero, std::vector<unsigned char>& mapping) const{
    return (*this)->patternCombine(y, f0x_is_zero, fx0_is_zero, mapping);
  }

  CCSSparsity CCSSparsity::patternCombine(const CCSSparsity& y, bool f0x_is_zero, bool fx0_is_zero) const{
    return (*this)->patternCombine(y, f0x_is_zero, fx0_is_zero);
  }

  CCSSparsity CCSSparsity::patternUnion(const CCSSparsity& y, std::vector<unsigned char>& mapping) const{
    return (*this)->patternCombine(y, false, false, mapping);
  }

  CCSSparsity CCSSparsity::patternUnion(const CCSSparsity& y) const{
    return (*this)->patternCombine(y, false, false);
  }

  CCSSparsity CCSSparsity::patternIntersection(const CCSSparsity& y, std::vector<unsigned char>& mapping) const{
    return (*this)->patternCombine(y, true, true, mapping);
  }

  CCSSparsity CCSSparsity::patternIntersection(const CCSSparsity& y) const{
    return (*this)->patternCombine(y, true, true);
  }

  CCSSparsity CCSSparsity::patternProduct(const CCSSparsity& y) const{
    return (*this)->patternProduct(y);
  }

  CCSSparsity CCSSparsity::patternProduct(const CCSSparsity& y_trans, std::vector< std::vector< pair<int,int> > >& mapping) const{
    return (*this)->patternProduct(y_trans,mapping);
  }

  bool CCSSparsity::isEqual(const CCSSparsity& y) const{
    return (*this)->isEqual(y);
  }

  bool CCSSparsity::isEqual(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row) const{
    return (*this)->isEqual(nrow,ncol,colind,row);
  }

  CCSSparsity CCSSparsity::operator+(const CCSSparsity& b) const {
    return (DMatrix(*this,1)+DMatrix(b,1)).sparsity();
  }

  CCSSparsity CCSSparsity::operator*(const CCSSparsity& b) const {
    std::vector< unsigned char > mapping;
    return patternIntersection(b, mapping);
  }
  
  CCSSparsity CCSSparsity::patternInverse() const {
    return (*this)->patternInverse();
  }

  void CCSSparsity::reserve(int nnz, int ncol){
    makeUnique();
    (*this)->reserve(nnz,ncol);
  }

  void CCSSparsity::append(const CCSSparsity& sp){
    // TODO: Rewrite more efficiently
    *this = vertcat(*this,sp);
  }

  void CCSSparsity::appendColumns(const CCSSparsity& sp){
    casadi_assert(this!=&sp); // NOTE: this case needs to be handled
    makeUnique();
    (*this)->appendColumns(sp);
  }

  CCSSparsity::CachingMap& CCSSparsity::getCache(){
    static CachingMap ret;
    return ret;
  }

  const CCSSparsity& CCSSparsity::getScalar(){
    static ScalarSparsity ret;
    return ret;
  }

  const CCSSparsity& CCSSparsity::getScalarSparse(){
    static ScalarSparseSparsity ret;
    return ret;
  }

  const CCSSparsity& CCSSparsity::getEmpty(){
    static EmptySparsity ret;
    return ret;
  }

  void CCSSparsity::enlarge(int nrow, int ncol, const std::vector<int>& jj, const std::vector<int>& ii){
    enlargeColumns(ncol,ii);
    enlargeRows(nrow,jj);
  }

  void CCSSparsity::enlargeColumns(int ncol, const std::vector<int>& ii){
    makeUnique();
    (*this)->enlargeColumns(ncol,ii);
  }

  void CCSSparsity::enlargeRows(int nrow, const std::vector<int>& jj){
    makeUnique();
    (*this)->enlargeRows(nrow,jj);
  }

  CCSSparsity CCSSparsity::createDiagonal(int n){
    return createDiagonal(n,n);
  }

  CCSSparsity CCSSparsity::createDiagonal(int m, int n){
    CCSSparsity ret(m,n);
  
    // Set rows
    std::vector<int> &c = ret.rowRef();
    c.resize(min(n,m));
    for(int i=0; i<c.size(); ++i)
      c[i] = i;
  
    // Set col indices
    std::vector<int> &r = ret.colindRef();
    for(int i=0; i<n && i<m; ++i)
      r[i] = i;
  
    for(int i=min(n,m); i<n+1; ++i)
      r[i] = c.size();
  
    return ret;
  }

  CCSSparsity CCSSparsity::makeDense(std::vector<int>& mapping) const{
    return (*this)->makeDense(mapping);
  }

  std::string CCSSparsity::dimString() const { 
    return (*this)->dimString();
  }

  CCSSparsity CCSSparsity::diag(std::vector<int>& mapping) const{
    return (*this)->diag(mapping);
  }

  std::vector<int> CCSSparsity::eliminationTree(bool ata) const{
    return (*this)->eliminationTree(ata);
  }

  int CCSSparsity::depthFirstSearch(int j, int top, std::vector<int>& xi, std::vector<int>& pstack, const std::vector<int>& pinv, std::vector<bool>& marked) const{
    return (*this)->depthFirstSearch(j,top,xi,pstack,pinv,marked);
  }

  int CCSSparsity::stronglyConnectedComponents(std::vector<int>& p, std::vector<int>& r) const{
    return (*this)->stronglyConnectedComponents(p,r);
  }

  int CCSSparsity::dulmageMendelsohnQQQ(std::vector<int>& rowperm, std::vector<int>& colperm, std::vector<int>& rowblock, std::vector<int>& colblock, std::vector<int>& coarse_rowblock, std::vector<int>& coarse_colblock, int seed) const{
    return (*this)->dulmageMendelsohnQQQ(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock, seed);
  }

  bool CCSSparsity::rowsSequential(bool strictly) const{
    return (*this)->rowsSequential(strictly);
  }

  void CCSSparsity::removeDuplicates(std::vector<int>& mapping){
    makeUnique();
    (*this)->removeDuplicates(mapping);
  }

  std::vector<int> CCSSparsity::getElements(bool col_major) const{
    std::vector<int> loc;
    getElements(loc,col_major);
    return loc;
  }

  void CCSSparsity::getElements(std::vector<int>& loc, bool col_major) const{
    (*this)->getElements(loc,col_major);
  }

  void CCSSparsity::getNZInplace(std::vector<int>& indices) const{
    (*this)->getNZInplace(indices);
  }

  CCSSparsity CCSSparsity::unidirectionalColoring(const CCSSparsity& AT, int cutoff) const{
    if(AT.isNull()){
      return (*this)->unidirectionalColoring(transpose(),cutoff);
    } else {
      return (*this)->unidirectionalColoring(AT,cutoff);
    }
  }

  CCSSparsity CCSSparsity::starColoring(int ordering, int cutoff) const{
    return (*this)->starColoring(ordering,cutoff);
  }

  CCSSparsity CCSSparsity::starColoring2(int ordering, int cutoff) const{
    return (*this)->starColoring2(ordering,cutoff);
  }

  std::vector<int> CCSSparsity::largestFirstOrdering() const{
    return (*this)->largestFirstOrdering();
  }

  CCSSparsity CCSSparsity::pmult(const std::vector<int>& p, bool permute_rows, bool permute_cols, bool invert_permutation) const{
    return (*this)->pmult(p,permute_rows,permute_cols,invert_permutation);
  }

  void CCSSparsity::spyMatlab(const std::string& mfile) const{
    (*this)->spyMatlab(mfile);
  }

  void CCSSparsity::spy(std::ostream &stream) const{
    (*this)->spy(stream);
  }

  bool CCSSparsity::isTranspose(const CCSSparsity& y) const{
    return (*this)->isTranspose(*static_cast<const CCSSparsityInternal*>(y.get()));
  }

  std::size_t CCSSparsity::hash() const{
    return (*this)->hash();
  }

  void CCSSparsity::assignCached(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row){

    // Scalars and empty patterns are handled separately
    if(ncol==0 && nrow==0){
      // If empty    
      *this = getEmpty();
      return;
    } else if(ncol==1 && nrow==1){
      if(row.empty()){        
        // If sparse scalar
        *this = getScalarSparse();
        return;
      } else {
        // If dense scalar
        *this = getScalar();
        return;
      }
    }

    // Hash the pattern
    std::size_t h = hash_sparsity(nrow,ncol,colind,row);

    // Get a reference to the cache
    CachingMap& cache = getCache();
    
    // Record the current number of buckets (for garbage collection below)
#ifdef USE_CXX11
    int bucket_count_before = cache.bucket_count();
#endif // USE_CXX11

    // WORKAROUND, functions do not appear to work when bucket_count==0
#ifdef USE_CXX11
    if(bucket_count_before>0){
#endif // USE_CXX11

      // Find the range of patterns equal to the key (normally only zero or one)
      pair<CachingMap::iterator,CachingMap::iterator> eq = cache.equal_range(h);

      // Loop over maching patterns
      for(CachingMap::iterator i=eq.first; i!=eq.second; ++i){
      
        // Get a weak reference to the cached sparsity pattern
        WeakRef& wref = i->second;
      
        // Check if the pattern still exists
        if(wref.alive()){
        
          // Get an owning reference to the cached pattern
          CCSSparsity ref = shared_cast<CCSSparsity>(wref.shared());
        
          // Check if the pattern matches
          if(ref.isEqual(nrow,ncol,colind,row)){
          
            // Found match!
            assignNode(ref.get());
            return;

          } else {
            // There are two options, either the pattern has changed or there is a hash collision, so let's rehash the pattern
            std::size_t h_ref = ref.hash();
          
            if(h_ref!=h){ // The sparsity pattern has changed (the most likely event)

              // Create a new pattern
              assignNode(new CCSSparsityInternal(nrow, ncol, colind, row));

              // Cache this pattern instead of the old one
              wref = *this;

              // Recache the old sparsity pattern 
              // TODO: recache "ref"
              return;

            } else { // There is a hash rowision (unlikely, but possible)
              // Leave the pattern alone, continue to the next matching pattern
              continue; 
            }
          }
        } else {
          // Check if one of the other cache entries indeed has a matching sparsity
          CachingMap::iterator j=i;
          j++; // Start at the next matching key
          for(; j!=eq.second; ++j){
            if(j->second.alive()){
            
              // Recover cached sparsity
              CCSSparsity ref = shared_cast<CCSSparsity>(j->second.shared());
            
              // Match found if sparsity matches
              if(ref.isEqual(nrow,ncol,colind,row)){
                assignNode(ref.get());
                return;
              }
            }
          }

          // The cached entry has been deleted, create a new one
          assignNode(new CCSSparsityInternal(nrow, ncol, colind, row));
        
          // Cache this pattern
          wref = *this;

          // Return
          return;
        }
      }

      // END WORKAROUND
#ifdef USE_CXX11
    }
#endif // USE_CXX11

    // No matching sparsity pattern could be found, create a new one
    assignNode(new CCSSparsityInternal(nrow, ncol, colind, row));

    // Cache this pattern
    //cache.insert(eq.second,std::pair<std::size_t,WeakRef>(h,ret));
    cache.insert(std::pair<std::size_t,WeakRef>(h,*this));

    // Garbage collection (currently only supported for unordered_multimap)
#ifdef USE_CXX11
    int bucket_count_after = cache.bucket_count();
    
    // We we increased the number of buckets, take time to garbage-collect deleted references
    if(bucket_count_before!=bucket_count_after){
      CachingMap::const_iterator i=cache.begin();
      while(i!=cache.end()){
        if(!i->second.alive()){
          i = cache.erase(i);
        } else {
          i++;
        }
      }
    }
#endif // USE_CXX11    
  }

  void CCSSparsity::clearCache(){
    getCache().clear();
  }


} // namespace CasADi
