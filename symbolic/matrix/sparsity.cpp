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

#include "sparsity_internal.hpp"
#include "sparsity_tools.hpp"
#include "../matrix/matrix.hpp"
#include "../stl_vector_tools.hpp"
#include <climits>

using namespace std;

namespace CasADi{

  // Singletons
  class EmptySparsity : public Sparsity{  
  public:
    EmptySparsity(){
      std::vector<int> colind(1,0), row;
      assignNode(new SparsityInternal(0,0,colind,row));
    }
  };

  class ScalarSparsity : public Sparsity{  
  public:
    ScalarSparsity(){
      std::vector<int> colind(2),row(1,0);
      colind[0] = 0;
      colind[1] = 1;
      assignNode(new SparsityInternal(1,1,colind,row));
    }
  };

  class ScalarSparseSparsity : public Sparsity{  
  public:
    ScalarSparseSparsity(){
      std::vector<int> colind(2,0), row;
      assignNode(new SparsityInternal(1,1,colind,row));
    }
  };
  
  Sparsity::Sparsity(int dummy){
    casadi_assert(dummy==0);
  }
  
  Sparsity Sparsity::create(SparsityInternal *node){
    Sparsity ret;
    ret.assignNode(node);
    return ret;
  }

  Sparsity::Sparsity(int nrow, int ncol, bool dense){
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

  Sparsity::Sparsity(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row){
    assignCached(nrow, ncol, colind, row);
  }

  void Sparsity::reCache(){
    assignCached(size1(),size2(),colind(),row());
  }
 
  SparsityInternal* Sparsity::operator->(){
    makeUnique();
    return static_cast<SparsityInternal*>(SharedObject::operator->());
  }

  const SparsityInternal* Sparsity::operator->() const{
    return static_cast<const SparsityInternal*>(SharedObject::operator->());
  }

  SparsityInternal& Sparsity::operator*(){
    makeUnique();
    return *static_cast<SparsityInternal*>(get());
  }

  const SparsityInternal& Sparsity::operator*() const{
    return *static_cast<const SparsityInternal*>(get());
  }
  
  bool Sparsity::checkNode() const{
    return dynamic_cast<const SparsityInternal*>(get())!=0;
  }

  int Sparsity::size1() const{
    return (*this)->nrow_;
  }

  int Sparsity::size2() const{
    return (*this)->ncol_;
  }
    
  int Sparsity::numel() const{
    return (*this)->numel();
  }

  bool Sparsity::empty() const{
    return (*this)->empty();
  }
  
  bool Sparsity::null() const{
    return (*this)->null();
  }
    
  int Sparsity::size() const{
    return (*this)->size();
  }
    
  std::pair<int,int> Sparsity::shape() const{
    return (*this)->shape();
  }
    
  const std::vector<int>& Sparsity::row() const{
    return (*this)->row_;
  }
    
  const std::vector<int>& Sparsity::colind() const{
    return (*this)->colind_;
  }
    
  std::vector<int>& Sparsity::rowRef(){
    makeUnique();
    return (*this)->row_;
  }
    
  std::vector<int>& Sparsity::colindRef(){
    makeUnique();
    return (*this)->colind_;
  }
    
  int Sparsity::row(int el) const{
    return row().at(el);
  }
    
  int Sparsity::colind(int col) const{
    return colind().at(col);
  }

  void Sparsity::sanityCheck(bool complete) const { 
    (*this)->sanityCheck(complete);
  }
    
  void Sparsity::resize(int nrow, int ncol){
    makeUnique();
    (*this)->resize(nrow,ncol);
  }

  int Sparsity::getNZ(int rr, int cc){
    // If negative index, count from the back
    if(rr<0) rr += size1();
    if(cc<0) cc += size2();

    // Check consistency
    casadi_assert_message(rr>=0 && rr<size1(), "Row index out of bounds");
    casadi_assert_message(cc>=0 && cc<size2(), "Column index out of bounds");

    // Quick return if matrix is dense
    if(dense()) return rr+cc*size1();
  
    // Quick return if we are adding an element to the end
    if(colind(cc)==size() || (colind(cc+1)==size() && row().back()<rr)){
      std::vector<int>& rowv = rowRef();
      std::vector<int>& colindv = colindRef();
      rowv.push_back(rr);
      for(int c=cc; c<size2(); ++c){
        colindv[c+1]++;
      }
      return rowv.size()-1;
    }

    // go to the place where the element should be
    int ind;
    for(ind=colind(cc); ind<colind(cc+1); ++ind){ // better: loop from the back to the front
      if(row(ind) == rr){
        return ind; // element exists
      } else if(row(ind) > rr){
        break;                // break at the place where the element should be added
      }
    }
  
    // insert the element
    std::vector<int>& rowv = rowRef();
    std::vector<int>& colindv = colindRef();
    rowv.insert(rowv.begin()+ind,rr);
    for(int c=cc+1; c<size2()+1; ++c)
      colindv[c]++;
  
    // Return the location of the new element
    return ind;
  }

  bool Sparsity::hasNZ(int rr, int cc) const {
    return (*this)->getNZ(rr,cc)!=-1;
  }


  int Sparsity::getNZ(int rr, int cc) const{
    return (*this)->getNZ(rr,cc);
  }

  Sparsity Sparsity::reshape(int nrow, int ncol) const{
    return (*this)->reshape(nrow,ncol);
  }

  std::vector<int> Sparsity::getNZ(const std::vector<int>& rr, const std::vector<int>& cc) const{
    return (*this)->getNZ(rr,cc);
  }

  bool Sparsity::scalar(bool scalar_and_dense) const{
    return (*this)->scalar(scalar_and_dense);
  }

  bool Sparsity::dense() const{
    return (*this)->dense();
  }

  bool Sparsity::diagonal() const{
    return (*this)->diagonal();
  }

  bool Sparsity::square() const{
    return (*this)->square();
  }

  bool Sparsity::symmetric() const{
    return (*this)->symmetric();
  }

  bool Sparsity::tril() const{
    return (*this)->tril();
  }

  bool Sparsity::triu() const{
    return (*this)->triu();
  }

  Sparsity Sparsity::sub(const std::vector<int>& jj, const std::vector<int>& ii, std::vector<int>& mapping) const{
    return (*this)->sub(jj,ii,mapping);
  }

  std::vector<int> Sparsity::erase(const std::vector<int>& jj, const std::vector<int>& ii){
    makeUnique();
    return (*this)->erase(jj,ii);
  }

  int Sparsity::sizeL() const{
    return (*this)->sizeL();
  }

  int Sparsity::sizeU() const{
    return (*this)->sizeU();
  }

  int Sparsity::sizeD() const{
    return (*this)->sizeD();
  }

  std::vector<int> Sparsity::getCol() const{
    return (*this)->getCol();
  }

  void Sparsity::getSparsityCCS(std::vector<int>& colind, std::vector<int>& row) const{
    colind = this->colind();
    row = this->row();
  }

  void Sparsity::getSparsityCRS(std::vector<int>& rowind, std::vector<int>& col) const {
    transpose().getSparsityCCS(rowind,col);
  }
    

  void Sparsity::getSparsity(std::vector<int>& row, std::vector<int>& col) const{
    row = this->row();
    col = this->getCol();
  }

  Sparsity Sparsity::transpose(std::vector<int>& mapping, bool invert_mapping) const{
    return (*this)->transpose(mapping,invert_mapping);
  }

  Sparsity Sparsity::transpose() const{
    return (*this)->transpose();
  }

  Sparsity Sparsity::patternCombine(const Sparsity& y, bool f0x_is_zero, bool fx0_is_zero, std::vector<unsigned char>& mapping) const{
    return (*this)->patternCombine(y, f0x_is_zero, fx0_is_zero, mapping);
  }

  Sparsity Sparsity::patternCombine(const Sparsity& y, bool f0x_is_zero, bool fx0_is_zero) const{
    return (*this)->patternCombine(y, f0x_is_zero, fx0_is_zero);
  }

  Sparsity Sparsity::patternUnion(const Sparsity& y, std::vector<unsigned char>& mapping) const{
    return (*this)->patternCombine(y, false, false, mapping);
  }

  Sparsity Sparsity::patternUnion(const Sparsity& y) const{
    return (*this)->patternCombine(y, false, false);
  }

  Sparsity Sparsity::patternIntersection(const Sparsity& y, std::vector<unsigned char>& mapping) const{
    return (*this)->patternCombine(y, true, true, mapping);
  }

  Sparsity Sparsity::patternIntersection(const Sparsity& y) const{
    return (*this)->patternCombine(y, true, true);
  }

  Sparsity Sparsity::patternProduct(const Sparsity& y) const{
    return (*this)->patternProduct(y);
  }

  Sparsity Sparsity::patternProduct(const Sparsity& y_trans, std::vector< std::vector< pair<int,int> > >& mapping) const{
    return (*this)->patternProduct(y_trans,mapping);
  }

  bool Sparsity::isEqual(const Sparsity& y) const{
    return (*this)->isEqual(y);
  }

  bool Sparsity::isEqual(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row) const{
    return (*this)->isEqual(nrow,ncol,colind,row);
  }

  Sparsity Sparsity::operator+(const Sparsity& b) const {
    return (DMatrix(*this,1)+DMatrix(b,1)).sparsity();
  }

  Sparsity Sparsity::operator*(const Sparsity& b) const {
    std::vector< unsigned char > mapping;
    return patternIntersection(b, mapping);
  }
  
  Sparsity Sparsity::patternInverse() const {
    return (*this)->patternInverse();
  }

  void Sparsity::reserve(int nnz, int ncol){
    makeUnique();
    (*this)->reserve(nnz,ncol);
  }

  void Sparsity::append(const Sparsity& sp){
    if(sp.size1()==0 && sp.size2()==0){
      // Appending pattern is empty
      return;
    } else if(size1()==0 && size2()==0){
      // This is empty
      *this = sp;
    } else {
      casadi_assert_message(size2()==sp.size2(),"Sparsity::append: Dimension mismatch. You attempt to append a shape " << sp.dimString() << " to a shape " << dimString() << ". The number of columns must match.");
      if(sp.size1()==0){
        // No rows to add
        return;
      } else if(size1()==0){
        // No rows before
        *this = sp;
      } else if(vector()){
        // Append to vector (efficient)
        makeUnique();
        (*this)->append(*sp);
      } else {
        // Append to matrix (inefficient)
        *this = vertcat(*this,sp);
      }
    }
  }

  void Sparsity::appendColumns(const Sparsity& sp){
    if(sp.size1()==0 && sp.size2()==0){
      // Appending pattern is empty
      return;
    } else if(size1()==0 && size2()==0){
      // This is empty
      *this = sp;
    } else {
      casadi_assert_message(size1()==sp.size1(),"Sparsity::appendColumns: Dimension mismatch. You attempt to append a shape " << sp.dimString() << " to a shape " << dimString() << ". The number of rows must match.");
      if(sp.size2()==0){
        // No columns to add
        return;
      } else if(size2()==0){
        // No columns before
        *this = sp;
      } else {
        // Append to matrix (efficient)
        makeUnique();
        (*this)->appendColumns(*sp);
      }
    }
  }

  Sparsity::CachingMap& Sparsity::getCache(){
    static CachingMap ret;
    return ret;
  }

  const Sparsity& Sparsity::getScalar(){
    static ScalarSparsity ret;
    return ret;
  }

  const Sparsity& Sparsity::getScalarSparse(){
    static ScalarSparseSparsity ret;
    return ret;
  }

  const Sparsity& Sparsity::getEmpty(){
    static EmptySparsity ret;
    return ret;
  }

  void Sparsity::enlarge(int nrow, int ncol, const std::vector<int>& jj, const std::vector<int>& ii){
    enlargeColumns(ncol,ii);
    enlargeRows(nrow,jj);
  }

  void Sparsity::enlargeColumns(int ncol, const std::vector<int>& ii){
    makeUnique();
    (*this)->enlargeColumns(ncol,ii);
  }

  void Sparsity::enlargeRows(int nrow, const std::vector<int>& jj){
    makeUnique();
    (*this)->enlargeRows(nrow,jj);
  }

  Sparsity Sparsity::createDiagonal(int n){
    return createDiagonal(n,n);
  }

  Sparsity Sparsity::createDiagonal(int m, int n){
    Sparsity ret(m,n);
  
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

  Sparsity Sparsity::makeDense(std::vector<int>& mapping) const{
    return (*this)->makeDense(mapping);
  }

  std::string Sparsity::dimString() const { 
    return (*this)->dimString();
  }

  Sparsity Sparsity::diag(std::vector<int>& mapping) const{
    return (*this)->diag(mapping);
  }

  std::vector<int> Sparsity::eliminationTree(bool ata) const{
    return (*this)->eliminationTree(ata);
  }

  int Sparsity::depthFirstSearch(int j, int top, std::vector<int>& xi, std::vector<int>& pstack, const std::vector<int>& pinv, std::vector<bool>& marked) const{
    return (*this)->depthFirstSearch(j,top,xi,pstack,pinv,marked);
  }

  int Sparsity::stronglyConnectedComponents(std::vector<int>& p, std::vector<int>& r) const{
    return (*this)->stronglyConnectedComponents(p,r);
  }

  int Sparsity::dulmageMendelsohn(std::vector<int>& rowperm, std::vector<int>& colperm, std::vector<int>& rowblock, std::vector<int>& colblock, std::vector<int>& coarse_rowblock, std::vector<int>& coarse_colblock, int seed) const{
    return (*this)->dulmageMendelsohn(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock, seed);
  }

  bool Sparsity::rowsSequential(bool strictly) const{
    return (*this)->rowsSequential(strictly);
  }

  void Sparsity::removeDuplicates(std::vector<int>& mapping){
    makeUnique();
    (*this)->removeDuplicates(mapping);
  }

  std::vector<int> Sparsity::getElements(bool col_major) const{
    std::vector<int> loc;
    getElements(loc,col_major);
    return loc;
  }

  void Sparsity::getElements(std::vector<int>& loc, bool col_major) const{
    (*this)->getElements(loc,col_major);
  }

  void Sparsity::getNZInplace(std::vector<int>& indices) const{
    (*this)->getNZInplace(indices);
  }

  Sparsity Sparsity::unidirectionalColoring(const Sparsity& AT, int cutoff) const{
    if(AT.isNull()){
      return (*this)->unidirectionalColoring(transpose(),cutoff);
    } else {
      return (*this)->unidirectionalColoring(AT,cutoff);
    }
  }

  Sparsity Sparsity::starColoring(int ordering, int cutoff) const{
    return (*this)->starColoring(ordering,cutoff);
  }

  Sparsity Sparsity::starColoring2(int ordering, int cutoff) const{
    return (*this)->starColoring2(ordering,cutoff);
  }

  std::vector<int> Sparsity::largestFirstOrdering() const{
    return (*this)->largestFirstOrdering();
  }

  Sparsity Sparsity::pmult(const std::vector<int>& p, bool permute_rows, bool permute_cols, bool invert_permutation) const{
    return (*this)->pmult(p,permute_rows,permute_cols,invert_permutation);
  }

  void Sparsity::spyMatlab(const std::string& mfile) const{
    (*this)->spyMatlab(mfile);
  }

  void Sparsity::spy(std::ostream &stream) const{
    (*this)->spy(stream);
  }

  bool Sparsity::isTranspose(const Sparsity& y) const{
    return (*this)->isTranspose(*y);
  }

  std::size_t Sparsity::hash() const{
    return (*this)->hash();
  }

  void Sparsity::assignCached(int nrow, int ncol, const std::vector<int>& colind, const std::vector<int>& row){

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
          Sparsity ref = shared_cast<Sparsity>(wref.shared());
        
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
              assignNode(new SparsityInternal(nrow, ncol, colind, row));

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
              Sparsity ref = shared_cast<Sparsity>(j->second.shared());
            
              // Match found if sparsity matches
              if(ref.isEqual(nrow,ncol,colind,row)){
                assignNode(ref.get());
                return;
              }
            }
          }

          // The cached entry has been deleted, create a new one
          assignNode(new SparsityInternal(nrow, ncol, colind, row));
        
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
    assignNode(new SparsityInternal(nrow, ncol, colind, row));

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

  void Sparsity::clearCache(){
    getCache().clear();
  }

  Sparsity Sparsity::lower(bool includeDiagonal) const{
    return (*this)->lower(includeDiagonal);
  }

  Sparsity Sparsity::upper(bool includeDiagonal) const{
    return (*this)->upper(includeDiagonal);
  }

  std::vector<int> Sparsity::lowerNZ() const{
    return (*this)->lowerNZ();
  }

  std::vector<int> Sparsity::upperNZ() const{
    return (*this)->upperNZ();
  }


} // namespace CasADi
