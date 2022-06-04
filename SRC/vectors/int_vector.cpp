
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "vector.hpp"
#include "int_vector.hpp"
#include "vectorops.hpp"
#include "../utils/utils.hpp"

#include <cstring>
#ifdef SCHURCHEB_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace schurcheb
{
   
   template <typename T>
   IntVectorClass<T>::IntVectorClass()
   {
      this->_data = NULL;
      this->_length = 0;
      this->_maxlength = 0;
      this->_hold_data = true;
      this->_location = kMemoryHost;
   }
   template IntVectorClass<int>::IntVectorClass();
   template IntVectorClass<long int>::IntVectorClass();
   
   template <typename T>
   IntVectorClass<T>::IntVectorClass(const IntVectorClass<T> &vec) : VectorVirtualClass<T>(vec)
   {
      if(vec._hold_data)
      {
         SCHURCHEB_MALLOC( this->_data, vec._maxlength, vec._location, T);
         SCHURCHEB_MEMCPY( this->_data, vec._data, vec._length, vec._location, vec._location, T);
         this->_hold_data = true;
      }
      else
      {
         this->_data = vec._data;
         this->_hold_data = false;
      }
      
      this->_location = vec._location;
      this->_length = vec._length;
      this->_maxlength = vec._maxlength;
   }
   template IntVectorClass<int>::IntVectorClass(const IntVectorClass<int> &vec);
   template IntVectorClass<long int>::IntVectorClass(const IntVectorClass<long int> &vec);
   
   template <typename T>
   IntVectorClass<T>::IntVectorClass( IntVectorClass<T> &&vec) : VectorVirtualClass<T>(std::move(vec))
   {
      this->_data = vec._data;
      vec._data = NULL;
      this->_length = vec._length;
      vec._length = 0;
      this->_maxlength = vec._maxlength;
      vec._maxlength = 0;
      this->_hold_data = vec._hold_data;
      vec._hold_data = true;
      this->_location = vec._location;
      vec._location = kMemoryHost;
   }
   template IntVectorClass<int>::IntVectorClass( IntVectorClass<int> &&vec);
   template IntVectorClass<long int>::IntVectorClass( IntVectorClass<long int> &&vec);
   
   template <typename T>
   IntVectorClass<T>& IntVectorClass<T>::operator= (const IntVectorClass<T> &vec)
   {
      this->Clear();
      ParallelLogClass::operator=(vec);
      if(vec._hold_data)
      {
         SCHURCHEB_MALLOC( this->_data, vec._maxlength, vec._location, T);
         SCHURCHEB_MEMCPY( this->_data, vec._data, vec._length, vec._location, vec._location, T);
         this->_hold_data = true;
      }
      else
      {
         this->_data = vec._data;
         this->_hold_data = false;
      }
      
      this->_location = vec._location;
      this->_length = vec._length;
      this->_maxlength = vec._maxlength;
      return *this;
   }
   template IntVectorClass<int>&      IntVectorClass<int>::operator= (const IntVectorClass<int> &vec);
   template IntVectorClass<long int>&     IntVectorClass<long int>::operator= (const IntVectorClass<long int> &vec);

   template <typename T>
   IntVectorClass<T>& IntVectorClass<T>::operator= ( IntVectorClass<T> &&vec)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(vec));
      this->_data = vec._data;
      vec._data = NULL;
      this->_length = vec._length;
      vec._length = 0;
      this->_maxlength = vec._maxlength;
      vec._maxlength = 0;
      this->_hold_data = vec._hold_data;
      vec._hold_data = true;
      this->_location = vec._location;
      vec._location = kMemoryHost;
      return *this;
   }
   template IntVectorClass<int>&      IntVectorClass<int>::operator= ( IntVectorClass<int> &&vec);
   template IntVectorClass<long int>&     IntVectorClass<long int>::operator= ( IntVectorClass<long int> &&vec);

   template <typename T>
   T& IntVectorClass<T>::operator[] (int i)
   {
      SCHURCHEB_CHKERR(i < 0);
      SCHURCHEB_CHKERR(i >= this->_length);
      return (this->_data[i]);
   }
   template int&      IntVectorClass<int>::operator[] (int i);
   template long int&     IntVectorClass<long int>::operator[] (int i);

   template <typename T>
   T& IntVectorClass<T>::Back()
   {
      return (this->_data[this->_length-1]);
   }
   template int&      IntVectorClass<int>::Back();
   template long int&     IntVectorClass<long int>::Back();

   template <typename T>
   int IntVectorClass<T>::Setup(int length)
   {
      return Setup( length, length, this->_location, false);
   }
   template int IntVectorClass<int>::Setup(int length);
   template int IntVectorClass<long int>::Setup(int length);

   template <typename T>
   int IntVectorClass<T>::Setup(int length, bool setzero)
   {
      return Setup( length, length, this->_location, setzero);
   }
   template int IntVectorClass<int>::Setup(int length, bool setzero);
   template int IntVectorClass<long int>::Setup(int length, bool setzero);

   template <typename T>
   int IntVectorClass<T>::Setup(int length, int location, bool setzero)
   {
      return Setup( length, length, location, setzero);
   }
   template int IntVectorClass<int>::Setup(int length, int location, bool setzero);
   template int IntVectorClass<long int>::Setup(int length, int location, bool setzero);

   template <typename T>
   int IntVectorClass<T>::Setup(int length, int reserve, int location, bool setzero)
   {
      SCHURCHEB_CHKERR(length < 0);
      SCHURCHEB_CHKERR(length > reserve);

      /* first check if we can use current memory */
      if(location == this->_location && this->_hold_data && this->_maxlength >= reserve)
      {
         /* we have enough space, don't need to reset */
         if(setzero)
         {
            /* Fill the vector with 0 when necessary */
            this->_length = this->_maxlength;
            Fill(0);
         }
         this->_length = length;
      }
      else
      {
         /* in this case, we need new memory location */
         Clear();
         if(reserve == 0)
         {
            this->_data = NULL;
         }
         else
         {
            if(setzero)
            {
               SCHURCHEB_CALLOC( this->_data, reserve, location, T);
            }
            else
            {
               SCHURCHEB_MALLOC( this->_data, reserve, location, T);
            }
        }
        this->_hold_data = true;
        this->_location = location;
        this->_length = length;
        this->_maxlength = reserve;
      }
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Setup(int length, int reserve, int location, bool setzero);
   template int IntVectorClass<long int>::Setup(int length, int reserve, int location, bool setzero);

   template <typename T>
   int IntVectorClass<T>::SetupPtr( T* data, int length, int location)
   {
      return this->SetupPtr( data, length, location, false);
   }
   template int IntVectorClass<int>::SetupPtr( int *data, int length, int location);
   template int IntVectorClass<long int>::SetupPtr( long int *data, int length, int location);
   
   template <typename T>
   int IntVectorClass<T>::SetupPtr( T* data, int length, int location, bool hold_data)
   {
      
      SCHURCHEB_CHKERR(length < 0 );
      
      /* first of all free current vector */
      Clear();
      this->_data = data;
      this->_hold_data = hold_data;
      this->_location = location;
      this->_length = length;
      this->_maxlength = length;// this value is not referenced
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::SetupPtr( int *data, int length, int location, bool hold_data);
   template int IntVectorClass<long int>::SetupPtr( long int *data, int length, int location, bool hold_data);
   
   template <typename T>
   int IntVectorClass<T>::SetupPtr(const IntVectorClass<T> &vec, int length, int shift)
   {
      SCHURCHEB_CHKERR( shift < 0);
      SCHURCHEB_CHKERR( vec.GetLengthLocal() < length + shift );
      
      return SetupPtr( vec.GetData() + shift, length, vec.GetDataLocation(), false);
   }
   template int IntVectorClass<int>::SetupPtr(const vector_int &vec, int length, int shift);
   template int IntVectorClass<long int>::SetupPtr(const vector_long &vec, int length, int shift);

   template <typename T>
   int IntVectorClass<T>::SetupPtr(const IntVectorClass<T> &vec, int length, int shift, bool hold_data)
   {
      SCHURCHEB_CHKERR( shift < 0);
      SCHURCHEB_CHKERR( vec.GetLengthLocal() < length + shift );
      
      return SetupPtr( vec.GetData() + shift, length, vec.GetDataLocation(), hold_data);
   }
   template int IntVectorClass<int>::SetupPtr(const vector_int &vec, int length, int shift, bool hold_data);
   template int IntVectorClass<long int>::SetupPtr(const vector_long &vec, int length, int shift, bool hold_data);
   
   template <typename T>
   int IntVectorClass<T>::Copy(const T *data, int length, int loc_from, int loc_to)
   {

      SCHURCHEB_CHKERR(length < 0 );

      /* first of all free current vector */
      Clear();
      SCHURCHEB_MALLOC( this->_data, length, loc_to, T);
      SCHURCHEB_MEMCPY( this->_data, data, length, loc_to, loc_from, T);
      this->_hold_data = true;
      this->_location = loc_to;
      this->_length = length;
      this->_maxlength = length;// this value is not referenced
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Copy(const int *data, int length, int loc_from, int loc_to);
   template int IntVectorClass<long int>::Copy(const long int *data, int length, int loc_from, int loc_to);
   
   template <typename T>
   int IntVectorClass<T>::Clear()
   {
      /* base class clear */
      VectorVirtualClass<T>::Clear();
      
      if( this->_hold_data && this->_maxlength > 0)
      {
         SCHURCHEB_FREE( this->_data, this->_location);
      }
      else
      {
         this->_data = NULL;
      }
      this->_length = 0;
      this->_maxlength = 0;
      this->_location = kMemoryHost;
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Clear();
   template int IntVectorClass<long int>::Clear();
   
   template <typename T>
   int IntVectorClass<T>::PushBack(T v)
   {
      
      SCHURCHEB_CHKERR(!_hold_data);

      while( this->_length >= this->_maxlength)
      {
         Resize( _length, SchurchebMax( this->_length+1, (int)( this->_length * schurcheb::SCHURCHEB_global::_expand_fact)), true, false);
      }
      this->_data[ (this->_length)++] = v;
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::PushBack(int v);
   template int IntVectorClass<long int>::PushBack(long int v);

   template <typename T>
   int IntVectorClass<T>::Resize(int length, bool keepdata, bool setzero)
   {
      return Resize(length, length, keepdata, setzero);
   }
   template int IntVectorClass<int>::Resize(int length, bool keepdata, bool setzero);
   template int IntVectorClass<long int>::Resize(int length, bool keepdata, bool setzero);

   template <typename T>
   int IntVectorClass<T>::Resize(int length, int reserve, bool keepdata, bool setzero)
   {

      SCHURCHEB_CHKERR(length < 0)
      SCHURCHEB_CHKERR(length > reserve);

      IntVectorClass<T>    temp;
      T*                   temp_data;
      if(_hold_data)
      {
         /* if we don't have enough memory, need to allocate new memory */
         if(reserve > _maxlength)
         {
            /* we are epxanding the memory */
            if(setzero)
            {
               SCHURCHEB_CALLOC(temp_data, reserve, this->_location, T);
            }
            else
            {
               SCHURCHEB_MALLOC(temp_data, reserve, this->_location, T);
            }
            if(keepdata)
            {
               SCHURCHEB_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
            }
            SCHURCHEB_FREE( this->_data, this->_location);
            this->_data = temp_data;
            this->_length = length;
            this->_maxlength = reserve;
         }
         else
         {
            /* we have enouth memory */
            if(setzero)
            {
               if(length > _length)
               {
                  /* expanding */
                  temp.SetupPtr( *this, this->_maxlength - this->_length, this->_length);
                  temp.Fill(0);
               }
               else
               {
                  temp.SetupPtr( *this, this->_maxlength - length, length);
                  temp.Fill(0);
               }
            }
            _length = length;
         }
      }
      else
      {
         /* in this case just change the length */
         _length = length;
      }
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Resize(int length, int reserve, bool keepdata, bool setzero);
   template int IntVectorClass<long int>::Resize(int length, int reserve, bool keepdata, bool setzero);

   template <typename T>
   IntVectorClass<T>::~IntVectorClass()
   {
      Clear();
   }
   template IntVectorClass<int>::~IntVectorClass();
   template IntVectorClass<long int>::~IntVectorClass();
   
   template <typename T>
   T* IntVectorClass<T>::GetData() const
   {
      return this->_data;
   }
   template int* IntVectorClass<int>::GetData() const;
   template long int* IntVectorClass<long int>::GetData() const;
   
   template <typename T>
   int IntVectorClass<T>::GetDataLocation() const
   {
#ifndef SCHURCHEB_CUDA
      return kMemoryHost;
#else
      return this->_location;
#endif
   }
   template int IntVectorClass<int>::GetDataLocation() const;
   template int IntVectorClass<long int>::GetDataLocation() const;
   
   template <typename T>
   int IntVectorClass<T>::GetLengthLocal() const
   {
      return this->_length;
   }
   template int IntVectorClass<int>::GetLengthLocal() const;
   template int IntVectorClass<long int>::GetLengthLocal() const;
   
   template <typename T>
   int IntVectorClass<T>::BinarySearch(const T &val, int &idx, bool ascending)
   {
      /* TODO: No OpenMP implemntation yet, we don't have that large array to search from */
      if(this->_location == kMemoryDevice)
      {
         std::cout<<"Current version only support BinarySearch on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      
      return VectorPBsearchHost( this->_data, val, 0, this->_length-1, idx, ascending, 2);
   }
   template int IntVectorClass<int>::BinarySearch(const int &val, int &idx, bool ascending);
   template int IntVectorClass<long int>::BinarySearch(const long int &val, int &idx, bool ascending);
   
#ifndef SCHURCHEB_CUDA 
   template <typename T>
   int IntVectorClass<T>::Fill(const T &v)
   {
      int i;
#ifdef SCHURCHEB_OPENMP
      /* avoid nested OpenMP call */
      int num_threads = SchurchebGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = v;
         }
      }
      else
      {
#endif
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = v;
         }
#ifdef SCHURCHEB_OPENMP
      }
#endif

      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Fill(const int &v);
   template int IntVectorClass<long int>::Fill(const long int &v);

   template <typename T>
   int IntVectorClass<T>::UnitPerm()
   {
      int i;
#ifdef SCHURCHEB_OPENMP
      /* avoid nested OpenMP call */
      int num_threads = SchurchebGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = i;
         }
      }
      else
      {
#endif
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = i;
         }
#ifdef SCHURCHEB_OPENMP
      }
#endif

      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::UnitPerm();
   
   template <typename T>
   T IntVectorClass<T>::Max() const
   {
      T* val;
      val = std::max_element( this->_data, this->_data+this->_length);
      return *val;
   }
   template int IntVectorClass<int>::Max() const;
   template long int IntVectorClass<long int>::Max() const;
   
   template <typename T>
   int IntVectorClass<T>::MaxIndex() const
   {
      return std::distance( this->_data, std::max_element( this->_data, this->_data+this->_length));
   }
   template int IntVectorClass<int>::MaxIndex() const;
   template int IntVectorClass<long int>::MaxIndex() const;
   
   template <typename T>
   T IntVectorClass<T>::Min() const
   {
      T* val;
      val = std::min_element( this->_data, this->_data+this->_length);
      return *val;
   }
   template int IntVectorClass<int>::Min() const;
   template long int IntVectorClass<long int>::Min() const;
   
   template <typename T>
   int IntVectorClass<T>::MinIndex() const
   {
      return std::distance( this->_data, std::min_element( this->_data, this->_data+this->_length));
   }
   template int IntVectorClass<int>::MinIndex() const;
   template int IntVectorClass<long int>::MinIndex() const;
   
   template <typename T>
   int IntVectorClass<T>::Sort(bool ascending)
   {
      if(ascending)
      {
         std::sort( this->_data, this->_data + this->_length, std::less<T>());
      }
      else
      {
         std::sort( this->_data, this->_data + this->_length, std::greater<T>());
      }

      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Sort(bool ascending);
   template int IntVectorClass<long int>::Sort(bool ascending);
   
   template <typename T>
   int IntVectorClass<T>::Sort(IntVectorClass<int> &order, bool ascending, bool stable)
   {
      std::vector<CompareStruct<T> > compare;
      
      int i;
      int n = order.GetLengthLocal();
      
      if(n != this->_length)
      {
         order.Setup( this->_length);
      }
      
      compare.resize( this->_length);
      
      for( i = 0 ; i < this->_length ; i ++)
      {
         compare[i].val = this->_data[i];
         compare[i].ord = i;
      }
      
      if(stable)
      {
         if(ascending)
         {
            std::stable_sort( compare.begin(), compare.end(), CompareStructLess<CompareStruct<T> >());
         }
         else
         {
            std::stable_sort( compare.begin(), compare.end(), CompareStructGreater<CompareStruct<T> >());
         }
      }
      else
      {
         if(ascending)
         {
            std::sort( compare.begin(), compare.end(), CompareStructLess<CompareStruct<T> >());
         }
         else
         {
            std::sort( compare.begin(), compare.end(), CompareStructGreater<CompareStruct<T> >());
         }
      }
      
      for( i = 0 ; i < this->_length ; i ++)
      {
         order[i] = compare[i].ord;
      }
      
      std::vector<CompareStruct<T> >().swap(compare);
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   template int IntVectorClass<long int>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   
   template <typename T>
   int IntVectorClass<T>::Perm(IntVectorClass<int> &perm)
   {
      int   i;
      T     *temp_data;
      
      SCHURCHEB_CHKERR( perm.GetLengthLocal() != this->_length);
      
      SCHURCHEB_MALLOC( temp_data, this->_length, this->_location, T);
      SCHURCHEB_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
      
      for(i = 0 ; i < this->_length ; i ++)
      {
         this->_data[i] = temp_data[perm[i]];
      }
      
      SCHURCHEB_FREE( temp_data, this->_location);
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::Perm(IntVectorClass<int> &perm);
   template int IntVectorClass<long int>::Perm(IntVectorClass<int> &perm);
   
   template <typename T>
   template <typename T1>
   int IntVectorClass<T>::ScatterRperm(const VectorVirtualClass<T1> &v_in, VectorVirtualClass<T1> &v_out)
   {
      SCHURCHEB_CHKERR(v_in.GetLengthLocal() < this->_length);
      
      int i;
      T1* v_in_data = v_in.GetData();
      T1* v_out_data = v_out.GetData();

#ifdef SCHURCHEB_OPENMP
      if(SchurchebGetOpenmpMaxNumThreads() > 1)
      {
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
         for(i = 0 ; i < this->_length ; i ++)
         {
           v_out_data[this->_data[i]] = v_in_data[i];
         }
      }
      else
      {
#endif
         for(i = 0 ; i < this->_length ; i ++)
         {
           v_out_data[this->_data[i]] = v_in_data[i];
         }
#ifdef SCHURCHEB_OPENMP
      }
#endif
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<int> &v_in, VectorVirtualClass<int> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<long int> &v_in, VectorVirtualClass<long int> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<float> &v_in, VectorVirtualClass<float> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<double> &v_in, VectorVirtualClass<double> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<complexs> &v_in, VectorVirtualClass<complexs> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<complexd> &v_in, VectorVirtualClass<complexd> &v_out);
   
   template <typename T>
   template <typename T1>
   int IntVectorClass<T>::GatherPerm(const VectorVirtualClass<T1> &v_in, VectorVirtualClass<T1> &v_out)
   {
      SCHURCHEB_CHKERR(v_out.GetLengthLocal() < this->_length);
      
      int i;
      T1* v_in_data = v_in.GetData();
      T1* v_out_data = v_out.GetData();
   
#ifdef SCHURCHEB_OPENMP
      if(SchurchebGetOpenmpMaxNumThreads() > 1)
      {
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
         for(i = 0 ; i < this->_length ; i ++)
         {
           v_out_data[i] = v_in_data[this->_data[i]];
         }
      }
      else
      {
#endif
         for(i = 0 ; i < this->_length ; i ++)
         {
           v_out_data[i] = v_in_data[this->_data[i]];
         }
#ifdef SCHURCHEB_OPENMP
      }
#endif
      
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<int> &v_in, VectorVirtualClass<int> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<long int> &v_in, VectorVirtualClass<long int> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<float> &v_in, VectorVirtualClass<float> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<double> &v_in, VectorVirtualClass<double> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<complexs> &v_in, VectorVirtualClass<complexs> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<complexd> &v_in, VectorVirtualClass<complexd> &v_out);
   
   template <typename T>
   int IntVectorClass<T>::MoveData( const int &location)
   {
      //SCHURCHEB_CHKERR( location != kMemoryHost);
      return SCHURCHEB_SUCCESS;
   }
   template int IntVectorClass<int>::MoveData( const int &location);
   template int IntVectorClass<long int>::MoveData( const int &location);
#endif
   
   template <typename T>
   int IntVectorClass<T>::Plot( int conditiona, int conditionb, int width)
   {
      if(this->_location == kMemoryDevice)
      {
         std::cout<<"Plot vector only can be used on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      return VectorPlotHost( *this, conditiona, conditionb, width);
   }
   template int IntVectorClass<int>::Plot( int conditiona, int conditionb, int width);
   template int IntVectorClass<long int>::Plot( int conditiona, int conditionb, int width);
   
}
