#include "classify.h"
#include <omp.h>

int numberOfRanges ;

Data classify(Data &D, const Ranges &R, unsigned int numt)
{ // Classify each item in D into intervals (given by R). Finally, produce in D2 data sorted by interval
   assert(numt < MAXTHREADS);
   numberOfRanges = R.num() ;
   Counter counts[numt] ;

   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num(); // I am thread number tid
      int size = D.ndata / numt ; 
      int modulo = D.ndata % numt ; // these many initial threads will process one element extra
      int low_limit = tid*size + std::min(tid, modulo) ;
      int high_limit = low_limit + size ;
      if( tid < modulo ) high_limit ++ ;
      if(tid == numt - 1 ){
         high_limit = D.ndata ;
      }
      for(int i = low_limit ; i < high_limit ; i ++ ){
         // this loops through all ranges to find which range contains the data item 
         int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key, // returns the index in the range array
							  // and store the interval id in value. D is changed.
         counts[tid].increase(v) ;
      }
   }

   // Accumulate all sub-counts (in each interval;'s counter) into rangecount
   // each thread incremenets a subcounter and we need to accumulate all these subcount values to get the final subcount
   unsigned int *rangecount = new unsigned int[R.num()];
   for( int t = 0 ; t < numt ; t ++ ){
      for( int r = 0 ; r < R.num() ; r ++ ){
         if(t==0) rangecount[r] = 0 ;
         rangecount[r] += counts[t].get(r) ;
         // if(t == numt - 1) std::cout << rangecount[r] << " elements in Range " << r << "\n"; // Debugging statement
      }

   }
   // at the end of this loop rangecount has count of number of elements in the range
   // Compute prefx sum on rangecount.
   for(int i=1; i<R.num(); i++) {
      rangecount[i] += rangecount[i-1];
   }

   // Now rangecount[i] has the number of elements in intervals before the ith interval.

   Data D2 = Data(D.ndata); // Make a copy
   /// this loop reorders the data items according to the ranges, first elements in the first range came then elements of the next range and so on .... 
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      int size = R.num() / numt ; 
      int modulo = R.num() % numt ;
      int low_limit = tid * size + std::min(tid, modulo);
      int high_limit = (low_limit + size) ; 
      if(tid < modulo) high_limit ++ ;
      if(tid == numt - 1 ){
         high_limit = R.num() ;
      }
      // without algo optimization 
      // for(int r = low_limit ; r < high_limit ; r ++ ){
      //    int rcount = 0;
      //    int store = rangecount[r-1] ;
      //    for(int d=0; d<D.ndata; d++){ // For each interval, thread loops through all of data and  
      //        if( D.data[d].value != r) // If the data item is in this interval 
      //            continue ;
      //       else
      //            D2.data[store+rcount++]= D.data[d]; // Copy it to the appropriate place in D2.4
      //    }
      // }
      
      // with algo optimization 
      for( int d = 0 ; d < D.ndata ; d++){
         if(D.data[d].value < low_limit or D.data[d].value > high_limit){
            continue ;
         }else{
            D2.data[rangecount[D.data[d].value-1]++] = D.data[d] ;
         }
      }
   }

   return D2;
}
