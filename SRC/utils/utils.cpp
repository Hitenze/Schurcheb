
#include "utils.hpp"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <limits.h>

namespace schurcheb
{
   
   double SCHURCHEB_global::_expand_fact = 1.3;
   int SCHURCHEB_global::_coo_reserve_fact = 7;
   int SCHURCHEB_global::_dash_line_width = 32;
   int SCHURCHEB_global::_openmp_min_loopsize = 12;
   int SCHURCHEB_global::_metis_refine = 0;
   double SCHURCHEB_global::_metis_loading_balance_tol = 0.2;
   int SCHURCHEB_global::_minsep = 10;
   double SCHURCHEB_global::_tr_factor = 0.25;
   double SCHURCHEB_global::_orth_tol = 1e-14;
   double SCHURCHEB_global::_reorth_tol = 1.0/sqrt(2.0);
   std::random_device SCHURCHEB_global::_random_device;
   std::mt19937 SCHURCHEB_global::_mersenne_twister_engine(SCHURCHEB_global::_random_device());
   std::uniform_int_distribution<int> SCHURCHEB_global::_uniform_int_distribution(0, INT_MAX);
   FILE* SCHURCHEB_global::_out_file = stdout;
   
   int SCHURCHEB_global::_gram_schmidt = 0;
   
   template <typename T>
   bool operator<(const CompareStruct<T> &a, const CompareStruct<T> &b)
   {
      return a.val < b.val;
   }
   template bool operator<(const CompareStruct<int> &a, const CompareStruct<int> &b);
   template bool operator<(const CompareStruct<long int> &a, const CompareStruct<long int> &b);
   
   template <typename T>
   T SchurchebMax(T a, T b)
   {
      return a >= b ? a : b;
   }
   template int SchurchebMax(int a, int b);
   template long int SchurchebMax(long int a, long int b);
   template float SchurchebMax(float a, float b);
   template double SchurchebMax(double a, double b);
   
   template <typename T>
   T SchurchebMin(T a, T b)
   {
      return a <= b ? a : b;
   }
   template int SchurchebMin(int a, int b);
   template long int SchurchebMin(long int a, long int b);
   template float SchurchebMin(float a, float b);
   template double SchurchebMin(double a, double b);
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, T>::type
   SchurchebAbs( const T &a)
   {
      return std::abs(a);
   }
   template int SchurchebAbs( const int &a);
   template long int SchurchebAbs( const long int &a);
   template float SchurchebAbs( const float &a);
   template double SchurchebAbs( const double &a);
   
   float SchurchebAbs( const complexs &a)
   {
      return a.Abs();
   }
   
   double SchurchebAbs( const complexd &a)
   {
      return a.Abs();
   }
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, T>::type
   SchurchebReal( const T &a)
   {
      return a;
   }
   template int SchurchebReal( const int &a);
   template long int SchurchebReal( const long int &a);
   template float SchurchebReal( const float &a);
   template double SchurchebReal( const double &a);
   
   float SchurchebReal( const complexs &a)
   {
      return a.Real();
   }
   
   double SchurchebReal( const complexd &a)
   {
      return a.Real();
   }
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, T>::type
   SchurchebConj( const T &a)
   {
      return a;
   }
   template int SchurchebConj( const int &a);
   template long int SchurchebConj( const long int &a);
   template float SchurchebConj( const float &a);
   template double SchurchebConj( const double &a);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, T>::type
   SchurchebConj( const T &a)
   {
      return a.Conj();
   }
   template complexs SchurchebConj( const complexs &a);
   template complexd SchurchebConj( const complexd &a);
   
   template <typename T>
   typename std::enable_if<SchurchebIsInteger<T>::value, int>::type
   SchurchebValueRandHost(T &a)
   {
      a = (T)(SCHURCHEB_global::_uniform_int_distribution(SCHURCHEB_global::_mersenne_twister_engine));
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebValueRandHost(int &a);
   template int SchurchebValueRandHost(long int &a);
   
   template <typename T>
   typename std::enable_if<SchurchebIsReal<T>::value, int>::type
   SchurchebValueRandHost(T &a)
   {
      a = (SCHURCHEB_global::_uniform_int_distribution(SCHURCHEB_global::_mersenne_twister_engine))/((T)SCHURCHEB_global::_uniform_int_distribution.max());
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebValueRandHost(float &a);
   template int SchurchebValueRandHost(double &a);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebValueRandHost(T &a)
   {
      a = T((SCHURCHEB_global::_uniform_int_distribution(SCHURCHEB_global::_mersenne_twister_engine))/((double)SCHURCHEB_global::_uniform_int_distribution.max()),(SCHURCHEB_global::_uniform_int_distribution(SCHURCHEB_global::_mersenne_twister_engine))/((double)SCHURCHEB_global::_uniform_int_distribution.max()));
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebValueRandHost(complexs &a);
   template int SchurchebValueRandHost(complexd &a);
   
   int SchurchebPrintSpace(int width)
   {
      int i;
      for(i = 0 ; i < width ; i ++)
      {
         SCHURCHEB_PRINT(" ");
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebPrintDashLine(int width)
   {
      int i;
      for(i = 0 ; i < width ; i ++)
      {
         SCHURCHEB_PRINT("-");
      }
      SCHURCHEB_PRINT("\n");
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   typename std::enable_if<SchurchebIsInteger<T>::value, int>::type 
   SchurchebPrintValueHost(T val, int width)
   {
      long int vall = (long int) val;
      if(vall < 0)
      {
         SCHURCHEB_PRINT("-%*ld ",width,-vall);
      }
      else
      {
         SCHURCHEB_PRINT("+%*ld ",width,vall);
      }
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebPrintValueHost(int val, int width);
   template int SchurchebPrintValueHost(long int val, int width);
   
   template <typename T>
   typename std::enable_if<SchurchebIsReal<T>::value, int>::type 
   SchurchebPrintValueHost(T val, int width)
   {
      if(val < 0)
      {
         SCHURCHEB_PRINT("-%*.*f",width,width-2,SchurchebAbs(val));
      }
      else
      {
         SCHURCHEB_PRINT("+%*.*f",width,width-2,SchurchebAbs(val));
      }
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebPrintValueHost(float val, int width);
   template int SchurchebPrintValueHost(double val, int width);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type 
   SchurchebPrintValueHost(T val, int width)
   {
      double val_r, val_i;
      val_r = val.Real();
      val_i = val.Imag();
      
      if(val_r < 0)
      {
         SCHURCHEB_PRINT("-%*.*f",width,width-2,SchurchebAbs(val_r));
      }
      else
      {
         SCHURCHEB_PRINT("+%*.*f",width,width-2,SchurchebAbs(val_r));
      }
      if(val_i < 0)
      {
         SCHURCHEB_PRINT("-%*.*f",width,width-2,SchurchebAbs(val_i));
      }
      else
      {
         SCHURCHEB_PRINT("+%*.*f",width,width-2,SchurchebAbs(val_i));
      }
      SCHURCHEB_PRINT("i");
      
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebPrintValueHost(ComplexValueClass<float> val, int width);
   template int SchurchebPrintValueHost(ComplexValueClass<double> val, int width);
   
   void SchurchebReadFirstWord(char *pin, char **pout)
   {
      char *p_read, *p_write;
      p_read = pin;
      
      /* locate the start of the first word */                                
      while (' ' == *p_read) 
      {                        
         p_read++;                                     
      }                                    
      /* allocate the return buffer */
      *pout = (char*)malloc(sizeof(char)*(strlen(p_read)+1));
      p_write = *pout;                                   
      while (' ' != *p_read) 
      {
         /* convert to upper */           
         *p_write = toupper(*p_read);                       
         p_write++;                                     
         p_read++;                                    
      }
      /* only keep the first word */                     
      *p_write = '\0';
   }
   
   int SchurchebReadInputArg(const char *argname, int amount, float *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = atof(argv[i+j+1]);
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, int amount, double *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = atof(argv[i+j+1]);
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, int amount, complexs *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = complexs(atof(argv[i+2*j+1]),atof(argv[i+2*j+2]));
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, int amount, complexd *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = complexd(atof(argv[i+2*j+1]),atof(argv[i+2*j+2]));
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, int amount, int *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = atoi(argv[i+j+1]);
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, char *val, int argc, char **argv)
   {
      int i;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            sprintf(val, "%s", argv[i+1]);
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, bool *val, int argc, char **argv)
   {
      int i, temp_bool;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            temp_bool = atoi(argv[i+1]);
            val[0] = temp_bool ? true : false;
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int SchurchebReadInputArg(const char *argname, int argc, char **argv)
   {
      int i;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            return 1;
         }
      }
      return 0;
   }
   
   template <typename T>
   typename std::enable_if<!(SchurchebIsComplex<T>::value), int>::type
   SchurchebPlotData(T* ydata, int length, int numx, int numy)
   {
      SCHURCHEB_CHKERR(length <= 0 || numx <= 0 || numy <= 0);
      /* is the total plot number greater than length? */
      numx = numx > length ? length : numx;
      int unitx = length/numx;
      
      /* now, collect data every unitx points */
      double maxval = 0.0, minval = 0.0;
      std::vector<double> vals;
      vals.resize(numx);
      
      for(int i = 0 ; i < numx-1 ; i ++)
      {
         int i1 = i*unitx;
         int i2 = i1+unitx;
         vals[i] = 0;
         for(int j = i1 ; j < i2 ; j ++)
         {
            vals[i] += (double)ydata[j]/(double)unitx;
         }
         maxval = maxval > vals[i] ? maxval : vals[i];
         minval = minval < vals[i] ? minval : vals[i];
      }
      
      int i1 = (numx-1)*unitx;
      int i2 = length;
      unitx = i2 - i1;
      vals[numx-1] = 0;
      for(int j = i1 ; j < i2 ; j ++)
      {
         vals[numx-1] += (double)ydata[j]/(double)unitx;
      }
      maxval = maxval > vals[numx-1] ? maxval : vals[numx-1];
      minval = minval < vals[numx-1] ? minval : vals[numx-1];
      
      /* all data collected, start plot */
      double maxabsval = SchurchebMax(maxval, -minval);
      int imaxabsval = (int)maxabsval;
      numy = numy > imaxabsval ? imaxabsval : numy;
      if(numy == 0)
      {
         numy = 1;
      }
      int unity = maxabsval/numy;
      if(unity*numy<maxabsval || unity == 0)
      {
         unity++;
      }
      int width = 0;
      while(imaxabsval>0)
      {
         imaxabsval/=10;
         width++;
      }
      width = SchurchebMax(6, width);
      
      std::vector<std::vector<int> > val2d_positive;
      std::vector<std::vector<int> > val2d_negative;
      val2d_positive.resize(numy+1);
      val2d_negative.resize(numy);
      
      for(int i = 0 ; i <= numy ; i ++)
      {
         val2d_positive[i].resize(numx, 0);
      }
      
      for(int i = 0 ; i < numy ; i ++)
      {
         val2d_negative[i].resize(numx, 0);
      }
      
      for(int i = 0 ; i < numx ; i ++)
      {
         int ival = (int)vals[i] / unity;
         if(ival >= 0)
         {
            val2d_positive[ival][i] = 2;
            for(int j = 0 ; j < ival ; j ++)
            {
               val2d_positive[j][i] = 1;
            }
         }
         else
         {
            val2d_negative[-1-ival][i] = 2;
            for(int j = 0 ; j < -1-ival ; j ++)
            {
               val2d_negative[j][i] = 1;
            }
         }
      }
      
      SCHURCHEB_PRINT("yunit: %d ymax: %f\n",unity,maxabsval);
      for(int i = numy ; i >= 0 ; i --)
      {
         SCHURCHEB_PRINT("%*d| ",width,i*unity);
         for(int j = 0 ; j < numx ; j ++)
         {
            if(val2d_positive[i][j] == 2)
            {
               SCHURCHEB_PRINT("* ");
            }
            else if(val2d_positive[i][j] == 1)
            {
               SCHURCHEB_PRINT("| ");
            }
            else
            {
               SCHURCHEB_PRINT("  ");
            }
         }
         SCHURCHEB_PRINT("\n");
      }
      SCHURCHEB_PRINT("xvals: ");
      unitx = length/numx;
      for(int i = 0 ; i < numx ; i ++)
      {
          SCHURCHEB_PRINT("-|");
      }
      SCHURCHEB_PRINT("- xunit: %d xmax: %d\n",unitx,length);
      if(minval < 0.0)
      {
         for(int i = 0 ; i < numy ; i ++)
         {
            SCHURCHEB_PRINT("%*d| ",width,(-i-1)*unity);
            for(int j = 0 ; j < numx ; j ++)
            {
               if(val2d_negative[i][j] == 2)
               {
                  SCHURCHEB_PRINT("* ");
               }
               else if(val2d_negative[i][j] == 1)
               {
                  SCHURCHEB_PRINT("| ");
               }
               else
               {
                  SCHURCHEB_PRINT("  ");
               }
            }
            SCHURCHEB_PRINT("\n");
         }
      }
      
      return 0;
   }
   template int SchurchebPlotData(int* ydata, int length, int numx, int numy);
   template int SchurchebPlotData(long int* ydata, int length, int numx, int numy);
   template int SchurchebPlotData(float* ydata, int length, int numx, int numy);
   template int SchurchebPlotData(double* ydata, int length, int numx, int numy);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebPlotData(T* ydata, int length, int numx, int numy)
   {
      
      int i;
      std::vector<double> rvals, ivals;
      
      rvals.resize(length);
      ivals.resize(length);
      
      for(i = 0 ; i < length ; i ++)
      {
         rvals[i] = ydata[i].Real();
         ivals[i] = ydata[i].Imag();
      }
      
      SchurchebPlotData( (double*)rvals.data(), length, numx, numy);
      SchurchebPlotData( (double*)ivals.data(), length, numx, numy);
      
      std::vector<double>().swap(rvals);
      std::vector<double>().swap(ivals);
      
      return 0;
   }
   template int SchurchebPlotData(complexs* ydata, int length, int numx, int numy);
   template int SchurchebPlotData(complexd* ydata, int length, int numx, int numy);
   
   int SchurchebSetOutputFile(const char *filename)
   {
      
      if( SCHURCHEB_global::_out_file != stdout )
      {
         /* free the current */
         fclose(SCHURCHEB_global::_out_file);
      }
      
      if ((SCHURCHEB_global::_out_file = fopen(filename, "w")) == NULL)
      {
         printf("Can't open output file.\n");
         SCHURCHEB_global::_out_file = stdout;
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      return SCHURCHEB_SUCCESS;
   }
   
}
