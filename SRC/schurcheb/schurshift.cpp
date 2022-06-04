#include "schurshift.hpp"

/**
 * @file schurshift.cpp
 * @brief Helper class for the eigenvalues corresponding to each shift
 */

namespace schurcheb
{
   
   SchurEigvalueClass::SchurEigvalueClass(const SchurEigvalueClass &str)
   {
      SCHURCHEB_ERROR("Copy constructor not implemented for class SchurEigvalueClass.");
   }
   
   SchurEigvalueClass::SchurEigvalueClass(SchurEigvalueClass &&str)
   {
      SCHURCHEB_ERROR("Move constructor not implemented for class SchurEigvalueClass.");
   }
   
   SchurEigvalueClass& SchurEigvalueClass::operator= (const SchurEigvalueClass &str)
   {
      SCHURCHEB_ERROR("= operator not implemented for class SchurEigvalueClass.");
      return *this;
   }
   
   SchurEigvalueClass& SchurEigvalueClass::operator= (SchurEigvalueClass &&str)
   {
      SCHURCHEB_ERROR("= operator not implemented for class SchurEigvalueClass.");
      return *this;
   }
   
   SchurEigvalueClass::SchurEigvalueClass() : ArnoldiMatrixClass<vector_par_double, double>()
   {
      _shift = 0.0;
      _nB = 0;
      _nC = 0;
      _n = 0;
      
      _bonly = false;
      
      _mem = 0;
      _nnz = 0;
      
      _shift_invert = false;
      _direct_solver_opt = 1;
      
#ifdef SCHURCHEB_MKL
      _phase = 0;
#endif

#ifdef SCHURCHEB_MUMPS
      _mphase = 0;
      _mphases = 0;
      _mumpss._print_level  = 0;
#endif

   }

   int SchurEigvalueClass::Clear()
   {
      _B.Clear();
      _E.Clear();
      _F.Clear();
      _C.Clear();
      
      _MB.Clear();
      _ME.Clear();
      _MF.Clear();
      _MC.Clear();
      
      _B_solve.Clear();
      
      _temp_x_B.Clear();
      _temp_y_B.Clear();
      _temp_x_C.Clear();
      
      _shift = 0.0;
      _nB = 0;
      _nC = 0;
      _n = 0;

      _mem = 0;
      _nnz = 0;

#ifdef SCHURCHEB_MKL
      
      if(_solve_opt == 1 && _phase != 0)
      {
         /* clear */
         int err = 0;
         _phase = -1;
         pardiso ( _pt.GetData(), 
                     &_maxfct, 
                     &_mnum, 
                     &_mtype, 
                     &_phase, 
                     &_nB, 
                     _BsMB.GetData(), 
                     _BsMB.GetI(), 
                     _BsMB.GetJ(), 
                     NULL, 
                     &_nrhs, 
                     _iparam.GetData(), 
                     &_msglvl, 
                     NULL, 
                     NULL, 
                     &err);
         
         SCHURCHEB_CHKERR(err != 0);
      }
      _phase = 0;
      
      _BsMB.Clear();
      _Bu.Clear();
      _MBu.Clear();

      _pt.Clear();
      _iparam.Clear();
#endif

#ifdef SCHURCHEB_MUMPS
      if(_solve_opt==2 && _mphase != 0)
      {
         MUMPSFree(_mumps);
         _mphase = 0;
      }
      if(_mphases != 0)
      {
         MUMPSFree(_mumpss);
         _mphases = 0;
      }
      _CsMC.Clear();
#endif

      return SCHURCHEB_SUCCESS;
   }

   SchurEigvalueClass::~SchurEigvalueClass()
   {
      /* destructor */
      this->Clear();
   }

   void SchurEigvalueClass::SetSolveOption(int solve_opt)
   {
      switch(solve_opt)
      {
         case 1:
         {
#ifdef SCHURCHEB_MKL
            _solve_opt = 1;
#else
            _solve_opt = 0;
#endif
            break;
         }
         case 2:
         {
#ifdef SCHURCHEB_MUMPS
            _solve_opt = 2;
#else
            _solve_opt = 0;
#endif
            break;
         }
         default:
         {
            _solve_opt = 0;
         }
      }
   }

   void SchurEigvalueClass::SetPrintLevel(int print_level)
   {
#ifdef SCHURCHEB_MUMPS
            _mumps._print_level = print_level;
#endif
   }

   int SchurEigvalueClass::UpdateShift(double shift)
   {
      
      this->_shift = shift;
      
      /* Setup preconditioner for (B-shift*MB) */
      matrix_csr_double sMB, BsMB;

      if(_solve_opt != 1)
      {
         sMB = this->_MB.GetDiagMat();
         sMB.Scale(-shift);
         
         /* BsMB = B - shift*MB */
         CsrMatrixAddHost( this->_B.GetDiagMat(), sMB, BsMB);
      }

      switch(_solve_opt)
      {
         case 1:
         {
#ifdef SCHURCHEB_MKL
         
            _phase = -1; /* clean memory */
            
            int err = 0;
            pardiso ( _pt.GetData(), 
                        &_maxfct, 
                        &_mnum, 
                        &_mtype, 
                        &_phase, 
                        &_nB, 
                        _BsMB.GetData(), 
                        _BsMB.GetI(), 
                        _BsMB.GetJ(), 
                        NULL, 
                        &_nrhs, 
                        _iparam.GetData(), 
                        &_msglvl, 
                        NULL, 
                        NULL, 
                        &err);
            
            SCHURCHEB_CHKERR(err != 0);
            
            _BsMB.Clear();
            sMB = _MBu;
            sMB.Scale(-shift);
            
            /* BsMB = B - shift*MB */
            CsrMatrixAddHost( _Bu, sMB, _BsMB);
            _pt.Fill(0);
            
            _phase = 12; /* setup phase */
            
            pardiso ( _pt.GetData(), 
                        &_maxfct, 
                        &_mnum, 
                        &_mtype, 
                        &_phase, 
                        &_nB, 
                        _BsMB.GetData(), 
                        _BsMB.GetI(), 
                        _BsMB.GetJ(), 
                        NULL, 
                        &_nrhs, 
                        _iparam.GetData(), 
                        &_msglvl, 
                        NULL, 
                        NULL, 
                        &err);
            
            _mem += (_iparam[14]+_iparam[15])/1024.0;
            _nnz += _iparam[16];
      
            SCHURCHEB_CHKERR(err != 0);
      
#endif
            break;
         }
         case 2:
         {
#ifdef SCHURCHEB_MUMPS
            _mphase = 1;
            MUMPSFree(_mumps);
            MUMPSInitSeq(_mumps, BsMB);
#endif
            break;
         }
         default:
         {
            vector_seq_double dummyx, dummyrhs;
            
            _B_solve.Clear();
            
            /* assign matrix */
            _B_solve.SetMatrix(BsMB);
            
            /* we use exact solve */
            _B_solve.SetMaxNnzPerRow( this->_nB);
            _B_solve.SetDropTolerance( 0.0);
            
            /* set solve option */
            _B_solve.SetOption(kIluOptionILUT);
            
            /* use RCM ordering */
            _B_solve.SetPermutationOption(kIluReorderingRcm);
            
            /* setup the solver */
            _B_solve.Setup(dummyx, dummyrhs);
         }
      }

      if(_shift_invert)
      {
         switch(_direct_solver_opt)
         {
            case 1:
            {
#ifdef SCHURCHEB_MUMPS

               parallel_log parlog = _CsMC;

               if(_mphases != 0)
               {
                  MUMPSFree(_mumpss);
                  _mphases = 0;
               }
               _CsMC.Clear();
               
               /* select a shift */
               
               char aropt[2];
               aropt[0] = 'S';
               aropt[1] = 'R';
               
               int msteps = SchurchebMin( _nC, 20);
               int niter = 1000;
               int nev = 1;
               double tol_eig = 1e-01;
               bool sym = false;
               matrix_dense_double Z;
               int nmvs;
               double tmvs;
               
               vector_seq_double dr, di;
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = *this;
               
               _shift_invert = false;
               ArpackArnoldi<vector_par_double>( temp_mat, msteps, niter, nev, aropt, sym,
                  tol_eig, Z, dr, di, false, nmvs, tmvs, parlog);
               _shift_invert = true;

               _shift_invert_shift = SchurchebMax( -dr[0], 0.0) + tol_eig;

               //cout<<_shift_invert_shift<<endl;

               /* form the S matrix
                * C - sMC - ((E-sME)(B-sMB)^{-1}(F-sMF))
                */
               matrix_csr_par_double sMC;
               sMC = _MC;
               sMC.Scale(-_shift);
               ParallelCsrMatrixAddHost(_C, sMC, _CsMC);
               
               /* now compute the E, note that this is symmetric so 
                * we only need to compute one of them 
                */
               matrix_csr_double sME, EsME, EBF, CsMCd;
               
               CsMCd = _CsMC.GetDiagMat();
               
               sME = this->_ME.GetDiagMat();
               sME.Scale(-shift);
               
               /* BsMB = B - shift*MB */
               CsrMatrixAddHost( this->_E.GetDiagMat(), sME, EsME);
               
               /* next we need to form the diagonal of the matrix 
                * CsMC - EsME*(BsMB)^{-1}*FsMF.
                */
               vector_seq_double vF, vBF, vEBF;
               vF.Setup(_nB);
               vBF.Setup(_nB);
               vEBF.Setup(_nC);
               
               double one = 1.0, zero = 0.0, mone = -1.0;
               
               int EBFidx = 0;
               EBF.Setup(_nC, _nC, _shift_invert_reserve, true);
               
               for( int i = 0 ; i < _nC ; i ++)
               {
                  /* setup right hand side */
                  vF.Fill(0.0);
                  int j1 = EsME.GetI()[i];
                  int j2 = EsME.GetI()[i+1];
                  for( int j = j1 ; j < j2 ; j ++)
                  {
                     vF[EsME.GetJ()[j]] = EsME.GetData()[j];
                  }
                  
                  /* solve */
                  
                  switch(_solve_opt)
                  {
                     case 1:
                     {
                        /* y_B = (B-sMB)^{-1}(F-sMF)) x */
#ifdef SCHURCHEB_MKL
                        _phase = 33; /* solve */
                        
                        int err = 0;
                        pardiso ( _pt.GetData(), 
                                    &_maxfct, 
                                    &_mnum, 
                                    &_mtype, 
                                    &_phase, 
                                    &_nB, 
                                    _BsMB.GetData(), 
                                    _BsMB.GetI(), 
                                    _BsMB.GetJ(), 
                                    NULL, 
                                    &_nrhs, 
                                    _iparam.GetData(), 
                                    &_msglvl, 
                                    vF.GetData(), 
                                    vBF.GetData(), 
                                    &err);
                        
                        SCHURCHEB_CHKERR(err != 0);
#endif
                        break;
                     }
                     case 2:
                     {
#ifdef SCHURCHEB_MUMPS
                        _mphase = 2;
                        MUMPSSolveSeq(_mumps, vF.GetData());
                        SCHURCHEB_MEMCPY( vBF.GetData(), vF.GetData(), _nB, kMemoryHost, kMemoryHost, double);
#endif
                        break;
                     }
                     default:
                     {
                        this->_B_solve.Solve( vBF, vF);
                     }
                  }
                  
                  /* MatVec with E */
                  EsME.MatVec( 'N', mone, vBF, zero, vEBF);
                  
                  vEBF[i] += _shift_invert_shift;
                  
                  /* put into CSR matrix */
                  for(int j = 0 ; j < _nC ; j ++)
                  {
                     if(vEBF[j] != zero)
                     {
                        /* an nnz, put into the matrix EBF */
                        EBF.PushBack( j, vEBF[j]);
                        EBFidx++;
                     }
                  }
                  EBF.GetI()[i+1] = EBFidx;
               }
               EBF.SetNumNonzeros();
               
               CsrMatrixAddHost( CsMCd, EBF, _CsMC.GetDiagMat());
               
               EBF.Clear();
               CsMCd.Clear();
               sME.Clear();
               EsME.Clear();
               
               if(parallel_log::_grank == 0)
               {
                  //_CsMC.GetDiagMat().Plot(NULL,0,0,6);
               }
               
               MUMPSInit(_mumpss, _CsMC);
               _mphases = 1;
               
#else
               _shift_invert = false;
#endif
               break;
            }
            default:
            {
               break;
            }
         }
      }
      
      /* done, free matrices */
      sMB.Clear();
      BsMB.Clear();
      
      return SCHURCHEB_SUCCESS;
   }

   int SchurEigvalueClass::Setup(matrix_csr_par_double &B,
               matrix_csr_par_double &E,
               matrix_csr_par_double &F,
               matrix_csr_par_double &C,
               matrix_csr_par_double &MB,
               matrix_csr_par_double &ME,
               matrix_csr_par_double &MF,
               matrix_csr_par_double &MC,
               double shift,
               parallel_log &parlog)
   {
      
      this->Clear();
      
      this->_B = B;
      this->_E = E;
      this->_F = F;
      this->_C = C;
      this->_MB = MB;
      this->_ME = ME;
      this->_MF = MF;
      this->_MC = MC;
      
      this->_C.SetupMatvec();
      this->_MC.SetupMatvec();
      
      this->_nB = B.GetNumRowsLocal();
      this->_nC = C.GetNumRowsLocal();
      
      this->_n = this->_nB + this->_nC;
      
      this->_shift = shift;
      
      /* Setup preconditioner for (B-shift*MB) */
      matrix_csr_double sMB, BsMB;
      
      if(_solve_opt == 1)
      {
#ifdef SCHURCHEB_MKL
         /* in this case we store only the upper triangular part */
         matrix_csr_double Bu_temp, MBu_temp;
         matrix_csr_double Bu0, MBu0;
         TriU( this->_B.GetDiagMat(), Bu_temp);
         TriU( this->_MB.GetDiagMat(), MBu_temp);
         
         Bu0 = Bu_temp;
         MBu0 = MBu_temp;
         
         Bu0.Fill(0.0);
         MBu0.Fill(0.0);
         
         /* add _B and _MB with zero matrices so that they have the same pattern */
         CsrMatrixAddHost( Bu_temp, MBu0, _Bu);
         CsrMatrixAddHost( MBu_temp, Bu0, _MBu);
         
         _Bu.SortRow();
         _MBu.SortRow();
         
         Bu_temp.Clear();
         MBu_temp.Clear();
         Bu0.Clear();
         MBu0.Clear();
         
         sMB = _MBu;
         sMB.Scale(-shift);
         CsrMatrixAddHost( _Bu, sMB, _BsMB);
         
#endif
      }
      else
      {
         sMB = this->_MB.GetDiagMat();
         sMB.Scale(-shift);
         
         /* BsMB = B - shift*MB */
         CsrMatrixAddHost( this->_B.GetDiagMat(), sMB, BsMB);
      }
      
      vector_seq_double dummyx, dummyrhs;
      
      switch(_solve_opt)
      {
         case 1:
         {
#ifdef SCHURCHEB_MKL
         
            _pt.Setup(64, true);
            _maxfct = 1; /* only one matrix at same time */
            _mnum = 1; /* only one matrix */
            //_mtype = 1; /* real and structually symmetric */
            _mtype = -2; /* symmetric indefinite (A-sigma*M can be indefinite) */
            _nrhs = 1; /* single rhs */
            _iparam.Setup(64, true);
            
            /* https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html */
            _iparam[0] = 1; /* do not use default */
            _iparam[1] = 0; /* MD ordering */
            _iparam[3] = 0; /* always compute the factorization */
            _iparam[4] = 0; /* use the perm in iparam[1] */
            _iparam[7] = 2; /* max num of iterative refinement */
            _iparam[9] = 13; /* default for mat type 11 */
            _iparam[10] = 1; /* enable scaling */
            _iparam[12] = 1; /* Enable matching */
            _iparam[34] = 1; /* C style zero-based */
            
            _msglvl = 0; /* no outputs */
            
            _phase = 12; /* setup phase */
            
            int err = 0;
            
            pardiso ( _pt.GetData(), 
                        &_maxfct, 
                        &_mnum, 
                        &_mtype, 
                        &_phase, 
                        &_nB, 
                        _BsMB.GetData(), 
                        _BsMB.GetI(), 
                        _BsMB.GetJ(), 
                        NULL, 
                        &_nrhs, 
                        _iparam.GetData(), 
                        &_msglvl, 
                        NULL, 
                        NULL, 
                        &err);
            
            SCHURCHEB_CHKERR(err != 0);
      
            _mem += (_iparam[14]+_iparam[15])/1024.0;
            _nnz += _iparam[16];
      
            //_msglvl = 0; /* see outputs */
            
#endif
            break;
         }
         case 2:
         {
#ifdef SCHURCHEB_MUMPS
            _mphase = 1;
            MUMPSInitSeq(_mumps, BsMB);
#endif
            break;
         }
         default:
         {

            /* assign matrix */
            _B_solve.SetMatrix(BsMB);
            
            /* we use exact solve */
            _B_solve.SetMaxNnzPerRow( this->_nB);
            _B_solve.SetDropTolerance( 0.0);
            
            /* set solve option */
            _B_solve.SetOption(kIluOptionILUT);
            
            /* use RCM ordering */
            _B_solve.SetPermutationOption(kIluReorderingRcm);
            
            /* setup the solver */
            _B_solve.Setup(dummyx, dummyrhs);
            
            //BsMB.Plot(NULL, 0, parallel_log::_grank, 6); 
            //_B_solve.GetL().Plot(NULL, 0, parallel_log::_grank, 6); 
            //_B_solve.GetD().Plot(0, parallel_log::_grank, 6); 
            //_B_solve.GetU().Plot(NULL, 0, parallel_log::_grank, 6); 
         }
      }

      /* create buffer vectors */
      this->_temp_x_B.Setup( this->_B.GetNumRowsLocal(), this->_B.GetRowStartGlobal(), this->_B.GetNumRowsGlobal(), true, this->_B);
      this->_temp_y_B.Setup( this->_B.GetNumRowsLocal(), this->_B.GetRowStartGlobal(), this->_B.GetNumRowsGlobal(), true, this->_B);
      this->_temp_x_C.Setup( this->_C.GetNumRowsLocal(), this->_C.GetRowStartGlobal(), this->_C.GetNumRowsGlobal(), true, this->_C);
      
      if(_shift_invert)
      {
         switch(_direct_solver_opt)
         {
            case 1:
            {
#ifdef SCHURCHEB_MUMPS
               /* select a shift */
               
               char aropt[2];
               aropt[0] = 'S';
               aropt[1] = 'R';
               
               int msteps = SchurchebMin( _nC, 20);
               int niter = 1000;
               int nev = 1;
               double tol_eig = 1e-01;
               bool sym = false;
               matrix_dense_double Z;
               int nmvs;
               double tmvs;
               
               vector_seq_double dr, di;
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = *this;
               
               _shift_invert = false;
               ArpackArnoldi<vector_par_double>( temp_mat, msteps, niter, nev, aropt, sym,
                  tol_eig, Z, dr, di, false, nmvs, tmvs, parlog);
               _shift_invert = true;

               _shift_invert_shift = SchurchebMax( -dr[0], 0.0) + tol_eig;

               //cout<<_shift_invert_shift<<endl;

               /* form the S matrix
                * C - sMC - ((E-sME)(B-sMB)^{-1}(F-sMF))
                */
               matrix_csr_par_double sMC;
               sMC = _MC;
               sMC.Scale(-_shift);
               ParallelCsrMatrixAddHost(_C, sMC, _CsMC);
               
               /* now compute the E, note that this is symmetric so 
                * we only need to compute one of them 
                */
               matrix_csr_double sME, EsME, EBF, CsMCd;
               
               CsMCd = _CsMC.GetDiagMat();
               
               sME = this->_ME.GetDiagMat();
               sME.Scale(-shift);
               
               /* BsMB = B - shift*MB */
               CsrMatrixAddHost( this->_E.GetDiagMat(), sME, EsME);
               
               /* next we need to form the diagonal of the matrix 
                * shift*I + CsMC - EsME*(BsMB)^{-1}*FsMF.
                */
               vector_seq_double vF, vBF, vEBF;
               vF.Setup(_nB);
               vBF.Setup(_nB);
               vEBF.Setup(_nC);
               
               double one = 1.0, zero = 0.0, mone = -1.0;
               
               int EBFidx = 0;
               EBF.Setup(_nC, _nC, _nC*_nC, true);
               
               for( int i = 0 ; i < _nC ; i ++)
               {
                  /* setup right hand side */
                  vF.Fill(0.0);
                  int j1 = EsME.GetI()[i];
                  int j2 = EsME.GetI()[i+1];
                  for( int j = j1 ; j < j2 ; j ++)
                  {
                     vF[EsME.GetJ()[j]] = EsME.GetData()[j];
                  }
                  
                  /* solve */
                  
                  switch(_solve_opt)
                  {
                     case 1:
                     {
                        /* y_B = (B-sMB)^{-1}(F-sMF)) x */
#ifdef SCHURCHEB_MKL
                        _phase = 33; /* solve */
                        
                        int err = 0;
                        pardiso ( _pt.GetData(), 
                                    &_maxfct, 
                                    &_mnum, 
                                    &_mtype, 
                                    &_phase, 
                                    &_nB, 
                                    _BsMB.GetData(), 
                                    _BsMB.GetI(), 
                                    _BsMB.GetJ(), 
                                    NULL, 
                                    &_nrhs, 
                                    _iparam.GetData(), 
                                    &_msglvl, 
                                    vF.GetData(), 
                                    vBF.GetData(), 
                                    &err);
                        
                        SCHURCHEB_CHKERR(err != 0);
#endif
                        break;
                     }
                     case 2:
                     {
#ifdef SCHURCHEB_MUMPS
                        _mphase = 2;
                        MUMPSSolveSeq(_mumps, vF.GetData());
                        SCHURCHEB_MEMCPY( vBF.GetData(), vF.GetData(), _nB, kMemoryHost, kMemoryHost, double);
#endif
                        break;
                     }
                     default:
                     {
                        this->_B_solve.Solve( vBF, vF);
                     }
                  }
                  
                  /* MatVec with E */
                  EsME.MatVec( 'N', mone, vBF, zero, vEBF);
                  
                  vEBF[i] += _shift_invert_shift;
                  
                  /* put into CSR matrix */
                  for(int j = 0 ; j < _nC ; j ++)
                  {
                     if(vEBF[j] != zero)
                     {
                        /* an nnz, put into the matrix EBF */
                        EBF.PushBack( j, vEBF[j]);
                        EBFidx++;
                     }
                  }
                  EBF.GetI()[i+1] = EBFidx;
               }
               EBF.SetNumNonzeros();
               
               CsrMatrixAddHost( CsMCd, EBF, _CsMC.GetDiagMat());
               
               //CsMCd.Plot(NULL,parallel_log::_grank,0,6);
               //EBF.Plot(NULL,parallel_log::_grank,0,6);
               
               EBF.Clear();
               CsMCd.Clear();
               sME.Clear();
               EsME.Clear();
               
               //cout<<_shift<<endl;
               //_CsMC.GetDiagMat().Plot(NULL,parallel_log::_grank,0,6);
               //_CsMC.GetOffdMat().Plot(NULL,parallel_log::_grank,0,6);
               
               MUMPSInit(_mumpss, _CsMC);
               _mphases = 1;
               
#else
               _shift_invert = false;
#endif
               break;
            }
            default:
            {
               break;
            }
         }
      }
       
      /* done, free matrices */
      sMB.Clear();
      BsMB.Clear();
      
      return SCHURCHEB_SUCCESS;
   }

   int SchurEigvalueClass::MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y)
   {
      
      if(this->_bonly)
      {
         // in this case apply the operattor B^{-1}MB
         SCHURCHEB_CHKERR(_shift_invert);
         SCHURCHEB_CHKERR(this->_shift != 0.0);
         
         double zero = 0.0;
         double one = 1.0;
         
         /* x_B = MB x */
         this->_MB.GetDiagMat().MatVec( 'N', alpha, x.GetDataVector(), zero, this->_temp_x_B.GetDataVector());
         
         /* y = B \ x_B */
         switch(_solve_opt)
         {
            case 1:
            {
               /* y_B = (B-sMB)^{-1}(F-sMF)) x */
#ifdef SCHURCHEB_MKL
               _phase = 33; /* solve */
               
               int err = 0;
               pardiso ( _pt.GetData(), 
                           &_maxfct, 
                           &_mnum, 
                           &_mtype, 
                           &_phase, 
                           &_nB, 
                           _BsMB.GetData(), 
                           _BsMB.GetI(), 
                           _BsMB.GetJ(), 
                           NULL, 
                           &_nrhs, 
                           _iparam.GetData(), 
                           &_msglvl, 
                           this->_temp_x_B.GetData(), 
                           this->_temp_y_B.GetData(), 
                           &err);
               
               SCHURCHEB_CHKERR(err != 0);
#endif
               break;
            }
            case 2:
            {
#ifdef SCHURCHEB_MUMPS
               _mphase = 2;
               MUMPSSolveSeq(_mumps, this->_temp_x_B.GetData());
               SCHURCHEB_MEMCPY( this->_temp_y_B.GetData(), this->_temp_x_B.GetData(), _nB, kMemoryHost, kMemoryHost, double);
#endif
               break;
            }
            default:
            {
               this->_B_solve.Solve( this->_temp_y_B.GetDataVector(), this->_temp_x_B.GetDataVector());
            }
         }
         
         y.Scale(beta);
         y.Axpy( one, this->_temp_y_B);
         
         return SCHURCHEB_SUCCESS;
      }
      
      /* matvec with shift value s
       * y = alpha * (C - sMC) x - alpha * ((E-sME)(B-sMB)^{-1}(F-sMF)) x + beta * y
       */
      if(_shift_invert)
      {
#ifdef SCHURCHEB_MUMPS
         this->_temp_x_C.Fill(0.0);
         this->_temp_x_C.Axpy( alpha, x);
         MUMPSSolve( _mumpss, this->_temp_x_C.GetData());
         y.Scale(beta);
         y.Axpy( 1.0, this->_temp_x_C);
#endif
         return SCHURCHEB_SUCCESS;
      }

      double zero = 0.0;
      double one = 1.0;
      double mone = -1.0;
      double mshift = - this->_shift;
      
      /* x_C = (C - sMC) x */
      this->_C.MatVec( 'N', one, x, zero, this->_temp_x_C);
      this->_MC.MatVec( 'N', mshift, x, one, this->_temp_x_C);
      
      /* x_B = (F-sMF)) x */
      this->_F.GetDiagMat().MatVec( 'N', one, x.GetDataVector(), zero, this->_temp_x_B.GetDataVector());
      this->_MF.GetDiagMat().MatVec( 'N', mshift, x.GetDataVector(), one, this->_temp_x_B.GetDataVector());
      
      
      switch(_solve_opt)
      {
         case 1:
         {
            /* y_B = (B-sMB)^{-1}(F-sMF)) x */
#ifdef SCHURCHEB_MKL
            _phase = 33; /* solve */
            
            int err = 0;
            pardiso ( _pt.GetData(), 
                        &_maxfct, 
                        &_mnum, 
                        &_mtype, 
                        &_phase, 
                        &_nB, 
                        _BsMB.GetData(), 
                        _BsMB.GetI(), 
                        _BsMB.GetJ(), 
                        NULL, 
                        &_nrhs, 
                        _iparam.GetData(), 
                        &_msglvl, 
                        this->_temp_x_B.GetData(), 
                        this->_temp_y_B.GetData(), 
                        &err);
            
            SCHURCHEB_CHKERR(err != 0);
#endif
            break;
         }
         case 2:
         {
#ifdef SCHURCHEB_MUMPS
            _mphase = 2;
            MUMPSSolveSeq(_mumps, this->_temp_x_B.GetData());
            SCHURCHEB_MEMCPY( this->_temp_y_B.GetData(), this->_temp_x_B.GetData(), _nB, kMemoryHost, kMemoryHost, double);
#endif
            break;
         }
         default:
         {
            this->_B_solve.Solve( this->_temp_y_B.GetDataVector(), this->_temp_x_B.GetDataVector());
         }
      }

      /* x_C = (C - sMC) x - ((E-sME)(B-sMB)^{-1}(F-sMF)) x */
      this->_E.GetDiagMat().MatVec( 'N', mone, this->_temp_y_B.GetDataVector(), one, this->_temp_x_C.GetDataVector());
      this->_ME.GetDiagMat().MatVec( 'N', this->_shift, this->_temp_y_B.GetDataVector(), one, this->_temp_x_C.GetDataVector());
      
      /* y = alpha*x_C + beta * y */
      y.Scale(beta);
      y.Axpy(alpha, this->_temp_x_C);
      
      return SCHURCHEB_SUCCESS;
   }

   int SchurEigvalueClass::MatVec2( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y)
   {
      /* matvec with shift value s
       * y = - alpha * ((E-sME)(B-sMB)^{-1}(F-sMF)) x + beta * y
       * y is of length n_B
       */
      
      double zero = 0.0;
      double one = 1.0;
      //double mone = -1.0;
      double mshift = - this->_shift;
      double malpha = - alpha;
      
      /* x_B = (F-sMF)) x */
      this->_F.GetDiagMat().MatVec( 'N', one, x.GetDataVector(), zero, this->_temp_x_B.GetDataVector());
      this->_MF.GetDiagMat().MatVec( 'N', mshift, x.GetDataVector(), one, this->_temp_x_B.GetDataVector());
      
      switch(_solve_opt)
      {
         case 1:
         {
         /* y_B = (B-sMB)^{-1}(F-sMF)) x */
#ifdef SCHURCHEB_MKL
            _phase = 33; /* solve */
            
            int err = 0;
            pardiso ( _pt.GetData(), 
                        &_maxfct, 
                        &_mnum, 
                        &_mtype, 
                        &_phase, 
                        &_nB, 
                        _BsMB.GetData(), 
                        _BsMB.GetI(), 
                        _BsMB.GetJ(), 
                        NULL, 
                        &_nrhs, 
                        _iparam.GetData(), 
                        &_msglvl, 
                        this->_temp_x_B.GetData(), 
                        this->_temp_y_B.GetData(), 
                        &err);
            
            SCHURCHEB_CHKERR(err != 0);
#endif
            break;
         }
         case 2:
         {
#ifdef SCHURCHEB_MUMPS
            _mphase = 2;
            MUMPSSolveSeq(_mumps, this->_temp_x_B.GetData());
            SCHURCHEB_MEMCPY( this->_temp_y_B.GetData(), this->_temp_x_B.GetData(), _nB, kMemoryHost, kMemoryHost, double);
#endif
            break;
         }
         default:
         {
            this->_B_solve.Solve( this->_temp_y_B.GetDataVector(), this->_temp_x_B.GetDataVector());
         }
      }
      
      /* y = - alpha * x_C + beta * y */
      y.Scale(beta);
      y.Axpy(malpha, this->_temp_y_B);
      
      return 0;
   }

   int SchurEigvalueClass::SetupVectorPtrStr(vector_par_double &v)
   {
      /* this function is used in the Krylov method */
      _bonly ? v.SetupPtrStr( this->_B) : v.SetupPtrStr( this->_C);
      return SCHURCHEB_SUCCESS;
   }

   int SchurEigvalueClass::GetNumRowsLocal() const
   {
      return _bonly ? _nB : _nC;
   }

   int SchurEigvalueClass::GetNumColsLocal() const
   {
      return _bonly ? _nB : _nC;
   }

   int SchurEigvalueClass::SetBSolveOption(bool bonly)
   {
      this->_bonly = bonly;
      return 0;
   }

   MPI_Comm SchurEigvalueClass::GetComm() const
   {
      /* return the global one */
      return _bonly ? this->_B.GetComm() : this->_C.GetComm();
   }

   int SchurEigvalueClass::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
   {
      _bonly ? this->_B.GetMpiInfo(np, myid, comm) : this->_C.GetMpiInfo(np, myid, comm);
      return 0;
   }

   int SchurEigvalueClass::GetDataLocation() const
   {
      /* currently only supports host version */
      return kMemoryHost;
   }
}
