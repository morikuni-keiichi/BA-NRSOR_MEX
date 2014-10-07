/* BANRSOR.c */
#include "mex.h"
#include <math.h>
#include <stdlib.h>

double *AC, *b, *Aei;
mwIndex *ia, *jp;
double eps = 1.0e-6, omg, one = 1.0, zero = 0.0;
int m, maxit, n, nin;
// mwIndex nnz;


// How to use
void usage()
{
    mexPrintf("BANRSOR: BA-GMRES method preconditioned by NR-SOR inner iterations\n");
    mexPrintf("This Matlab-MEX function solves linear least squares problems.\n");
    mexPrintf("  x = BANRSOR(A, b);\n");
    mexPrintf("  [x, relres, iter] = BANRSOR(A, b, tol, maxit);\n\n");
    mexPrintf("  valuable | size | remark \n");
    mexPrintf("  A         m-by-n   coefficient matrix. must be sparse array.\n");
    mexPrintf("  b         m-by-1   right-hand side vector\n");
    mexPrintf("  tol       scalar   tolerance for stopping criterion.\n");
    mexPrintf("  maxit     scalar   maximum number of iterations.\n");
    mexPrintf("  x         n-by-1   resulting approximate solution.\n");
    mexPrintf("  relres   iter-by-1 relative residual history.\n");
    mexPrintf("  iter      scalar   number of iterations required for convergence.\n");
}


// 2-norm
double nrm2(double *x, mwSize k) {

	double absxi, scale = zero, ssq = one, tmp;
	int i;

	for (i=0; i<k; i++) {
		if (x[i] != zero) {
	  		absxi = fabs(x[i]);
	    	if (scale <= absxi) {
	    		tmp = scale/absxi;
	    		ssq = one + ssq*tmp*tmp;
	    		scale = absxi;
	    	} else {
	    		tmp = absxi/scale;
	    		ssq += tmp*tmp;
			}
		}
	}

	return scale*sqrt(ssq);
}


// Givens rotation
void drotg(double *da, double *db, double *c, double *s)
{
	double r, roe, scale, z;

	roe = *db;

    if (fabs(*da) > fabs(*db)) roe = *da;

    scale = fabs(*da) + fabs(*db);

    if (scale != zero) {
   	
	   	r = scale*sqrt(pow(*da/scale, 2.0) + pow(*db/scale, 2.0));
		 
		if (roe<0) r = -r;
	    *c = *da / r;
	    *s = *db / r;
	    z = one;

	    if (fabs(*da) > fabs(*db)) z = *s;

	    if (fabs(*db) >= fabs(*da) && *c != zero) z = one / *c;
		
		*da = r;
		*db = z;

 	} else {

 		*c = one;
    	*s = zero; 
    	r = zero;
    	z = zero;

    	*da = r;
    	*db = z;
    }

}


// NR-SOR inner iterations
void NRSOR(double *rhs, double *x)
{
	double d;
	mwSize i, j, k, k1, k2, l;

	for (j=0; j<n; j++) x[j] = zero;

	for (k=1; k<=nin; k++) {
		for (j=0; j<n; j++) {
			k1 = jp[j];
			k2 = jp[j+1];
			d = zero;
			for (l=k1; l<k2; l++) d += AC[l]*rhs[ia[l]];
			d = d * Aei[j];			
			x[j] = x[j] + d;
			if (k == nin && j == n-1) return;
			for (l=k1; l<k2; l++) {
				i = ia[l];
				rhs[i] -= d*AC[l];
			}
		}
	}
}


// Automatic parameter tuning for NR-SOR inner iterations
void opNRSOR(double *rhs, double *x)
{
	double d, e, res1, res2 = zero, tmp, *y = NULL, *tmprhs = NULL;
	int k;
	mwSize i, ii, j, k1, k2, l;

	// Allocate y
	if ((y = (double *)malloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

	// Allocate tmprhs
	if ((tmprhs = (double *)malloc(sizeof(double) * (m))) == NULL) {
		mexErrMsgTxt("Failed to allocate tmprhs");
	}

	// Initilize
	for (i=0; i<m; i++) tmprhs[i] = rhs[i];

	for (j=0; j<n; j++) {
		x[j] = zero;
		y[j] = zero;	
	}

	// Tune the number of inner iterations 
	for (k=1; k<=100; k++) {

		for (j=0; j<n; j++) {
			k1 = jp[j];
			k2 = jp[j+1];
			d = zero;
			for (l=k1; l<k2; l++) d += AC[l]*rhs[ia[l]];
			d = d * Aei[j];			
			x[j] = x[j] + d;
			for (l=k1; l<k2; l++) {
				i = ia[l];
				rhs[i] -= d*AC[l];
			}
		}

		d = zero;
		for (j=0; j<n; j++) { 
			tmp = fabs(x[j]);
			if (d < tmp) d = tmp;
		}

		e = zero;
		for (j=0; j<n; j++) { 
			tmp = fabs(x[j] - y[j]);
			if (e < tmp) e = tmp;
		}

		if (e<1.0e-1*d || k == 100) {
			nin = k;
			break;

		}

		for (j=0; j<n; j++) y[j] = x[j];

	}

	// Tune the relaxation parameter
	for (k=19; k>0; k--) {
		omg = 1.0e-1 * (double)(k); // omg = 1.9, 1.8, ..., 0.1

		for (i=0; i<m; i++) rhs[i] = tmprhs[i];

		for (j=0; j<n; j++) x[j] = zero;

		for (i=1; i<=nin; i++) {
			for (j=0; j<n; j++) {
				k1 = jp[j];
				k2 = jp[j+1];
				d = zero;
				for (l=k1; l<k2; l++) d += AC[l]*rhs[ia[l]];
				d = omg * d * Aei[j];			
				x[j] += d;
				for (l=k1; l<k2; l++) {
					ii = ia[l];
					rhs[ii] -= d*AC[l];
				}
			}
		}

		res1 = nrm2(rhs, m);

		if (k < 19) {
			if (res1 > res2) {
				omg += 1.0e-1;
				for (j=0; j<n; j++) x[j] = y[j];					
				return;
			} else if (k == 1) {
				omg = 1.0e-1;
				return;
			}
		}

		res2 = res1;

		for (j=0; j<n; j++) y[j] = x[j];
	}
}


// Outer iterations: BA-GMRES 
void BAGMRES(double *iter, double *relres, double *x){

	double **H = NULL, **V = NULL;
	double *c = NULL, *g = NULL, *r = NULL, *w = NULL, *y = NULL, *s = NULL;
	double beta, inprod, nrmATb, nrmATr, tmp, Tol;
	int ind_i, k;
	mwSize i, j, k1, k2, l;

	// Allocate H[maxit]
	if ((H = malloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate H");
	}

	// Allocate H[maxit][maxit+1]
	for (k=0; k<maxit; k++) {
		if ((H[k] = (double *)malloc(sizeof(double) * (k+2))) == NULL) {
			mexErrMsgTxt("Failed to allocate H");
		}
	}

	// Allocate V[maxit+1]
	if ((V = malloc(sizeof(double) * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate V");
	}

	// Allocate V[maxit+1][n]
	for (j=0; j<maxit+1; j++) {
		if ((V[j] = (double *)malloc(sizeof(double) * n)) == NULL) {
			mexErrMsgTxt("Failed to allocate V");
		}
	}	

	// Allocate r
	if ((r = (double *)malloc(sizeof(double) * (m))) == NULL) {
		mexErrMsgTxt("Failed to allocate r");
	}

	// Allocate w
	if ((w = (double *)malloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate w");
	}

	// Allocate
	if ((Aei = (double *)malloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate Aei");
	}

	// Allocate c
	if ((c = (double *)malloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate c");
	}

	// Allocate g
	if ((g = (double *)malloc(sizeof(double) * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate g");
	}

	// Allocate s
	if ((s = (double *)malloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate s");
	}

	// Allocate y
	if ((y = (double *)malloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

	for (j=0; j<n; j++) {
		inprod = zero;
		for (l=jp[j]; l<jp[j+1]; l++) inprod += AC[l]*AC[l];
		if (inprod > zero) {
			Aei[j] = one / inprod;			
		} else {
			mexErrMsgTxt("'warning: ||aj|| = 0");
		}
	}

	// w = A^T * b
	for (j=0; j<n; j++) {
		tmp = zero;
		k1 = jp[j];
		k2 = jp[j+1];
		for (l=k1; l<k2; l++) tmp += AC[l]*b[ia[l]];
		w[j] = tmp;
	}

	// norm of A^T b
  	nrmATb = nrm2(w, n);

  	// Stopping criterion
  	Tol = eps * nrmATb;

  	// r = b  (x0 = 0)
  	for (i=0; i<m; i++) r[i] = b[i];

  	// NR-SOR inner iterations: w = B r
  	opNRSOR(r, w); 

  	for (j=0; j<n; j++) Aei[j] = omg * Aei[j];

  	// beta = norm(Bb)
  	beta = nrm2(w, n);

 	// Normalize
  	tmp = one / beta;
  	for (j=0; j<n; j++) V[0][j] = tmp * w[j]; 

  	// beta e1
  	g[0] = beta;

  	// Main loop
  	for (k=0; k<maxit; k++) {

		for (i=0; i<m; i++) r[i] = zero;
	 	for (j=0; j<n; j++) {
	 		tmp = V[k][j];
	 		for (l=jp[j]; l<jp[j+1]; l++) {
	 			i = ia[l];
	 			r[i] = r[i] + tmp*AC[l];
	 		}
	 	}

	 	// NR-SOR inner iterations: w = B r
		NRSOR(r, w);	 

		// Modified Gram-Schmidt orthogonzlization
		for (i=0; i<k+1; i++) {
			tmp = zero;
			for (j=0; j<n; j++) tmp += w[j]*V[i][j];
			H[k][i] = tmp;
			for (j=0; j<n; j++) w[j] -= tmp*V[i][j];
		}

		// h_{kL1, k}
		H[k][k+1] = nrm2(w, n);

		// Check breakdown
		if (H[k][k+1] > zero) {
			tmp = one / H[k][k+1];
			for (j=0; j<n; j++) V[k+1][j] = tmp * w[j];
		} else {
			printf("h_k+1, k = %.15e, at step %d\n", H[k][k+1], k+1);
			mexErrMsgTxt("Breakdown.");
		}

		// Apply Givens rotations
		for (i=0; i<k; i++) {
			tmp = c[i]*H[k][i] + s[i]*H[k][i+1];
			H[k][i+1] = -s[i]*H[k][i] + c[i]*H[k][i+1];
			H[k][i] = tmp;
		}	

		// Compute Givens rotations
		drotg(&H[k][k], &H[k][k+1], &c[k], &s[k]);

		// Apply Givens rotations
		g[k+1] = -s[k] * g[k];
		g[k] = c[k] * g[k];

		relres[k] = fabs(g[k+1]) / beta; 

		// printf("%.15e\n", relres[k]);

  		if (relres[k] < eps) {

			// Derivation of the approximate solution x_k
			// Backward substitution		
			y[k] = g[k] / H[k][k];
			for (ind_i=k-1; ind_i>-1; ind_i--) {				
				tmp = zero;
				for (l=ind_i+1; l<k; l++) tmp += H[l][ind_i]*y[l];
				y[ind_i] = (g[ind_i] - tmp) / H[ind_i][ind_i];
			}

			// x = V y
			for (j=0; j<n; j++) x[j] = zero;
			for (l=0; l<k+1; l++) {
				for (j=0; j<n; j++) x[j] += V[l][j]*y[l];
			}

			// r = A x 	
			for (i=0; i<m; i++) r[i] = zero;
	 		for (j=0; j<n; j++) {
	 			tmp = x[j];
		 		for (l=jp[j]; l<jp[j+1]; l++) {
		 			i = ia[l];
		 			r[i] = r[i] + tmp*AC[l];
		 		}
		 	}

		 	for (i=0; i<m; i++) r[i] = b[i] - r[i];

			// w = A^T r
			for (j=0; j<n; j++) {
				tmp = zero;
				k1 = jp[j];
				k2 = jp[j+1];
				for (l=k1; l<k2; l++) tmp += AC[l]*r[ia[l]];
				w[j] = tmp;
			}

		 	nrmATr = nrm2(w, n);

			// printf("%d, %.15e\n", k, nrm2(w, n)/nrmATb);

		 	// Convergence check
		  	if (nrmATr < Tol) {

				*iter = k+1;

				for (j=0; j<maxit+1; j++) {
					free(V[j]);
				}		
				free(V);

			  	for (i=0; i<maxit; i++) {
					free(H[i]);
				}
				free(H);

				free(c);
				free(g);
				free(r);
				free(s);
				free(w);
				free(y);

  				// printf("Required number of iterations: %d\n", (int)(*iter));
  				printf("Successfully converged.\n");
  				

				return;

			}
		}
	}

	printf("Failed to converge.\n");

	*iter = k;

	for (j=0; j<maxit+1; j++) {
		free(V[j]);
	}		
	free(V);

  	for (i=0; i<maxit; i++) {
		free(H[i]);
	}
	free(H);

	free(c);
	free(g);
	free(r);
	free(s);
	free(w);
	free(y);

}


// Main
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *relres = NULL, *x = NULL, *iter;
	// mwSize nzmax;

	// Check the number of input arguments
    if(nrhs > 4){
    	usage();
        mexWarnMsgTxt("Too many inputs. Ignored extras.\n");
    }

    // Check the number of output arguments
    if(nlhs > 3){
		usage();    	
        mexWarnMsgTxt("Too many outputs. Ignored extras.");
    }

    /* Check for proper number of input and output arguments */
	if (nrhs < 2) {
		usage();
        mexErrMsgTxt("Please input b.");
    }

    // Check the number of input arguments
    if (nrhs < 1) {
    	usage();
        mexErrMsgTxt("Please input A.");
    }

	// Check the 1st argument
    if (!mxIsSparse(prhs[0]))  {
        mexErrMsgTxt("1st input argument must be a sparse array.");
    }

    if (mxIsComplex(prhs[0])) {
    	mexErrMsgTxt("1st input argument must be a real array.");
    }

    m = (int)mxGetM(prhs[0]);
    n = (int)mxGetN(prhs[0]);
    // nnz = *((int)mxGetJc(prhs[0]) + n);
    // nzmax = mxGetNzmax(prhs[0]);

    ia = mxGetIr(prhs[0]);
    jp = mxGetJc(prhs[0]);
    AC = mxGetPr(prhs[0]);

	// Check the 2nd argument
    if (mxGetM(prhs[1]) != m) {
    	mexErrMsgTxt("The length of b is not the numer of rows of A.");
    }    

    b = mxGetPr(prhs[1]);

	// Check the 3rd argument
    // Set eps
    if (nrhs < 3) {
        mexPrintf("Default: stopping criterion is set to 1e-6.\n");
    } else {
    	if (mxIsComplex(prhs[2]) || mxGetM(prhs[2])*mxGetN(prhs[2])!=1) {
    		mexErrMsgTxt("3nd argument must be a scalar");
    	} else {
    		eps = *(double *)mxGetPr(prhs[2]);
    		if (eps<zero || eps>=one) {
    			mexErrMsgTxt("3nd argument should be positive and less than or equal to 1.");
    		}
    	}
    }
    
	// Check the 4th argument
	// Set maxit
    if (nrhs < 4) {
    	maxit = n;
    	mexPrintf("Default: max number of iterations is set to the number of columns.\n");
   	} else {
   		if (mxIsComplex(prhs[3]) || mxGetM(prhs[3])*mxGetN(prhs[3])!=1) {
    		mexErrMsgTxt("4th argument must be a scalar");
    	} else {
   			maxit = (int)*mxGetPr(prhs[3]);
   			if (maxit < 1) {   				
   				mexErrMsgTxt("4th argument must be a positive scalar");
   			}
   		}
	}

    // Allocate x
	if ((x = (double *)malloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate x");
	}

	// Allocation
	if ((relres = (double *)malloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate relres");
	}	

	plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(maxit, 1, mxREAL); 
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);  	

    x = mxGetPr(plhs[0]);
    relres = mxGetPr(plhs[1]);
    iter = mxGetPr(plhs[2]);

	// BA-GMRES method
    BAGMRES(iter, relres, x);

    // Reshape relres
    mxSetPr(plhs[1], relres);
    mxSetM(plhs[1], (int)*iter);
    mxSetN(plhs[1], 1);

}
