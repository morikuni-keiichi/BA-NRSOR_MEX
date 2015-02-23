/* BANRSOR.c */
#include "mex.h"
#include "blas.h"


// compressed column storage data structure
typedef struct sparseCCS
{
    double *AC;  /* numerical values, size nzmax */
    mwIndex *ia; /* row indices */
    mwIndex *jp; /* column pointers (n+1) */
    mwSize m;    /* number of rows */
    mwSize n;    /* number of columns */
} ccs;


double eps = 1.0e-6, ieps = 1.0e-1, one = 1.0, zero = 0.0;
mwSize maxnin = 50;


// How to use
void usage()
{
    mexPrintf("BANRSOR: BA-GMRES method preconditioned by NR-SOR inner iterations\n");
    mexPrintf("         for solving linear least squares problems.\n");
    mexPrintf("  x = BANRSOR(A, b);\n");
    mexPrintf("  [x, relres, iter] = BANRSOR(A, b, tol, maxit);\n\n");
    mexPrintf("  valuable | size | remark \n");
    mexPrintf("  A         m-by-n   coefficient matrix. must be a sparse array\n");
    mexPrintf("  b         m-by-1   right-hand side vector\n");
    mexPrintf("  tol       scalar   tolerance for stopping criterion\n");
    mexPrintf("  maxit     scalar   maximum number of iterations\n");
    mexPrintf("  x         n-by-1   resulting approximate solution\n");
    mexPrintf("  relres   iter-by-1 relative residual history\n");
    mexPrintf("  iter      scalar   number of iterations required for convergence\n");
}


// Automatic parameter tuning for NR-SOR inner iterations
void opNRSOR(const ccs *A, double *rhs, double *Aei, double *x, double *omg, mwIndex *nin)
{
	double *AC = A->AC, d, e, res1, res10, res2 = zero, tmp, tmp1, tmp2, *y, *tmprhs;
	mwIndex i, *ia = A->ia, ii, inc1 = 1, j, *jp = A->jp, k, k1, k2, l;
	mwSize m = A->m, n = A->n;

	// Allocate y
	if ((y = (double *)mxCalloc(n, sizeof(double))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

	// Allocate tmprhs
	if ((tmprhs = (double *)mxMalloc(sizeof(double) * (m))) == NULL) {
		mexErrMsgTxt("Failed to allocate tmprhs");
	}

	// Initilize
	for (i=0; i<m; ++i) tmprhs[i] = rhs[i];

	for (j=0; j<n; ++j) x[j] = zero;

	// Tune the number of inner iterations
	for (k=0; k<maxnin; ++k) {

		for (j=0; j<n; j++) {
			k1 = jp[j];
			k2 = jp[j+1];
			d = zero;
			for (l=k1; l<k2; ++l) d += AC[l]*rhs[ia[l]];
			d *= Aei[j];
			x[j] += d;
			for (l=k1; l<k2; ++l) rhs[ia[l]] -= d*AC[l];
		}

		d = zero;
		e = zero;
		for (j=0; j<n; ++j) {
			tmp1 = fabs(x[j]);
			tmp2 = fabs(x[j] - y[j]);
			if (d < tmp1) d = tmp1;
			if (e < tmp2) e = tmp2;
		}

		if (e<ipes*d || k == 100) {
			nin = k;
			break;

		}

		for (j=0; j<n; j++) y[j] = x[j];

	}

	// Tune the relaxation parameter
	k = 20;
	while (k--) {

		omg = 1.0e-1 * (double)(k); // omg = 1.9, 1.8, ..., 0.1

		for (i=0; i<m; i++) rhs[i] = tmprhs[i];

		for (j=0; j<n; j++) x[j] = zero;

		for (i=1; i<=nin; i++) {
			for (j=0; j<n; j++) {
				k1 = jp[j];
				k2 = jp[j+1];
				d = zero;
				for (l=k1; l<k2; l++) d += AC[l]*rhs[ia[l]];
				d *= omg * Aei[j];
				x[j] += d;
				for (l=k1; l<k2; l++) rhs[ia[l]] -= d*AC[l];
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


// NR-SOR inner iterations
void NRSOR(double *rhs, double *x)
{
	double d;
	mwSize j, k, k1, k2, l;

	for (j=0; j<n; j++) x[j] = zero;

	for (k=1; k<=nin; k++) {
		for (j=0; j<n; j++) {
			k1 = jp[j];
			k2 = jp[j+1];
			d = zero;
			for (l=k1; l<k2; l++) d += AC[l]*rhs[ia[l]];
			d *= Aei[j];
			x[j] += d;
			if (k == nin && j == n-1) return;
			for (l=k1; l<k2; l++) rhs[ia[l]] -= d*AC[l];
		}
	}
}


// Outer iterations: BA-GMRES
void BAGMRES(const ccs *A, double *b, mwIndex maxit, double *iter, double *relres, double *x){

	double *c, *g, *r, *pt, *s, *w, *y, *tmp_x, *Aei, *AC = A->AC, *H, *V;
	double beta, inprod, min_nrmATr, nrmATb, nrmATr, tmp, Tol;
	int ii, k;
	mwSize i, j, k1, k2, l;

	// Allocate V[maxit+1]
	if ((V = mxMalloc(sizeof(double) * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate V");
	} else {	// Allocate V[maxit+1][n]
		for (j=0; j<maxit+1; j++) {
			if ((V[j] = (double *)mxMalloc(sizeof(double) * n)) == NULL) {
				mexErrMsgTxt("Failed to allocate V");
			}
		}
	}

	// Allocate H[maxit]
	if ((H = mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate H");
	} else {	// Allocate H[maxit][maxit+1]
		for (k=0; k<maxit; k++) {
			if ((H[k] = (double *)mxMalloc(sizeof(double) * (k+2))) == NULL) {
				mexErrMsgTxt("Failed to allocate H");
			}
		}
	}

	// Allocate r
	if ((r = (double *)mxMalloc(sizeof(double) * (m))) == NULL) {
		mexErrMsgTxt("Failed to allocate r");
	}

	// Allocate w
	if ((w = (double *)mxMalloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate w");
	}

	// Allocate tmp_x
	if ((tmp_x = (double *)mxMalloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate tmp_x");
	}

	// Allocate Aei
	if ((Aei = (double *)mxMalloc(sizeof(double) * (n))) == NULL) {
		mexErrMsgTxt("Failed to allocate Aei");
	}

	// Allocate g
	if ((g = (double *)mxMalloc(sizeof(double) * (maxit+1))) == NULL) {
		mexErrMsgTxt("Failed to allocate g");
	}

	// Allocate c
	if ((c = (double *)mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate c");
	}

	// Allocate s
	if ((s = (double *)mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate s");
	}

	// Allocate y
	if ((y = (double *)mxMalloc(sizeof(double) * (maxit))) == NULL) {
		mexErrMsgTxt("Failed to allocate y");
	}

	*iter = 0;
	min_nrmATr = 2.0e+52;

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

  	// automatic parameter tuning for NR-SOR inner iterations: w = B r
  	opNRSOR(r, w);   	

  	// beta = norm(Bb)
  	beta = nrm2(w, n);

  	// beta e1
  	g[0] = beta;
 	
  	tmp = one / beta;
  	for (j=0; j<n; j++) { 
  		Aei[j] *= omg;
  		V[0][j] = tmp * w[j]; // Normalize
  	}  	

  	// Main loop
  	for (k=0; k<maxit; k++) {

		for (i=0; i<m; i++) r[i] = zero;
	 	for (j=0; j<n; j++) {
	 		tmp = V[k][j];
	 		k1 = jp[j];
			k2 = jp[j+1];
	 		for (l=k1; l<k2; l++) r[ia[l]] += tmp*AC[l];
	 	}

	 	// NR-SOR inner iterations: w = B r
		NRSOR(r, w);	 

		// Modified Gram-Schmidt orthogonzlization
		for (i=0; i<k+1; i++) {
			tmp = zero;
			for (j=0; j<n; j++) tmp += w[j]*V[i][j];			
			for (j=0; j<n; j++) w[j] -= tmp*V[i][j];
			H[k][i] = tmp;
		}

		// h_{kL1, k}
		tmp = nrm2(w, n);
		H[k][k+1] = tmp;

		// Check breakdown
		if (H[k][k+1] > zero) {
			tmp = one / tmp;
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

		tmp = H[k][k];

		// Compute Givens rotations
		drotg(&tmp, &H[k][k+1], &c[k], &s[k]);

		H[k][k] = one / tmp;		

		// Apply Givens rotations
		g[k+1] = -s[k] * g[k];
		g[k] *= c[k];

		relres[k] = fabs(g[k+1]) / beta; 

		// printf("%.15e\n", relres[k]);

  		if (relres[k] < eps) {

			// Derivation of the approximate solution x_k
			// Backward substitution		
			y[k] = g[k] * H[k][k];
			ii = k;
			while (ii--) {	
				tmp = zero;
				for (l=ii+1; l<k; l++) tmp += H[l][ii]*y[l];
				y[ii] = (g[ii] - tmp) * H[ii][ii];
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
		 		for (l=jp[j]; l<jp[j+1]; l++) r[ia[l]] += tmp*AC[l];
		 	}

		 	// r = b - Ax
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

		 	if (nrmATr < min_nrmATr) {
	 			for (j=0; j<n; j++) x[j] = tmp_x[j];
	 			min_nrmATr = nrmATr;
	 			*iter = k+1;
	 		}

			// printf("%d, %.15e\n", k, nrm2(w, n)/nrmATb);

		 	// Convergence check
		  	if (nrmATr < Tol) {

				*iter = k+1;

				mxFree(y);
				mxFree(s);
				mxFree(c);
				mxFree(g);
				mxFree(tmp_x);				
				mxFree(Aei);
				mxFree(w);				
				mxFree(r);		
				

				i = maxit;
				while (i--) mxFree(H[i]);
				mxFree(H);	

				j = maxit + 1;
		  		while (j--) mxFree(V[j]);
				mxFree(V);	


  				// printf("Required number of iterations: %d\n", (int)(*iter));
  				printf("Successfully converged.\n");  				

				return;

			}
		}
	}

	mexPrintf("Failed to converge.\n");

	if (*iter == 0) {

		k = k-1;

		// Derivation of the approximate solution x_k
		// Backward substitution		
		y[k] = g[k] * H[k][k];
		ii = k;
		while (ii--) {		
			tmp = zero;
			for (l=ii+1; l<k; l++) tmp += H[l][ii]*y[l];
			y[ii] = (g[ii] - tmp) * H[ii][ii];
		}

		// x = V y
		for (j=0; j<n; j++) x[j] = zero;
		for (l=0; l<k; l++) {
			for (j=0; j<n; j++) x[j] += V[l][j]*y[l];
		}

		*iter = k;
	}

	// Derivation of the approximate solution x_k
	// Backward substitution
	y[k] = g[k] * H[k][k];
	for (ii=k-1; ii>-1; ii--) {
		tmp = zero;
		for (l=ii+1; l<k; l++) tmp += H[l][ii]*y[l];
		y[ii] = (g[ii] - tmp) * H[ii][ii];
	}

	// x = V y
	for (j=0; j<n; j++) x[j] = zero;
	for (l=0; l<k+1; l++) {
		for (j=0; j<n; j++) x[j] += V[l][j]*y[l];
	}

	*iter = k;

	mxFree(y);
	mxFree(s);
	mxFree(c);
	mxFree(g);
	mxFree(tmp_x);
	mxFree(Aei);
	mxFree(w);
	mxFree(r);

	if (nrmATr < min_nrmATr) {
		for (j=0; j<n; j++) x[j] = tmp_x[j];
		min_nrmATr = nrmATr;
		*iter = k+1;
	}

	i = maxit;
	while (i--) mxFree(H[i]);
	mxFree(H);

	j = maxit + 1;
	while (j--) mxFree(V[j]);
	mxFree(V);

}


/* form sparse matrix data structure */
void form_ccs(ccs *A, const mxArray *Amat)
{
    A->jp = (mwIndex *)mxGetJc(Amat);
    A->ia = (mwIndex *)mxGetIr(Amat);
    A->m = mxGetM(Amat);
    A->n = mxGetN(Amat);
    A->AC = mxGetPr(Amat);
    return;
}


// Main
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ccs A;
    double *b, *iter, *relres, *x;
    mwIndex maxit;
    mwSize m, n;

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

    // Check the number of input arguments
    if (nrhs < 1) {
    	usage();
        mexErrMsgTxt("Please input A.");
    } else if (nrhs < 2) {
		usage();
        mexErrMsgTxt("Please input b.");
    }

	// Check the 1st argument
    if (!mxIsSparse(prhs[0]))  {
    	usage();
        mexErrMsgTxt("1st input argument must be a sparse array.");
    } else if (mxIsComplex(prhs[0])) {
    	usage();
    	mexErrMsgTxt("1st input argument must be a real array.");
    }

    form_ccs(&A, prhs[0]);
    m = A.m;
    n = A.n;

	// Check the 2nd argument
    if (mxGetM(prhs[1]) != m) {
    	usage();
    	mexErrMsgTxt("The length of b is not the numer of rows of A.");
    }

    b = mxGetPr(prhs[1]);

	// Check the 3rd argument
    // Set eps
    if (nrhs < 3) {
        mexPrintf("Default: stopping criterion is set to 1e-6.\n");
    } else {
    	if (mxIsComplex(prhs[2]) || mxGetM(prhs[2])*mxGetN(prhs[2])!=1) {
    		usage();
    		mexErrMsgTxt("3nd argument must be a scalar");
    	} else {
    		eps = *(double *)mxGetPr(prhs[2]);
    		if (eps<zero || eps>=one) {
    			usage();
    			mexErrMsgTxt("3nd argument should be positive and less than or equal to 1.");
    		}
    	}
    }

	// Check the 4th argument
	// Set maxit
    if (nrhs < 4) {
    	maxit = n;
    	// mexPrintf("Default: max number of iterations is set to the number of columns.\n");
   	} else if (mxIsComplex(prhs[3]) || mxGetM(prhs[3])*mxGetN(prhs[3])!=1) {
    	usage();
    	mexErrMsgTxt("4th argument must be a scalar");
    } else {
   		maxit = (mwIndex)*mxGetPr(prhs[3]);
   		if (maxit < 1) {
   			usage();
   			mexErrMsgTxt("4th argument must be a positive scalar");
   		}
	}

	plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(maxit, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);

    x = mxGetPr(plhs[0]);
    relres = mxGetPr(plhs[1]);
    iter = mxGetPr(plhs[2]);

	// BA-GMRES method
    BAGMRES(&A, b, maxit, iter, relres, x);

    // Reshape relres
    mxSetPr(plhs[1], relres);
    mxSetM(plhs[1], (mwSize)iter[0]);
    mxSetN(plhs[1], 1);

}
