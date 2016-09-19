#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
// approximation can be improved by using LS fit on all objects of intresset!
// 1/x, log(x) * P(x)

// constant needed in
double const P001[] = {0.038607832450766433};
double const P008[] = { -0.042049020943654196, 0.038607832450766433, 0.};
double const C008[] ={ 1., 0.5, 1./12.};
double const P05[] = {-0.00066576282593196418
				 ,-0.327504940907552
				 ,-0.041221169337540275
				 ,-0.028612284551326601};
double const C05[] ={-0.00012098484397616241,
				  0.49314379795952901 ,
				   -0.044833752786686006,
				   0.025404785625869428};
double const P1[] = {-0.029684612767024068,
		 	 	 	 -0.53184189045070873 ,
					 0.014630382439326196,
					 0.14880335102535383};
double const C1[] ={-0.0068391485280378979,
					 0.37369983569078935 ,
					 -0.29611650243699073,
					 -0.037504774663373729};

double const P105[] = {1.25516231,
					   0.45637828,
					   -0.10162785,
					   0.03170288,
					   -0.00546217};

double const P2[] = {1.25401547,
					 0.46295212,
					 -0.11584736,
					 0.04546309,
					 -0.01048961};
double const P5[] = {1.25335066,
					 0.46922605,
					 -0.13996358,
					 0.09281368,
					 -0.05809823,
					 0.01963612};
double const P40[] = {1.25331423,
					  0.46998496,
					  -0.14660108,
					  0.12383611,
					  -0.1364926 ,
					  0.10526382};

double const P200[] = {1.25331414,
					   0.46999276,
					   -0.14686537,
					   0.12792201,
					   -0.15816086};
double const PINF[] = {1.,
					   0.375,
					   -0.1171875};

double limitK1_001(const double );
double limitK1_008(const double );
double limitK1_05(const double);
double limitK1_1(const double);
double limitK_upper(const double , const double[] , const int);
double polyEval(const double x, const double P[], const int n)
{
	double res = P[n - 1];

	for(int i = n -2; i >= 0 ; --i)
	{
		res *= x;
		res += P[i];
	}
	return res;
}
void bessel1(const double* x, double *res, const int n)
{
	/*
	 * Approximation of besselK(1, x) with absolute and relative error less then 0.5*1e-7
	 *
	 * x   - (n x 1) value
	 * res - (n x 1) result vector
	 */

	for(int i = 0; i < n; i++)
	{
		// return error if neg!
		// return inf if 0.
		if(x[i] < 0)
			res[i] = NAN;
		else if(x[i] == 0)
			res[i] = INFINITY;
		else if(x[i] < 0.001)
			res[i] = limitK1_001(x[i]);
		else if(x[i] < 0.08)
			res[i] = limitK1_008(x[i]);
		else if(x[i] < 0.5)
			res[i] = limitK1_05(x[i]);
		else if(x[i] < 1)
			res[i] = limitK1_1(x[i]);
		else if(x[i] < 1.5)
			res[i] = limitK_upper(x[i], P105, 5);
		else if(x[i] < 2)
			res[i] = limitK_upper(x[i], P2, 5);
		else if(x[i] < 5)
			res[i] = limitK_upper(x[i], P5, 6);
		else if(x[i] < 40)
			res[i] = limitK_upper(x[i], P40, 6);
		else if(x[i] < 200)
			res[i] = limitK_upper(x[i], P200, 5);
		else if(x[i] < 1000)
			res[i] = limitK_upper(x[i], PINF, 3);
		else
			res[i] = 0.;

	}

}
void bessel1order(const double* x, double *res, const int n)
{
	/*
	 * Approximation of besselK(1, x) with absolute and relative error less then 0.5*1e-7
	 *
	 * x   - (n x 1) value (orded)
	 * res - (n x 1) result vector
	 */
	int i = 0;
	while( i < n && x[i] < 0 ){
		res[i] = NAN; i++;}
	while( i < n && x[i] < 0.001){
		res[i] = limitK1_001(x[i]);i++;}
	while( i < n && x[i] < 0.08){
		res[i] = limitK1_008(x[i]);i++;}
	while( i < n && x[i] < 0.5){
		res[i] = limitK1_05(x[i]);i++;}
	while( i < n && x[i] < 1){
		res[i] = limitK1_1(x[i]);i++;}
	while( i < n && x[i] < 1.5){
		res[i] = limitK_upper(x[i], P105, 5);i++;}
	while( i < n && x[i] < 2){
		res[i] = limitK_upper(x[i], P2, 5);i++;}
	while( i < n && x[i] < 5){
		res[i] = limitK_upper(x[i], P5, 6);i++;}
	while( i < n && x[i] < 40){
		res[i] = limitK_upper(x[i], P40, 6);i++;}
	while( i < n && x[i] < 200){
		res[i] = limitK_upper(x[i], P200, 5);i++;}
	while( i < n && x[i] < 1000)
		res[i] = limitK_upper(x[i], PINF, 3);
	for(; i < n ; i++)
		res[i] = 0;


}

double limitK_upper(const double x, const double P[], const int Porder)
{
	/*
				Bessel approximation between [1.5, 2)
				using Hankel times poly
				|error| < 0.5*1e-8
		*/
	double res = 1.;
	res *= exp(-x) * polyEval(1. / x, P, Porder);
	res /= sqrt(x);
	return(res);
}



double limitK1_1(const double x)
{
	/*
			Bessel approximation between [0.5, 1)
			using Abramvoich 9.6.11 and poly approx
			|error| < 1e-10
	*/

	double res;
	res = 1./x;
	res += polyEval(x, P1, 4);
	res += log(x) * polyEval(x, C1, 4);
	return(res);
}
double limitK1_05(const double x)
{
	/*
			Bessel approximation between [0.08, 0.5)
			using Abramvoich 9.6.11 and poly approx
			|error| < 1e-10
	*/

	double res;
	res = 1./x;
	res += polyEval(x, P05, 4);
	res += log(x) * polyEval(x, C05, 4);
	return(res);
}

double limitK1_008(const double x)
{
	/*
			Bessel approximation between [0.01, 0.08)
			using Abramvoich 9.6.11
			|error| < 1e-10
	*/

	double res;
	res = 1./x;
	res += x * (P008[1]  + P008[0] * (x * x));
	double x05 = 0.5*x;
	double x05_2 = x05 * x05;
	double  prod_ = polyEval(x05_2, C008, 3);
	prod_ *= x05 * log(x05);
	return (res+ prod_);
}

double limitK1_001(const double x)
{
	/*
		Bessel approximation between (0., 0.01)
		using Abramvoich 9.6.11
		|error| < 1e-9
	*/
	double res;
	res = 1./x;
	res += P001[0] * x;
	double x05 = 0.5*x;
	double x05_2 = x05 * x05;
	double prod_ = C008[1];
	prod_ *= x05_2;
	prod_ += 1.;
	prod_ *= x05_2;
	prod_ += 1;
	prod_ *= x05 * log(x05);
	return (res+ prod_);
}

double const K0P8[] = {
					0.11593151620556186,
					0.27898398957826176,
					0.025267521444959137,
					0.00084126262966468526
					};
double const K0C8[] ={-0.99999999997659728,
						-0.24999957531125322,
						 -0.015608710836632057,
						-0.0003869991316737959};



double const K0_Pupper[]= {1.1600249425076035558e+02,
 			2.3444738764199315021e+03,
  			1.8321525870183537725e+04,
   			7.1557062783764037541e+04,
    		1.5097646353289914539e+05,
     		1.7398867902565686251e+05,
     		1.0577068948034021957e+05,
     		3.1075408980684392399e+04,
     		3.6832589957340267940e+03,
     		1.1394980557384778174e+02};
double const K0_Cupper[] = {9.2556599177304839811e+01,
			  1.8821890840982713696e+03,
			  1.4847228371802360957e+04,
			  5.8824616785857027752e+04,
			  1.2689839587977598727e+05,
			  1.5144644673520157801e+05,
			  9.7418829762268075784e+04,
			  3.1474655750295278825e+04,
			  4.4329628889746408858e+03,
			  2.0013443064949242491e+02,
			  1.0};
double limitK0_08(const double x)
{
	/*
		Bessel approximation between (0., 0.8)
		using Abramvoich 9.6.13
		|error| < 1e-9
	*/
	double res;
	double x2 = x * x;
	res = polyEval(x2, K0P8, 4);
	res += log(x) * polyEval(x2, K0C8, 4);
	return(res);
}
double limitK0_upper(const double x)
{
	/*
		Bessel approximation between (0.8, inf)
		using Boost approx
		|error| < 1e-9
	*/
	double res;
	double y = 1 / x;
	res =  polyEval(y, K0_Pupper, 10);
	res /= polyEval(y, K0_Cupper, 11);
	res /= exp(x)* sqrt(x);
	return(res);
}

void bessel0(const double* x, double *res, const int n)
{
	/*
	 * Approximation of besselK(0, x) with absolute and relative error less then 0.5*1e-7
	 *
	 * x   - (n x 1) value
	 * res - (n x 1) result vector
	 */

	for(int i = 0; i < n; i++)
	{
		// return error if neg!
		// return inf if 0.
		if(x[i] < 0)
			res[i] = NAN;
		else if(x[i] == 0)
			res[i] = INFINITY;
		else if(x[i] < 0.8)
			res[i] = limitK0_08(x[i]);
		else if (x[i] < 1000)
			res[i] = limitK0_upper(x[i]);
		else
			res[i] = 0;

	}

}

#ifdef __cplusplus
}
#endif
