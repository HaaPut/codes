#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>

#define CHOL 0
#define LU 1
#define LU_PP 2
#define LO_UNIT 3
#define LO 4
#define UP 5
#define ITER 6

#define DELTA 0.000000001
//#define DELTA 0.1
#define MAX_ITER_COUNT 10000

#define NEGATIVE_ROOT -1
#define DIVISION_BY_ZERO -2

//typedef float Numtype;
typedef long double Numtype;

using namespace std;

void print(Numtype**A, int n);
void printVec(Numtype *a, int n);
void printRoundedVec(Numtype *a, int n);
void known_A(Numtype** A);
void known_b(Numtype* b);
void generateA0(Numtype** A, int n);
void generateHilbertMatrix(Numtype **h, int n);

void chol(Numtype** A, Numtype** LL, int n);

void lu(Numtype** A, int n, Numtype* growth_factor = NULL);

void partialPivotingLU(Numtype** A, int* P, int n, Numtype* growth_factor = NULL);
void swapRows(Numtype** A, int j, int k);
int maxRow(Numtype** A, int col, int n);

void solve(Numtype** A, Numtype* x, Numtype* b, int n, int mode = LU, Numtype* growth_factor = NULL);
void unitForwardSubstitution(Numtype** L, Numtype *x, Numtype* b, int n);
void forwardSubstitution(Numtype** L, Numtype *x, Numtype* b, int n);
void backSubstitution(Numtype** U, Numtype* x, Numtype* b, int n);
void productLCol(Numtype** lu, Numtype* b, int n);
void productCol(Numtype** A, Numtype* x, Numtype* b, int n);

int iterativeRefinement(Numtype** A, Numtype* x, Numtype* b, int n, Numtype t = 0.001);

Numtype** productLU(Numtype** lu, int n);
Numtype** productChol(Numtype** LL,int n);
Numtype mse(Numtype* x, Numtype* y, int n);
Numtype mseMat(Numtype** A, Numtype** B, int n, int m = -42);
Numtype computeGrowthFactor(Numtype** A,Numtype** U, int n);
Numtype normFrobenius(Numtype **A, int n, int m = -42);
Numtype normL2(Numtype* x, int n);
void subtractVec(Numtype* x, Numtype* y, Numtype* z, int n);
void test_lu(Numtype** A, int n);

void assignment4b();
void assignment3();
void assignment4c();

int main(){
	/*
	//----------initializations----------------
	int n = 30;
	int* P = new int[n];//need only n-1.. 
	//change to n-1 after updating code in partialPivotingLU
	Numtype** A = new Numtype*[n];
	srand (time(NULL));
	for(int i = 0; i < n; i++){
		A[i] = new Numtype[n];
		for(int j = 0; j < n; j++){
			A[i][j] = rand()%100 + 1.0;
		}
	}
    
    Numtype* b = new Numtype[n];
    Numtype* x = new Numtype[n];
    */

    //---------------------------------------------

    /*
    known_A(A);
    printf("----------Matrix A----------\n");
    print(A,n);
	*/

	/*
    lu(A,n);
    printf("---------LU factors-----------\n");
    print(A,n);
    
	//printf("-------Product of L and U---------\n");
    //print(productLU(A,n),n);
	printf("\n\n");
 	*/

	/* 
    printf("----LU with partial pivoting----\n");
    known_A(A);
    partialPivotingLU(A,P,n);
    printf("---------LU_PP factors-----------\n");
    print(A,n);
	
    printf("-------Permutation Matrix---------\n");
    for(int i = 0 ; i < n-1;i++)printf("%d ",P[i] + 1);printf("\n");
    printf("-------Product of L and U---------\n");
    print(productLU(A,n),n);
	printf("\n\n");
	*/

	/*
	known_A(A);
	known_b(b);
	solve(A, x, b, n, LU);
	printf("--------Solution of Problem---------\n");
	printRoundedVec(x,n);
	*/
	assignment3();
	//assignment4b();
	//assignment4c();
    return 0;
}

void assignment4c(){
	int n = 30;
	//scanf("%d",&n);

	Numtype** A = new Numtype*[n];
	for(int i = 0; i < n; i++){
		A[i] = new Numtype[n];
	}
    generateA0(A,n);
	for(int i = 0; i < n; i++){
		A[i][i] += DELTA;
	}


    Numtype* b = new Numtype[n];
    for(int i = 0; i < n;i++) b[i] = n - (i*2);

    Numtype* x = new Numtype[n];
    Numtype* e = new Numtype[n];
    for(int i = 0; i < n; i++)e[i] = 1;

    Numtype tolerance = 0.0000000000000000001;
	
	solve(A,x,b,n,LU);
	for(int i = 0; i < n;i++) b[i] = n - (i*2);
	printVec(x,n);
    int iterCount = iterativeRefinement(A,x,b,n,tolerance);
	Numtype relative_error = sqrt(mse(e,x,n)) / sqrt(n);
	cout<<"iteration count = "<<iterCount<<endl;
	cout<<"relative error = "<<relative_error<<endl;
	
	for(int i = 0; i < n; i++) delete A[i];
	delete A;
	delete b;
	delete e;
	delete x;
	return;
}

void assignment3(){
	int siz[6] = {2,4,6,8,10,12};
	for(int s = 0; s < 4; s++){
		int n = siz[s];
		Numtype **A = new Numtype*[n];
		Numtype **L = new Numtype*[n];
		for(int i = 0 ; i < n;i++){
			A[i] = new Numtype[n];
			L[i] = new Numtype[n];
		}  
		generateHilbertMatrix(A,n);
		chol(A,L,n);
		//print(L,n);
		

		Numtype* x = new Numtype[n];
    	Numtype* e = new Numtype[n];
    	for(int i = 0; i < n; i++)e[i] = 1;
    	
    	Numtype* b = new Numtype[n];
		//define b = Ae
		productCol(A,e,b,n);
		//printVec(b,n);
    	//Ax=b => LL'x = b
    	//solve (LOwertriangular system) Ly = b
    	solve(L,x,b,n,LO);
    	
    	//solve (UPper triangular system)L'x = y
    	//use value of y(in x) from previous
    	//store final solution in b(overwrite)
    	solve(L,b,x,n,UP);
    	

    	cout<<"-------n = "<<s<<"------------"<<endl;
    	Numtype relativeError = sqrt(mse(b,e,n)) / sqrt(n);
		cout<<"relative error = "<<relativeError<<endl;

		//compute product of A and computed solution xc(in b) and store in e
		productCol(A,b,e,n);
		Numtype aF = normFrobenius(A,n);
		Numtype residualError=sqrt(mse(b,e,n))/(aF*normL2(b,n));
		cout<<"residual error = "<< residualError<<endl;

		Numtype ** LLt = productChol(L,n);
		Numtype relativeMatrixResidual = (sqrt(mseMat(A,LLt,n))) / (aF);
		cout<<"Relative Matrix Residual = "<< relativeMatrixResidual<<endl;
		cout<<endl;
    	//free up the memory
    	for(int i = 0; i < n; i++){
    		delete A[i];
    		delete L[i];
    	}
		delete A,L,b,x,e;
	}
}

void assignment4b(){
	int n = 30;
	//scanf("%d",&n);

	Numtype** A = new Numtype*[n];
	for(int i = 0; i < n; i++){
		A[i] = new Numtype[n];
	}
    generateA0(A,n);
	for(int i = 0; i < n; i++){
		A[i][i] += DELTA;
	}


    Numtype* b = new Numtype[n];
    for(int i = 0; i < n;i++) b[i] = n - (i*2);

    Numtype* x = new Numtype[n];
    Numtype* e = new Numtype[n];
    for(int i = 0; i < n; i++)e[i] = 1;

    Numtype growth_factor;
    solve(A, x, b, n, LU, &growth_factor);
	//printVec(x,n);
	Numtype relative_error = sqrt(mse(e,x,n)) / sqrt(n);
	cout<<"relative error = "<<relative_error<<endl;
	cout<<"growth factor = "<<growth_factor<<endl;
	for(int i = 0; i < n; i++) delete A[i];
	delete A;
	delete b;
	delete e;
	delete x;
	return;
}

void productCol(Numtype** A, Numtype* x, Numtype* b, int n){
	for(int i = 0; i<n; i++){
		b[i] = 0;
		for(int j = 0 ; j < n; j++){
			b[i] += x[j]*A[i][j];
		}
	}
}

Numtype normL2(Numtype* x, int n){
	Numtype norm = 0;
	for(int i = 0 ; i < n; i++){
		norm += x[i]*x[i];
	}
	return sqrt(norm);
}

Numtype normFrobenius(Numtype **A, int n, int m){
	if(m == -42)m = n;
	Numtype norm = 0;
	for(int i = 0;i < n; i++){
		for(int j = 0; j < m;j++){
			norm += A[i][j]*A[i][j];
		}
	}
	return sqrt(norm);
}

Numtype mse(Numtype* x, Numtype* y, int n){
	Numtype error = 0;
	for(int i = 0; i < n; i++){
		error += ((x[i] - y[i])*(x[i] - y[i]));
	}
	return error;
}

Numtype mseMat(Numtype** A, Numtype** B, int n, int m){
	if(m == -42) m = n;
	Numtype error = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			error += ((A[i][j] - B[i][j])*(A[i][j] - B[i][j]));
		}
	}
	return error;
}

Numtype computeGrowthFactor(Numtype** A,Numtype** U, int n){
	Numtype ainf = 1, uinf = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(j>=i){
				//stuff for u
				if(uinf < abs(U[i][j])) uinf = abs(U[i][j]);
			}
			if(ainf < abs(A[i][j])) ainf = abs(A[i][j]);
		}
	}
	return uinf/ainf;
}

void solve(Numtype** A, Numtype* x, Numtype* b, int n, int mode, Numtype* growth_factor){
	switch(mode){
		case LU_PP:{
			//Compute LU factorization of A with partial pivoting
			//Stored back into A
			int* P = new int[n];
			partialPivotingLU(A,P,n,growth_factor);
			
			//permute b
			for(int i = 0; i < n-1;i++){
				Numtype temp = b[i];
				b[i] = b[P[i]];
				b[P[i]] = temp;
			}
			/*------Solve like normal LU-------*/
			//solve Ly = b (forward substitution)
			//here L -> A,(matrix)
			// y -> x, (variable)
			// b -> b, (data vector)
			solve(A,x,b,n,LO_UNIT);
 			
 			//solve Ux = y (back substitution)
 			//here U -> A,(matrix)
			// x -> b (variable - has original b which is useless now)
 			// y -> x(data vector - sol. of previous)
 			solve(A,b,x,n,UP);
 			for(int i = 0; i < n; i++)x[i] = b[i];//return result in x
			break;
		}
		case LU:{
			//Compute LU factorization of A
			//Stored back into A
			lu(A,n,growth_factor);

			//solve Ly = b (forward substitution)
			//here L -> A,(matrix)
			// y -> x, (variable)
			// b -> b, (data vector)
			solve(A,x,b,n,LO_UNIT);
 			
 			//solve Ux = y (back substitution)
 			//here U -> A,(matrix)
			// x -> b (variable - has original b which is useless now)
 			// y -> x(data vector - sol. of previous)
 			solve(A,b,x,n,UP);
 			for(int i = 0; i < n; i++)x[i] = b[i];//return result in x
 			break;
		}
		case LO_UNIT:{
			unitForwardSubstitution(A,x,b,n);
			break;
		}
		case UP:{
			backSubstitution(A,x,b,n);
			break;
		}
		case LO:{
			forwardSubstitution(A,x,b,n);
			break;
		}
		default:{
		 	printf("the mode is unknown!.. \n Using LU instead\n");
		 	solve(A,x,b,n);
		 	break;
		}
	}
}

int iterativeRefinement(Numtype** lu, Numtype* x, Numtype* b, int n, Numtype t){
	Numtype *r = new Numtype[n];
	Numtype *d = new Numtype[n];
	Numtype error = 42;
	int iterCount = 0;
	cout<< "tolerance:"<<t<<endl;
	cout<<"error:"<<error<<endl;
	cout<<"iterCount:"<<iterCount<<endl;
	while((error > t) && (iterCount < MAX_ITER_COUNT)){
		iterCount++;
		//compute r = b - Ax
		//printVec(x,n);
		productCol(productLU(lu,n),x,r,n);//store Ax in r
		//cout<<"computed b:"<<endl;
		//printVec(r,n);
		subtractVec(b,r,r,n);//r = b - (old r) Ax
		//cout<<"residual vector r = :"<<endl;
		//printVec(r,n);
		solve(lu,d,r,n,LO_UNIT);
 		solve(lu,r,d,n,UP);
 		//cout<<"correction vector d = :"<<endl;
 		//for(int i = 0; i < n; i++)d[i] = r[i];//return result in x
		//solve(A,d,r,n,LU);
		//cout<<"----------d---------------"<<endl;
		//printVec(r,n);
		for(int i = 0; i <n; i++){
			x[i] += r[i];
		}
		error = normL2(r,n)/normL2(x,n);
		cout<<"iter number: "<<iterCount<<" current error = "<<error<<endl;
	}
	return iterCount;
}

void subtractVec(Numtype* x, Numtype* y, Numtype* z, int n){
	//x - y = z
	for(int i = 0; i < n;i++){
		z[i] = x[i] - y[i];
	}
}

void backSubstitution(Numtype** U, Numtype* x, Numtype* b, int n){
	if(U[n-1][n-1] == 0){
		printf("U has a zero on diagonal..\n");
		return;
	}
	x[n-1] = b[n-1]/U[n-1][n-1];
	for(int i = n-2; i >=0; i--){
		Numtype val = 0;
		for(int j = i+1; j < n; j++){
			val += U[i][j]*x[j];
		}
		if(U[i][i] == 0){
			printf("U has a zero on diagonal..\n");
			return;
		}
		x[i] = (b[i] - val)/U[i][i];
	}
	
}

void unitForwardSubstitution(Numtype** L, Numtype *x, Numtype* b, int n){
	x[0] = b[0];
	for(int i = 1; i < n; i++){
		Numtype val = 0;
		for(int j = 0 ;j < i; j++){
			val += L[i][j]*x[j];
		}
		x[i] = b[i] - val;
	}
	//for(int i = 0; i < n; i ++)printf("%f\n",x[i]);
}

void forwardSubstitution(Numtype** L, Numtype *x, Numtype* b, int n){
	//code not tested
	if(L[n-1][n-1] == 0){
		printf("L has a zero on diagonal..\n");
		return;
	}
	x[0] = b[0]/L[0][0];
	for(int i = 1; i < n; i++){
		Numtype val = 0;
		for(int j = 0 ;j < i; j++){
			val += L[i][j]*x[j];
		}
		if(L[i][i] == 0){
			printf("L has a zero on diagonal..\n");
			return;
		}
		x[i] = (b[i] - val)/L[i][i];
	}
	//for(int i = 0; i < n; i ++)printf("%f\n",x[i]);
}

void productLCol(Numtype** lu, Numtype* b, int n){
	//useless code.. but technique may be usefull...
	//inplace product with lower triangular..
	//correctness not checked 
	for(int i = n-1; i >= 0; i--){
		Numtype sum = 0;
		for(int j = 0; j <= i; j++){
			if(j == i){
				sum += 1*b[j];
			}else{
				sum += lu[i][j]*b[j];
			}
			//printf("col --> %d sum --> %f\n",j,sum);
		}
		b[i] = sum;
		//printf("%f ",sum);
	}
	printf("\n");
	//for(int i = 0; i < n; i++)printf("%f\n",b[i]);
}

Numtype** productChol(Numtype** ll, int n){
	Numtype** product = new Numtype*[n];
	for(int i = 0; i < n; i++){
		product[i] = new Numtype[n];
	}
	for(int i = 0 ; i < n; i ++){
		for(int j = 0 ; j <= i; j++){//rest by symmetry
			product[i][j] = 0;
			for(int k = 0 ; k <= j; k++){//rest is zeros in L or L'
				product[i][j] += ll[i][k]*ll[k][j];
			}
		}
	}
	//use symmetry property
	for(int i = 0 ; i < n; i++){
		for(int j = i+1; j < n; j++){
			product[i][j] = product[j][i];
		}
	}
	return product;
}

Numtype** productLU(Numtype** lu, int n){
	Numtype** product = new Numtype*[n];
	for(int i = 0; i < n; i++){
		product[i] = new Numtype[n];
	}
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n;j++){
			product[i][j] = 0;
			int lim = i<j?i:j;//rest is zeros
			for(int k = 0;k <= lim;k++){
				if(k == i){
					product[i][j] += 1*lu[k][j];
				}else{
					product[i][j] += lu[i][k]*lu[k][j];
				}
			}
		}
	}
	return product;
}

void partialPivotingLU(Numtype** A, int* P, int n, Numtype* growth_factor){
	//too tierd for an efficient computation of growth_factor
	//should be possible with this n^2 stuff..
	Numtype ainf = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(ainf < abs(A[i][j])) ainf = abs(A[i][j]);
		}
	}
	for(int j = 0 ; j < n; j++){
		//Insert pivoting stuff here!
		int maxRowIdx = maxRow(A,j,n);
		P[j] = maxRowIdx;//last idx is useless
		swapRows(A,j,maxRowIdx);
		for(int i = j+1; i < n; i++){
			Numtype l_ij = A[i][j]/A[j][j];
			A[i][j] = l_ij;//0 if you don't want L factor
			for(int k = j+1; k< n; k++){
				A[i][k] -= A[j][k]*l_ij; // update row 
			}
		}
	}
	Numtype uinf = 0;
	for(int i = 0; i < n; i++){
		for(int j = i; j < n; j++){
			if(uinf < abs(A[i][j])) uinf = abs(A[i][j]);
		}
	}
	if(growth_factor) *growth_factor = uinf/ainf;
}

void swapRows(Numtype** A, int j, int k){
	//swap 2 rows of A.. this swaps L as well
	//which is stored in lower part of A
	Numtype* temp = A[j];
	A[j] = A[k];
	A[k] = temp;
}

int maxRow(Numtype** A, int col, int n){
	int maxRow = abs(A[col][col]), idx = col;
	for(int i = col+1; i < n; i++){
		if(abs(A[i][col]) > maxRow){
			maxRow = A[i][col];
			idx = i;
		}
	}
	return idx;
}

void lu(Numtype** A, int n, Numtype* growth_factor){
	Numtype ainf = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(ainf < abs(A[i][j])) ainf = abs(A[i][j]);
		}
	}
	for(int j = 0 ; j < n; j++){
		for(int i = j+1; i < n; i++){
			Numtype l_ij = A[i][j]/A[j][j];
			A[i][j] = l_ij;//0 if you don't want L factor
			for(int k = j+1; k< n; k++){
				A[i][k] -= A[j][k]*l_ij; // update row 
			}
		}
	}
	Numtype uinf = 0;
	for(int i = 0; i < n; i++){
		for(int j = i; j < n; j++){
			if(uinf < abs(A[i][j])) uinf = abs(A[i][j]);
		}
	}
	if(growth_factor) *growth_factor = (uinf/ainf);
}

void chol(Numtype** A, Numtype** L, int n){
	for(int j = 0 ; j < n; j++){
		Numtype sumRow = 0;
		for(int k = 0; k < j; k++){
			sumRow += L[j][k]*L[j][k]; 
		}
		L[j][j] = A[j][j] - sumRow;
		try{
			if(L[j][j] < 0) throw NEGATIVE_ROOT;
			else if(L[j][j] == 0) throw DIVISION_BY_ZERO;

		}catch(int err){
			if(err == NEGATIVE_ROOT){
				printf("Negative Value in square root\n");
			}
			else if(err == DIVISION_BY_ZERO){
				printf("Division by zero error\n");
			}
		}
        L[j][j] = sqrt(L[j][j]);
		for(int i = j+1; i < n; i++){
			Numtype sumRow = 0;
			for(int k = 0; k < j; k++){
				sumRow += L[i][k]*L[j][k];
			}
			L[i][j] = (A[i][j] - sumRow)/L[j][j];
			L[j][i] = L[i][j];//store LL' in single matrix.
		}

	}
}

void test_lu(Numtype** A, int n){
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			Numtype a_ij = 0;
			for(int k = 0; k < i-1; k++){
				for(int l = 0; l < j; l++){
					//a_ij = A[i][k]*......;
					//finish this maybe later..
				}
			}
		}
	}
}

void known_A(Numtype** A){
	float knownA[][3] = {{3, -1, 2}, {9, -1, 13}, {6, -12, -26}};
	//float knownb = {5, 28, 50};
	//float knownx = {2, 3, 1}; 
	for(int i = 0; i < 3; i++){
		for(int j = 0 ; j < 3; j++){
			A[i][j] = knownA[i][j];
		}
	}
}

void known_b(Numtype* b){
	float knownb[3] = {5, 28, -50};
	for(int i = 0; i < 3; i++) b[i] = knownb[i];
}

void generateA0(Numtype** A, int n){
	for(int i = 0; i< n; i++){
		for(int j = 0; j < n;j++){
			if(j >= i) A[i][j] = 1;
			else A[i][j] = -1;
		}
	}
}

void generateHilbertMatrix(Numtype **h, int n){
	for(int i = 0; i < n; i++){
		for(int j = 0 ; j < n; j++){
			h[i][j] = 1./(i + j + 1);
		}
	}
}

void print(Numtype**A, int n){
	cout<<fixed;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n;j++){
		    cout<<setprecision(5)<<A[i][j]<<" ";
		    //printf("%3f ",A[i][j]);
		}
		printf("\n");
	}
	cout<<endl;
}

void printRoundedVec(Numtype *a, int n){
	//for(int i = 0; i < n; i ++)printf("%f\n",x[i]);
	for(int i = 0; i < n; i ++)cout<<(int)round(a[i])<<endl;
}

void printVec(Numtype *a, int n){
	for(int i = 0; i < n; i ++)cout<<a[i]<<endl;
}
