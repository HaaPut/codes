import numpy as np
import numpy.linalg as lin
import sys


def test_column_addition():
	#failure case
	#A = np.array([[1.0,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,0]])
	#a = np.array([1,2,3,4,5])
	A = np.random.rand(5,3)
	a = np.random.rand(5)
	loc = 2
	Q,R = lin.qr(A,mode='complete')
	
	Q1,R1 = add_column_qr(Q,R,a,2)
	print 'Updated Q:\n ', Q1
	print 'Updated R:\n ', R1
	
	A = np.insert(A,2,a,axis=1)
	
	QQ,RR = np.linalg.qr(A,mode = 'complete')
	#print 'numpy Q :\n', QQ
	print 'numpy R :\n', RR
	print 'Is the result accurate: ',np.allclose(abs(R1),abs(RR)) & np.allclose(abs(Q1), abs(QQ))
	
def add_column_qr(Q,R,a,col):
	m,n = Q.shape
	Q = Q.T
	a = np.dot(Q,a)
	R = np.insert(R,col,a,axis = 1)
	A = R.copy()
	for row in range(m-2, col-1,-1):
		if A[row+1, col] <= 10^-15:
				continue;
		r = np.sqrt(A[row,col]**2 + A[row+1,col]**2)
		b = A[row+1,col]
		a = A[row,col]
		
		A[row:row+2,col+1:] = (a*A[row:row+2,col+1:] + (np.array([[-b],[b]])*A[row:row+2,col+1:])[::-1])/r
			
		A[row,col] = r
		A[row+1,col] = 0

		G = np.array([[a/r,b/r],[-b/r,a/r]])
		Q[row:row+2,:] = np.dot(G,Q[row:row+2,:])
	return Q.T,A



def test_gramSchmidt():
	for m,n in [(20,10),(25,15), (30,20)]:
		print 'CGS and MGS for matrix of size : (%d, %d)' % (m,n)
		A = vandermod(m,n)
		#print 'vandermode matrix(7,4):\n',A
		#print 'numpy QR :\n', np.linalg.qr(A)[1]
		Q,R = CGS(A)
		#print 'CGS QR: \n', R
		print 'relative error: ', lin.norm(A - np.dot(Q,R),2)/lin.norm(A,2)
		print 'orthogonality error: ', lin.norm(np.dot(Q.T,Q) - np.identity(n),2)
		A = vandermod(m,n)
		Q,R = MGS(A)
		#print 'MGS QR: \n',R
		print 'relative error: ', lin.norm(A - np.dot(Q,R),2)/lin.norm(A,2)
		print 'orthogonality error: ', lin.norm(np.dot(Q.T,Q) - np.identity(n),2)
		print '-------------------------------------------\n'


def MGS(A):
	m,n = A.shape
	Q = np.zeros((m,n))
	R = np.zeros((n,n))
	for k in range(n):
		for i in range(k):
			R[i,k] = np.sum(Q[:,i]*A[:,k]) + 0.
			A[:,k] = A[:,k] - (R[i,k]*Q[:,i])
		R[k,k] = np.sqrt(np.sum(A[:,k] ** 2))
		Q[:,k] = A[:,k]/R[k,k]
	return Q,R

def CGS(A):
	m,n = A.shape
	Q = np.empty((m,n))
	R = np.zeros((n,n))
	for k in range(n):
		for i in range(k):
			R[i,k] = np.sum(Q[:,i]*A[:,k])
		#R[:,k] = np.sum(A[:,k][:,np.newaxis]*Q[:,:k])
		for i in range(k):
			A[:,k] = A[:,k] - R[i,k]*Q[:,i]
		R[k,k] = np.sqrt(np.sum((A[:,k] ** 2)))
		Q[:,k] = A[:,k]/R[k,k]
	return Q,R

	


def givens_qr(A):
	for col in range(A.shape[1]):#
		for row in range(A.shape[0]-2,col-1,-1):
			if A[row+1, col] <= 10^-15:
				continue;
			r = np.sqrt(A[row,col]**2 + A[row+1,col]**2)
			b = A[row+1,col]
			a = A[row,col]
			
			A[row:row+2,col+1:] = (a*A[row:row+2,col+1:] + (np.array([[-b],[b]])*A[row:row+2,col+1:])[::-1])/r
			
			A[row,col] = r
			A[row+1,col] = 0
			
	return A


def housholder_qr(A):
	for col in range(A.shape[1]):
		rho = np.sqrt(np.sum(A[col:,col]**2))
		v = A[col:,col] + 0;
		v[0] -= rho;
		vTv = np.sum(v**2)#2*rho*(v[0])
		#update columns
		A[col,col] =  rho
		A[col+1:,col] = 0#v[1:]

		vTx = np.sum(v[:,np.newaxis] * A[col:,col+1:],axis=0)
		A[col:,col+1:] =  A[col:,col+1:] - (((2*vTx/vTv)[:,np.newaxis])*np.tile(v,(vTx.shape[0],1))).T 
	return A

def vandermod(m,n):
	A = np.empty((m,n))
	A[0,:] = 1;
	r = np.array(range(1,n+1))/(n+0.)
	for row in range(1,m):
		A[row,:] = A[row-1,:]*	r
	return A

if __name__ == '__main__':
	float_formatter = lambda x : "%.4f" % x
	np.set_printoptions(formatter={'float_kind':float_formatter})
	test_column_addition()
