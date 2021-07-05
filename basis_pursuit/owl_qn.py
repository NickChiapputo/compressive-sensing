import numpy as np
from pylbfgs import owlqn
from util import idct2, dct2

global nx, ny, ri, b
def evaluate( x, g, step ):
	# Goal is to calculate:
	# (1) The norm squared of the residuals, sum( ( Ax - b ).^2 )
	# (2) The gradient 2*A'*( Ax - b )
	global nx, ny, ri, b

	# Expand x columns first
	x2 = x.reshape( ( nx, ny ) ).T

	# Ax is just the inverse 2D dct of x2
	Ax2 = idct2( x2 )

	# Stack columns and extract samples
	Ax = Ax2.T.flat[ ri ].reshape( b.shape )

	# Calculate the residual Ax-b and its 2-norm squared
	Axb = Ax - b
	fx = np.sum( np.power( Axb, 2 ) )

	# Project residual vector (k x 1) onto blank image (ny x nx)
	Axb2 = np.zeros( x2.shape )
	Axb2.T.flat[ ri ] = Axb # Fill columns first

	# A'(Ax-b) is just the 2D dct of Axb2
	AtAxb2 = 2 * dct2( Axb2 )
	AtAxb = AtAxb2.T.reshape( x.shape ) # Stack columns

	# Copy over the gradient vector
	np.copyto( g, AtAxb )

	return fx


def progress( x, g, fx, xnrom, gnorm, step, k, ls ):
	print( f'Iteration {k}', end = '\r' )
	return 0


def owl_qn_cs( image, n_x, n_y, indexes, measurements ):
	global nx, ny, ri, b
	nx = n_x
	ny = n_y
	ri = indexes
	b  = measurements

	# Get the dimensions of the image.
	ny, nx = image.shape

	# Take random samples of the image
	b = image.T.flat[ ri ].astype( float )

	# Perform the L1 minimization in memory.
	# Result is in the frequency domain.
	Xat2 = owlqn( nx * ny, evaluate, progress, 5 )
	print( '' )

	# Transform the output back into the spatial domain.
	Xat = Xat2.reshape( nx, ny ).T # Stack columns.
	Xa = idct2( Xat ).astype( 'uint8' )

	# Return the reconstructed signal.
	return Xa