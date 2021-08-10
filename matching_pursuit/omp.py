import numpy as np
import matplotlib.pyplot as plt

from util import randomly_sample
from util import create_mask
from util import dct2

from imageio import imwrite

def omp( A, b, K, N, M ):
	# System parameters
	residual = b		# r_0 = b
	x = np.empty( N, dtype='float64' )
	Lambda = np.empty( K )
	A_k = np.empty( (M,1) )

	M_range = np.arange( M )


	for k in range( 0, K ):
		print( f"\rIteration {k}", end = '' )

		max_val = 0
		max_ind = 0

		for i in np.setdiff1d( M_range, Lambda ):
			tmp = np.transpose( A[ :, i ] ) @ residual / np.linalg.norm( A[ :, i ] )
			if abs( tmp ) > max_val:
				max_val = abs( tmp )
				max_ind = i

		Lambda[ k ] = max_ind
		print( f"\n{A_k.shape}" )
		A_k = np.hstack( ( A_k, A[ :, max_ind ] ) )
		print( f"\n{A_k.shape}" )
		x_ls = np.linalg.pinv( A_k ) @ b
		residual = b - A_k @ x_ls


	# Calculate the column in the sensing matrix that has the 
	# largest absolute value of correlation with the residue.

def omp_cs( img, rows, cols, n_channels, sample_percentages ):
	# Define vectors to hold metric results.
	ssim_results = np.zeros( len( sample_percentages ) )
	mse_results  = np.zeros( len( sample_percentages ) )
	psnr_results = np.zeros( len( sample_percentages ) )

	A = dct2( np.identity( rows * cols ) )


	# Threshold number of coefficients
	K = 100


	# Iterate through each sample percentage value.
	for i, sample in enumerate( sample_percentages ):
		print( f'Samples = {100 * sample}%' )

		# Randomly sample from the image at the given rate.
		# Retrieve the sampled indices, the sampled measurements,
		# and the masked image.
		indices, y, masks = randomly_sample( img, rows * cols, sample )
		A_i = A[ indices ]


		# Iterate through and reconstruct at each channel of the image.
		for j in range( n_channels ):
			omp( A_i, y[ :, j ], K, cols, rows )
