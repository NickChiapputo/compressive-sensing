import numpy as np
import matplotlib.pyplot as plt

from util import randomly_sample
from util import create_mask
from util import dct2

from imageio import imwrite

def omp( y, dictionary ):
	# System parameters
	print( f'{dictionary.shape = }' )

	# Initialization
	residual = y 		# r_0 = b
	# estimated_signal = np.zeros(  )	# x_tilde = ∅
	# atoms = 


	# Calculate the column in the sensing matrix that has the 
	# largest absolute value of correlation with the residue.

def omp_cs( img, rows, cols, n_channels, sample_percentages ):
	# Define vectors to hold metric results.
	ssim_results = np.zeros( len( sample_percentages ) )
	mse_results  = np.zeros( len( sample_percentages ) )
	psnr_results = np.zeros( len( sample_percentages ) )

	A = dct2( np.identity( rows * cols ) )


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
			opm( y[ :, j ], A_i )
