import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import imageio
import cvxpy as cvx

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

from pylbfgs import owlqn

from math import trunc
from time import time


def dct2( x ):
	return spfft.dct( spfft.dct( x.T, norm='ortho', axis=0 ).T, norm='ortho', axis=0 )

def idct2( x ):
	return spfft.idct( spfft.idct( x.T, norm='ortho', axis=0 ).T, norm='ortho', axis=0 )


def evaluate( x, g, step ):
	# Goal is to calculate:
	# (1) The norm squared of the residuals, sum( ( Ax - b ).^2 )
	# (2) The gradient 2*A'*( Ax - b )

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


# Compute a percentage of the total number of pixels and then 
# generating a randomly permuted masked array of zeros and ones.
def generate_random_samples( total_samples, sample_percentage ):
	k = round( total_samples * sample_percentage )
	ri = np.random.choice( total_samples, k, replace = False )

	return ri


def block_indexes( width, height, rows, cols ):
	start_row = ( height // 2 ) - ( rows // 2 )
	end_row   = start_row + rows

	start_col = ( width // 2 ) - ( cols // 2 )
	end_col   = start_col + cols

	ri = []
	for i in range( 1, cols + 1 ):
		ri = np.hstack( [ ri, np.arange( ny * ( start_col - 1 + i ) + start_row, ny * ( start_col - 1 + i ) + start_row + rows + 1 ) ] )
	ri = ri.astype( 'int' )
	ri = np.setdiff1d( np.array( np.arange( 0, width * height ) ), ri )

	return ri


def create_mask( image, ri ):
	# Extract a small sample of the signal.
	b = image.T.flat[ ri ]

	# Create a blank white image and add in the randomly sampled pixels.
	Xm = 255 * np.ones( image.shape )
	Xm.T.flat[ ri ] = image.T.flat[ ri ]

	# Return the random indices, subsampled image, and the mask for visualization
	return b, Xm


def owl_qn_cs( image, ri, eval_callback, progress_callback ):
	# Get the dimensions of the image.
	ny, nx = image.shape

	# Take random samples of the image
	b = image.T.flat[ ri ].astype( float )

	# Perform the L1 minimization in memory.
	# Result is in the frequency domain.
	Xat2 = owlqn( nx * ny, eval_callback, progress_callback, 5 )
	print( '' )

	# Transform the output back into the spatial domain.
	Xat = Xat2.reshape( nx, ny ).T # Stack columns.
	Xa = idct2( Xat ).astype( 'uint8' )

	# Return the reconstructed signal.
	return Xa


if __name__ == '__main__':
	# Define image options and select one.
	image_paths = [ 'img/lena.png', 'img/mountain.jpeg', 'img/mandrill.png', 'img/neuschwanstein.jpg', '../van_gogh.jpg' ]
	save_folders = [ 'lena/', 'mountain/', 'mandrill/', 'neuschwanstein/', 'van_gogh/' ]
	image_selection = 0
	image_path = image_paths[ image_selection ]
	save_folder = save_folders[ image_selection ]

	# Define global parameters.
	sample_percentages = [ 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.01 ] # Percentage of pixel samples to keep [0.0, 1.0].
	image_reduction = 1.00      # Ratio of new image size to original [0.0, 1.0].

	# Define vectors to hold metric results.
	ssim_results = np.zeros( len( sample_percentages ) )
	mse_results  = np.zeros( len( sample_percentages ) )
	psnr_results = np.zeros( len( sample_percentages ) )

	# Read in image, resize, and calculate new size.
	original_image = imageio.imread( image_path, as_gray=False )
	zoomed_image = original_image
	# zoomed_image = spimg.zoom( original_image, image_reduction )
	ny, nx, n_channels = zoomed_image.shape


	final_result = np.zeros( zoomed_image.shape, dtype = 'uint8' )
	masks = np.zeros( zoomed_image.shape, dtype = 'uint8' )

	# Iterate through each sample percentage value.
	for i, sample_percentage in enumerate( sample_percentages ):
		print( f'Samples = {100 * sample_percentage}%' )
		start = time()

		# Get random sample indices so they're the same for all channels
		ri = generate_random_samples( nx * ny, sample_percentage )

		# Iterate through each color channel
		for j in range( n_channels ):
			# Randomly sample from the image with the given percentage.
			# Retrieve the samples (b) and the masked image.
			b, masks[ :, :, j ] = create_mask( zoomed_image[ :, :, j ], ri )

			# Compute results using OWL-QN
			final_result[ :, :, j ] = owl_qn_cs( zoomed_image[ :, :, j ], ri, evaluate, progress )


		# Compute Structural Similarity Index (SSIM) of 
		# reconstructed image versus original image.
		ssim_results[ i ] = structural_similarity( zoomed_image, final_result, data_range = final_result.max() - final_result.min(), multichannel = True )
		mse_results[ i ]  = mean_squared_error( zoomed_image, final_result )
		psnr_results[ i ] = peak_signal_noise_ratio( zoomed_image, final_result, data_range = final_result.max() - final_result.min() )
		# print( f'{ssim = }\n{mse = }\n{psnr = }' )


		# Save images.
		imageio.imwrite( f'results/{save_folder}mask_{trunc( 100 * sample_percentage )}.png', masks )
		imageio.imwrite( f'results/{save_folder}recover_{trunc( 100 * sample_percentage )}.png', final_result )

		print( f'Elapsed Time: {time() - start:.3f} seconds.\n' )

	for i, sample_percentage in enumerate( sample_percentages ):
		print( f'{trunc( 100 * sample_percentage ): 6.2f}%:\n    SSIM: {ssim_results[ i ]}\n    MSE:  {mse_results[ i ]}\n    PSNR: {psnr_results[ i ]}\n' )
