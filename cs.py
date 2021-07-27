import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as spimg
import imageio

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

from math import trunc
from time import time

from util import idct2, dct2
from util import generate_random_samples, block_indexes
from util import create_mask
from util import read_image

from basis_pursuit.owl_qn import owl_qn_cs
from matching_pursuit.omp import omp_cs


def compression( image_path, save_folder, sample_percentages ):
	# Define vectors to hold metric results.
	ssim_results = np.zeros( len( sample_percentages ) )
	mse_results  = np.zeros( len( sample_percentages ) )
	psnr_results = np.zeros( len( sample_percentages ) )


	# Read in image and calculate dimensions.
	original_image, ny, nx, n_channels = read_image( image_path, as_gray = False )


	final_result = np.zeros( original_image.shape, dtype = 'uint8' )
	masks = np.zeros( original_image.shape, dtype = 'uint8' )

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
			b, masks[ :, :, j ] = create_mask( original_image[ :, :, j ], ri )

			# Compute results using OWL-QN
			final_result[ :, :, j ] = owl_qn_cs( original_image[ :, :, j ], nx, ny, ri, b )


		# Compute Structural Similarity Index (SSIM) of 
		# reconstructed image versus original image.
		ssim_results[ i ] = structural_similarity( original_image, final_result, data_range = final_result.max() - final_result.min(), multichannel = True )
		mse_results[ i ]  = mean_squared_error( original_image, final_result )
		psnr_results[ i ] = peak_signal_noise_ratio( original_image, final_result, data_range = final_result.max() - final_result.min() )


		# Save images.
		imageio.imwrite( f'results/{save_folder}mask_{trunc( 100 * sample_percentage )}.png', masks )
		imageio.imwrite( f'results/{save_folder}recover_{trunc( 100 * sample_percentage )}.png', final_result )

		print( f'Elapsed Time: {time() - start:.3f} seconds.\n' )

	for i, sample_percentage in enumerate( sample_percentages ):
		print( f'{trunc( 100 * sample_percentage ): 6.2f}%:\n    SSIM: {ssim_results[ i ]}\n    MSE:  {mse_results[ i ]}\n    PSNR: {psnr_results[ i ]}\n' )


def upscale( image_path, save_folder ):
	global nx, ny, ri, b


	grayscale = True
	upscale_ratio = 2
	image_reduction = 0.125

	# Read in the original image.
	original_image = spimg.zoom( imageio.imread( image_path, as_gray = grayscale ), image_reduction )
	if grayscale:
		ny, nx = original_image.shape
		n_channels = 1
	else:
		ny, nx, n_channels = original_image.shape

	# Calculate the upscaled resolution.
	ny *= upscale_ratio
	nx *= upscale_ratio

	# Set the selected indices as every other index of the upscaled resolution.
	ri = np.array( [ ( ( y * nx ) + x ) for y in range( 0, ny, upscale_ratio ) for x in range( 0, nx, upscale_ratio ) ] )
	print( f'Selected Indices:\n{ri}\nNumber of indices: {ri.shape}' )


	# Create upscaled mask.
	Xm = 255 * np.ones( ( ny, nx ), dtype = 'uint8' )
	Xm.T.flat[ ri ] = original_image.T.flat


	# Reconstruction.
	final_result = np.zeros( ( ny, nx ), dtype = 'uint8' )
	b = original_image.T.flat[ : ].astype( float )

	# Perform the L1 minimization in memory.
	# Result is in the frequency domain.
	Xat2 = owlqn( nx * ny, evaluate, None, 5 )
	print( '' )

	# Transform the output back into the spatial domain.
	Xat = Xat2.reshape( nx, ny ).T # Stack columns.
	Xa = idct2( Xat ).astype( 'uint8' )

	# Show the masked and reconstructed image.
	plt.figure()
	f, axarr = plt.subplots( 1, 2 )

	axarr[ 0 ].imshow( Xm, cmap = 'gray' )
	axarr[ 1 ].imshow( Xa, cmap='gray' )
	plt.show()



if __name__ == '__main__':
	# Define image options and select one.
	image_paths = [ 'img/lena.png', 'img/mountain.jpeg', 'img/mandrill.png', 'img/neuschwanstein.jpg' ]
	save_folders = [ 'lena/', 'mountain/', 'mandrill/', 'neuschwanstein/' ]
	image_selection = 3
	image_path = image_paths[ image_selection ]
	save_folder = save_folders[ image_selection ]


	# Percentage of pixel samples to keep [0.0, 1.0].
	sample_percentages = [ 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.01 ] 
	sample_percentages = [ 0.50 ]

	# Flag to select upscaling or compressing implementation.
	upscaling = 0


	# Flag to select which CS technique to use.
	# Key:
	#	Basis Pursuit (OWL-QN)			: 0
	#	Orthogonal Matching Pursuit 	: 1 
	alg_select = 1


	# Read in the image and calculate its size.
	img, rows, cols, n_channels = read_image( image_path )


	if( alg_select == 0 ):
		if upscaling:
			upscale( image_path, save_folder )
		else:
			compression( image_path, save_folder, sample_percentages )
	elif( alg_select == 1 ):
		omp_cs( img, rows, cols, n_channels, sample_percentages )
	else:
		print( f'ERROR: Invalid algorithm selected ({alg_select = }).' )
