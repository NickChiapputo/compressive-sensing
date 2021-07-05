import scipy.fftpack as spfft
import numpy as np
from imageio import imread

def dct2( x ):
	return spfft.dct( spfft.dct( x.T, norm='ortho', axis=0 ).T, norm='ortho', axis=0 )


def idct2( x ):
	return spfft.idct( spfft.idct( x.T, norm='ortho', axis=0 ).T, norm='ortho', axis=0 )


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
	global b

	# Extract a small sample of the signal.
	b = image.T.flat[ ri ]

	# Create a blank white image and add in the randomly sampled pixels.
	Xm = 255 * np.ones( image.shape )
	Xm.T.flat[ ri ] = image.T.flat[ ri ]

	# Return the random indices, subsampled image, and the mask for visualization
	return b, Xm


def read_image( image_path, as_gray=False ):
	img = imread( image_path, as_gray=as_gray )
	ny, nx, n_channels = img.shape
	return img, ny, nx, n_channels