import scipy.fftpack as spfft
import scipy.ndimage as spimg
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
	# Extract a small sample of the signal.
	y = image.T.flat[ ri ]

	# Create a blank white image and add in the randomly sampled pixels.
	Xm = 255 * np.ones( image.shape )
	Xm.T.flat[ ri ] = image.T.flat[ ri ]

	# Return the random indices, subsampled image, and the mask for visualization
	return y, Xm


def randomly_sample( img, total_samples, sample_percentage ):
	# Randomly select indices from the image and calculate the
	# number of measurements taken (M)
	ri = generate_random_samples( total_samples, sample_percentage )
	n_samples = len( ri )


	# Calculate the number of channels in the image (e.g., 3 for RGB image)
	_, _, n_channels = img.shape

	
	# Calculate the measurement vector and mask matrix for each channel of the image.
	y = np.zeros( ( n_samples, n_channels ), dtype = 'uint8' )
	mask = np.zeros( img.shape, dtype = 'uint8' )
	for j in range( n_channels ):
		y[ :, j ], mask[ :, :, j ] = create_mask( img[ :, :, j ], ri )


	return ri, y, mask


def read_image( image_path, as_gray=False, zoom_level=1.0 ):
	img = imread( image_path, as_gray=as_gray )
	ny, nx, n_channels = img.shape

	img_zoom = np.zeros( ( int( ny * zoom_level ), int( nx * zoom_level ), n_channels ) )
	for i in range( n_channels ):
		img_zoom[ :, :, i ] = spimg.zoom( img[ :, :, i ], zoom_level )

	ny, nx, n_channels = img_zoom.shape
	return img_zoom, ny, nx, n_channels
