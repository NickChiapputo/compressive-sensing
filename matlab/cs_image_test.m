tic

% Read in the image.
I = imread( "lena.png" );

% Convert to grayscale if image is in RGB 
% to convert from 3D matrix to 2D.
if ndims( I ) ~= 2
	I = rgb2gray( I );
end


% Take a square portion of the image and convert to a 1-D double vector.
start_idx = 250;
end_idx = 299;
num_idxs = ( end_idx - start_idx ) + 1;
I = I( [ start_idx:end_idx ], [ start_idx:end_idx] );
x = double( I( : ) );
n = length( x );


% Generate pseudo-random Phi matrix of size mxn.
% m: Number of samples to take (e.g., m = n / 3 takes a third of the
% samples as the original image).
m = floor( n / 2 );
Phi = randn( m, n );


% We want to solve y = Phi * Psi * a for a, but we only know Phi and Psi.
% Since we also know that y = Phi * x, we can get y using this relation.
% Since Phi is an mxn matrix where m < n, we are also limiting the number
% of samples from the original image.
y = Phi*x;


% Calculate the product of the Phi and Psi matrices (named Theta).
% This has to be done one column at a time to reduce system resource
% over-usage (the matrices can become very large and use a large amount of 
% RAM).
Theta = zeros(m,n);
for idx = 1:n
% 	fprintf( "Iteration %d\r", idx );
	
	tmp = zeros( 1, n );
    tmp( idx ) = 1;
    psi = idct( tmp )';
    Theta( :, idx ) = Phi * psi;
end
psi_1 = psi;

% Solve the convex optimization problem using the L1-norm with the
% constraint y = Theta * a.
cvx_begin
	variable a( n );
	minimize( norm( a, 1 ) );
	subject to
		Theta * a == y;
cvx_end


% Reconstruct the image.
psi = zeros( n );
for idx = 1:n
% 	fprintf( "Iteration %d\r", idx );
	
    tmp = zeros( 1, n );
    tmp( idx ) = 1;
    psi( :, idx ) = idct( tmp )';
end

reconstructed_vector = psi * a;


%% Results
% Display the original image, Psi, Phi, and reconstructed result.
% Use 'axis image' for appropriately proportioned axes and a tight bounding.
figure( 2 )

subplot( 1, 2, 1 )
imagesc( reshape( x, num_idxs, num_idxs ) )
title( 'Original Image' )
axis image

subplot( 1, 2, 2 )
imagesc( reshape( reconstructed_vector, num_idxs, num_idxs ) )
title( 'L1 Minimization Result' )
axis image

colormap gray


toc
