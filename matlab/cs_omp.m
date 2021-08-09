%% clear all, close all, clc
clear_figs


%% System Parameters
n = 5000;
m = 500;
K = 100;


%% Generate Signal
% 'A' tone on a phone.
t = linspace( 0, 1/8, n );
f = sin( 1394 * pi * t ) + sin( 3266 * pi * t );

% Plot signal
figure( 1 );
subplot( 5, 1, 1 );
plot( t, f );
xlim( [ 0 1/8 ] );
amp = 1.2 * max( abs( f ) ); ylim( [ -amp amp ] );
title( 'Original Signal $\mathbf{f(t)}$', 'Interpreter', 'latex' );


%% Compute Sparse Components of Signal
ft = dct( f );

% Plot sparse components
% figure( 2 );
subplot( 5, 1, 2 );
plot( abs( ft ) );
title( 'DCT of F $|\mathbf{X(f)}|$', 'Interpreter', 'latex' );


%% Sub-Sample the Original Signal
x_m = zeros( n, 1 );
temp = randperm( n );	% Random permutation of [n]
ind = temp( 1:m );      % Grab the first 500 random indexes.
tr = t( ind );           % Grab x-axis values at each index.
x_m( ind ) = f( ind );
b = sin( 1394 * pi * tr ) + sin( 3266 * pi * tr );
b = b.';


% Take DCT of sub-sampled signal.
% ft_sub = dftmtx( fr );
% figure( 2 );
% hold on
% plot( ft_sub, 'Linewidth', [2] );


% Plot subsamples on original signal
% figure( 1 );
% hold on
subplot( 5, 1, 3 );
plot( t, x_m );
xlim( [ 0 1/8 ] );
title( 'Measured Values', 'Interpreter', 'latex' )


%% Generate Dictionary/Basis Matrix
% Figure out mapping from time to frequency domain.
D = dct( eye( n ) );	% Inverse DCT
A = D( ind, : );


%% Perform Recovery Operation
tic

residual = b;			% Residual.
x = zeros( n, 1 );		% Sparse coefficients.
Lambda = zeros( K, 1 );	% Selected support.
A_k = [];				% Selected column indices of A.

for k = 1:K
	max_val = 0;
	max_ind = 0;

	inds = setdiff( [ 1:m ], Lambda );
	for i = inds
		tmp = A( :, i )' * residual / norm( A( :, i ) );
		if abs( tmp ) > max_val
			max_val = abs( tmp );
			max_ind = i;
		end
	end
	Lambda( k ) = max_ind;
	A_k = [ A_k A( :, max_ind) ];
	x_ls = A_k \ b;
	residual = b - A_k * x_ls;
end

for i = 1:K
	x( Lambda( i ) ) = x_ls( i );
end

sig1 = dct( x );

toc
%% Plot the Results.
% Plot the recovered and original signals.
% figure( 3 ); 
subplot( 5, 1, 5 );
plot( t, f, t, sig1 );
xlim( [ 0+0.375*(1/8)/n 1/8*0.25+0.375*(1/8)/n ] );
amp = 1.2 * max( max( abs( f ) ), max( abs( sig1 ) ) ); ylim( [ -amp amp ] );
title( 'Recovered and Original Signals', 'Interpreter', 'latex' );

% subplot( 2, 1, 1 );
% plot( t, f, 'Linewidth', [2] );
% subplot( 2, 1, 2 );
% plot( t, sig1, 'Linewidth', [2] );
% title( 'Original vs Reconstructed Signal' );


% Plot the transform basis components.
% figure( 4 );
% hold on
% plot( ft, 'b', 'Linewidth', [2] );
% title( 'Original vs Reconstructed Components' );
subplot( 5, 1, 4 );
plot( abs( x ), 'r' );
title( 'Reconstructed Components $|\mathbf{\hat{X}(f)}|$', 'Interpreter', 'latex' )


%% Calculate Reconstruction Quality
mean_error = immse( sig1, f' );
psnr_error = psnr( sig1, f' );
fprintf( "K = %3d, MSE = %4.3f, PSNR = %5.3f\n", K, mean_error, psnr_error );
