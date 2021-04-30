%clear all, close all, clc
clear_figs

% 'A' tone on a phone.
n = 5000;
m = 500;
t = linspace( 0, 1/8, n );
f = sin( 1394 * pi * t ) + sin( 3266 * pi * t );

figure( 1 );
plot( t, f, 'Linewidth', 2 );
title( 'Original Signal (f)' );

ft = dct( f);
figure( 2 );
plot( ft, 'Linewidth', [2] );
title( 'dct( f )' );


% Sub-sample the original signal.
temp = randperm( n );	% Random permutation of [n]
ind = temp( 1:m );      % Grab the first 500 random indexes.
tr = t( ind );           % Grab x-axis values at each index.
fr = sin( 1394 * pi * tr ) + sin( 3266 * pi * tr );




figure( 1 );
hold on
plot( tr, fr, 'ro', 'Linewidth', 2 );


% Figure out mapping from time to frequency domain.
D = dct( eye( n, n ) );
A = D( ind, : );


% Least Square fit.
% This won't fit properly. Plotting the DCT of this will show the Fourier
% components. They will all be very small because least square wants to
% make everything small. It won't want to make the proper coefficients
% large like they should be because that increases the "error".
%x = pinv( A ) * fr'; % Least square fit.


% Perform convex optimization to minimize the L1-norm (least absolute
% deviations) subject to the constraint Ax = fr'
cvx_begin
	variable x( n );
	minimize( norm( x, 1 ) );
	subject to
		A*x == fr.';
cvx_end

sig1 = dct( x );


figure( 3 ); 
plot( t, f, t, sig1, 'Linewidth', [2] );

% subplot( 2, 1, 1 );
% plot( t, f, 'Linewidth', [2] );
% subplot( 2, 1, 2 );
% plot( t, sig1, 'Linewidth', [2] );

title( 'Original vs Reconstructed Signal' );

% Show the Fourier components.
figure( 4 );
plot( ft, 'b', 'Linewidth', [2] );
hold on
plot( x, 'r', 'Linewidth', [2] );
title( 'Original vs Reconstructed Components' );
