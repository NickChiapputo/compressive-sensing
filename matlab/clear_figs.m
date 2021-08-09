% Get all figure handles.
figHandles = get( groot, 'Children' );

% Iterate through each figure and clear them.
for i = 1:length( figHandles )
	clf( figHandles( i ) );
end