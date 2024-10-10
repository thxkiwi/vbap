% The azimuth angles of the loudspeakers in degrees
speakerAzimuthAngles = [30, 110, 150, -150, -110, -30];

% Unit vectors for the loudspeakers. Note that matrix is constructed as the transpose of the vectors.
% When P' = G * L, the result P' is the correct p = g1 L1 + g2 L2 by scaled vector addition. The the
% projections of the unit vectors sum to the unit vector in the direction of the source.
[x, y] = pol2cart(speakerAzimuthAngles / 180 * pi, ones(size(speakerAzimuthAngles)));
L = cat(2, x', y');

ngroups = [(1:length(speakerAzimuthAngles))', [2:length(speakerAzimuthAngles) 1]'];

% The number of pairs of loudspeakers (assuming they are in a circle also the number of loudspeakers).
N = size(ngroups, 1);

% The azimuth angles of the sources in degrees
sourceAngles = [15, -15, 180, 0, 30];

% The number of sources
M = length(sourceAngles);

[x, y] = pol2cart(sourceAngles / 180 * pi, ones(size(sourceAngles)));
P = cat(1, x, y);

Ln1n2 = zeros(2, 2);

G = zeros(M, N, 2);

for m = 1:M
    for n = 1:N
        Ln1n2(:,:) = L(ngroups(n, :), :);
        G(m, n, :) = P(:, m)' * inv(Ln1n2);  
    end
end

max(0, G)

% >> vbap

% ans(:,:,1) =

%     1.0116    1.1001         0         0         0    0.2989
%     0.8318    0.4027         0         0         0    0.8165
%          0         0    0.5774    1.4619    0.5077         0
%     0.9542    0.7779         0         0         0    0.5774
%     1.0000    1.3473         0         0         0         0


% ans(:,:,2) =

%          0         0         0    0.4027    0.8318    0.8165
%          0         0         0    1.1001    1.0116    0.2989
%     0.5077    1.4619    0.5774         0         0         0
%          0         0         0    0.7779    0.9542    0.5774
%     0.0000         0         0         0    0.6527    1.0000

% >> 
