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
sourceAngles = [15];

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
