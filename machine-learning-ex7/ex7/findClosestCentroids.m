function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

# More effecient to iterate through the centroids only:

# Distance matrix of size (m x K)
distance = zeros(size(X,1), K);
sq_distance = [];

for i = 1:K
		# Broadcast function applies function between 1st argument (matrix) 
		# and 2nd argument (vector)
		# Return [x1-mu1 x2-mu2]
		distance = bsxfun(@minus, X, centroids(i,:)); # centroids is (K x n)
		sq_distance = [sq_distance, sum(distance.^2, 2)];
endfor

# Sum across rows in distance to get (x1-mu1)^2+(x2-mu2)^2
[minval idx] = min(sq_distance, [], 2);

# Less effecient alternative looping throught examples and centroids:

### Loop over the examples
##for i = 1:length(idx)
##		distance = [];
##		# For each example, loop over the centroids
##		for j = 1:K
##				# Calculate magnitude^2: ||X(i)-centroid(i)||^2 = (x1-mu1)^2+(x2-mu2)^2
##				distance = [distance, sum((X(i,:)-centroids(j,:)).^2)];
##				if length(distance)==K
##						# Extract index corresponding to lowest distance
##						[value index] = min(distance);
##						# Add index to vector idx
##						idx(i) = index;
##				endif
##		endfor
##endfor

% =============================================================

end

