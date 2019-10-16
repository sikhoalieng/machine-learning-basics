function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

# List of C and sigma values to test
values_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
error_vec = [];
# Nested for loops to test every combinations of C and simga in the values_vec
for i = values_vec
		temp_C = i;
		for j = values_vec
				temp_sigma = j;
				# Example from line 108 of ex6.m
				model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma)); 
				# Make predictions using svmPredict
				predictions = svmPredict(model, Xval);
				# Calculate error probability where predictions!=yval/total predictions
				error = mean(double(predictions ~= yval));
				# Add elements of C, sigma, and error to error_vec
				error_vec = [error_vec; temp_C temp_sigma error];
				printf("For C=%d, sigma=%d: ", i,j), error
		endfor
endfor

# Minimum values and indices of each column in error_vec
[min_values, indices] = min(error_vec);
# Extract only the index from the error column (column 3)
min_index = indices(:, 3);
# Returns the value for C from column 1 of error_vec where row=min_index
C = error_vec(min_index, 1:1)
# Returns the value for sigma from column 2 of error_vec where row=min_index
sigma = error_vec(min_index, 2:2)

% =========================================================================

end
