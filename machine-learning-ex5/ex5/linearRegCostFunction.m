function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

# Unregularized cost function and gradient
J_unreg = 1/(2*m)*sum((X*theta-y).^2);
# Note:X is multiplied outside of the summation; 
# gradient needs to be 2x1 to match Theta (hence the transpose)
grad_unreg = 1/m*sum((X*theta-y).*X)';

# Theta(1) is set to zero to avoid regularizing the first theta value
theta(1) = 0;
J_reg = lambda/(2*m)*sum(theta.^2);
grad_reg = lambda/m*theta;

# Adding regularization to cost function and gradient
J = J_unreg + J_reg;
grad = grad_unreg + grad_reg;

% =========================================================================

grad = grad(:);

end
