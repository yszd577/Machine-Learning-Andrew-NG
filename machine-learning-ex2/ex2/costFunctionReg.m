function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%第一种做法
n = size(theta,1);
part1 = -1 * y' * log(sigmoid(X * theta));
part2 = (1 - y)' * log(1 - sigmoid(X * theta));
J = 1 / m * (part1 - part2) + lambda / (2 * m) * (theta(2:n)' * theta(2:n));
grad(1) = 1 / m * X(:,1)' * (sigmoid(X * theta) - y);
grad(2:n) = 1 / m * X(:,2:n)' * (sigmoid(X * theta) - y) + lambda / m * theta(2:n);

%第二种做法
 % theta0 不参与正则化。直接让变量等于theta，将第一个元素置为0，再参与和 λ 的运算
  %t = theta;  t(1) = 0; 
  
  % 第一项
  %part1 = -y' * log(sigmoid(X * theta));
  % 第二项
  %part2 = (1 - y)' * log(1 - sigmoid(X * theta));
  
  % 正则项
  %regTerm = lambda / 2 / m * t' * t;
  %J = 1 / m * (part1 - part2) + regTerm; 

  % 梯度
  %grad = 1 / m * X' *((sigmoid(X * theta) - y)) + lambda / m * t;

% =============================================================

end
