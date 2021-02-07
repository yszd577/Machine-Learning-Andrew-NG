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

%��һ������
n = size(theta,1);
part1 = -1 * y' * log(sigmoid(X * theta));
part2 = (1 - y)' * log(1 - sigmoid(X * theta));
J = 1 / m * (part1 - part2) + lambda / (2 * m) * (theta(2:n)' * theta(2:n));
grad(1) = 1 / m * X(:,1)' * (sigmoid(X * theta) - y);
grad(2:n) = 1 / m * X(:,2:n)' * (sigmoid(X * theta) - y) + lambda / m * theta(2:n);

%�ڶ�������
 % theta0 ���������򻯡�ֱ���ñ�������theta������һ��Ԫ����Ϊ0���ٲ���� �� ������
  %t = theta;  t(1) = 0; 
  
  % ��һ��
  %part1 = -y' * log(sigmoid(X * theta));
  % �ڶ���
  %part2 = (1 - y)' * log(1 - sigmoid(X * theta));
  
  % ������
  %regTerm = lambda / 2 / m * t' * t;
  %J = 1 / m * (part1 - part2) + regTerm; 

  % �ݶ�
  %grad = 1 / m * X' *((sigmoid(X * theta) - y)) + lambda / m * t;

% =============================================================

end
