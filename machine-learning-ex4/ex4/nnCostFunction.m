function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%----------------------------part1---------------------------------
%cost(Unregularization)
a1 = [ones(m, 1) X];
a2_ex = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) a2_ex];
h = sigmoid(a2 * Theta2');

%1.my own method,seems a littile complex
%for i = 1:m,
%  X1 = h(i,:);
%  y1 = zeros(num_labels,1);
%  y1(y(i)) = 1;
%  J = J + (log(X1) * (-y1) - log(1 - X1) * (1 - y1));
%end;
%J = J / m; 

%2.convert y to vector
c = 1:num_labels;
yt = zeros(m,num_labels);
for i = 1:m
  yt(i,:) = (c == y(i));
end;
part1 = -yt .* log(h);
part2 = (1 - yt) .* log(1 - h);
J = 1 / m * sum(sum(part1 - part2));

%Regularization
part3 = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));
J = J + lambda / (2*m) * part3;

%-----------------------------part2-------------------------------------
%Backpropagation:(we can also use martix above£¬but instruction rrquire us use for-loop)
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for t = 1:m
  %step 1
  a1 = [1 ; X(t,:)'];
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1 ; a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  %step 2
  c = 1:num_labels;
  yt = (c == y(t))';
  delta3 = a3 - yt;
  %step 3
  delta2 = (Theta2' * delta3) .* sigmoidGradient([1;z2]);
  %step 4
  D2 = D2 + delta3 * a2';
  D1 = D1 + delta2(2:end) * a1';
  %step 5
  Theta1_grad = D1 / m;
  Theta2_grad = D2 / m;
 end;
% --------------------------part3----------------------
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * Theta2(:,2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
