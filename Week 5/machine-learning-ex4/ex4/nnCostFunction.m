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

% PART 1
% Add a layer of 1s to X
A1 = [ones(m, 1) X];

% Compute A2 and add a layer of 1s to it
A2 = [ones(m, 1) sigmoid(A1*Theta1')];

% Compute h
h = sigmoid(A2*Theta2');

% Transform y into type [0 0 1 .. ]
y_mat = zeros(m, num_labels);

for i = 1:m
  range = 1:num_labels;
  y_mat(i, 1:num_labels) = (range == y(i, 1));
end 

% Compute cost (unregularized)
for i = 1:num_labels
  y_cur = y_mat(1:m, i);
  h_cur = h(1:m, i);
  J = J + (1/m)*(-y_cur'*log(h_cur) - (1 .- y_cur)'*log(1 .- h_cur));
end;

% Compute regularized cost
theta1_r = (Theta1(:, 2:input_layer_size + 1).^2)(:);
theta2_r = (Theta2(:, 2:hidden_layer_size + 1).^2)(:);

J = J + (sum(theta1_r) + sum(theta2_r))*lambda/(2*m);

% Part 2 (unregularized)
% Initialize deltas
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

% Iterate over each training example
for t = 1:m
  % Compute variables
  X_t = X(t, :);
  Y_t = y_mat(t, :);
  
  A1 = [1 X_t];
  Z2 = A1*Theta1';
  A2 = [1 sigmoid(Z2)];
  Z3 = A2*Theta2';
  A3 = sigmoid(Z3);
  
  % Calculate error terms
  D3 = (A3 - Y_t)';
  D2 = (Theta2'*D3).*sigmoidGradient([1 Z2]');
  
  % Modify delta term
  delta2 = delta2 + D3*(A2);
  delta1 = delta1 + (D2*(A1))(2:end, :);
end

Theta1_grad = delta1/m;
Theta2_grad = delta2/m;

% Part 3 (regularized)
Theta2_grad(:, 2:size(Theta2, 2)) = Theta2_grad(:, 2:size(Theta2, 2)) + lambda*Theta2(:, 2:size(Theta2, 2))/m;
Theta1_grad(:, 2:size(Theta1, 2)) = Theta1_grad(:, 2:size(Theta1, 2)) + lambda*Theta1(:, 2:size(Theta1, 2))/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
