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


L1 = X;
a1 = [ones(size(X)(1), 1), X]; % padding bias
L2 = sigmoid(a1 * Theta1'); 
a2 = [ones(size(L2)(1),1), L2]; % padding bias
L3 = sigmoid(a2 * Theta2');

%% convert each y label to Y matrix
%%  1 -> [1;0;0; ... ;0]
%%  2 -> [0;1;0; ... ;0]
%% 10 -> [0;0;0; ... ;1]
Y = [];
labels = max(y(:));
for i = 1:size(y)(1)
  newrow = zeros(1,labels);
  newrow(y(i)) = 1; 
  Y = [Y;newrow];
end

%% calculate the J(theta)
J_nonReg = (log(L3) .* -Y - log(1-L3) .* (1-Y)) / m;% matrix J(theta), m x size(Theta2)(1)

Theta1NoBias = Theta1(:,2:end);
Theta2NoBias = Theta2(:,2:end);
SumSqrTheta1NoBias = sum((Theta1NoBias .* Theta1NoBias)(:));
SumSqrTheta2NoBias = sum((Theta2NoBias .* Theta2NoBias)(:));
J = sum(J_nonReg(:)) + lambda * (SumSqrTheta1NoBias + SumSqrTheta2NoBias)/(2*m)

delta3 = L3 - Y; % 5000 * 10
delta2 = (delta3 * Theta2 ) .* (a2 .* (1-a2)); % 5000 * 26 

Delta_L2 = delta3' * a2; % 10 * 26 
Delta_L1 = delta2' * a1; % 26 * 401
Delta_L1 = Delta_L1(2:end,:); % 25 * 401, L1 does not have bias

Theta2ZeroBias = [zeros(size(Theta2)(1),1), Theta2NoBias];
Theta1ZeroBias = [zeros(size(Theta1)(1),1), Theta1NoBias];

Theta2_grad = Delta_L2/m + lambda * Theta2ZeroBias / m;
Theta1_grad = Delta_L1/m + lambda * Theta1ZeroBias / m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
