function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%% prevent overflow
y      = theta * data;
y      = bsxfun(@minus, y, max(y, [], 1));
y_exp  = exp(y);

y_sum  = sum(y_exp,1);
y_norm = y_exp ./ y_sum; 
y_log  = log(y_norm);

cost = - sum( (y_log .* groundTruth)(:) ) ./ numCases + 0.5 * lambda * sum((theta.^2)(:));
thetagrad = ((y_norm - groundTruth) * data') ./ numCases + lambda * theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
