function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

y      = theta * data;
y      = bsxfun(@minus, y, max(y, [], 1));
y_exp  = exp(y);
y_sum  = sum(y_exp,1);
y_norm = y_exp ./ y_sum; 
y_log  = log(y_norm);

[_, pred] = max(y_log);


% ---------------------------------------------------------------------

end

