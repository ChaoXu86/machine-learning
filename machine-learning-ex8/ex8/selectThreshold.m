function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    predictions = pval < epsilon;
    
    %% ---------------
    %%          act 1  act 0
    %%  pred 1  x      y
    %%  pred 0  z      w
    %%
    %%  F1 = 2(x/(x+z) * x/(x+y)) / (x/(x+z) + x/(x+y))
    %%     = 2 x^2 / (x(x+y) + x(x+z))
    %%     = 2x / (2x + y + z)
    %%
    %%  x   : number of pred == yval == 1
    %%  y+z : number of pred /= yval
    x = sum(predictions(predictions == yval) == 1);
    y_plus_z = sum(predictions ~= yval);
    F1 = 2*x/(2*x + y_plus_z);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
