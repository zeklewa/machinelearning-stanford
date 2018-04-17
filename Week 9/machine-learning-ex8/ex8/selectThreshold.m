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

    result_p = (pval < epsilon);
    
    % Values for false positives, false negatives, true positives, true negatives
    f_p = 0;
    f_n = 0;
    t_p = 0;
    t_n = 0;
    
    for i = 1:size(yval, 1)
      % p - y
      % False positive and true positive
      if result_p(i, 1) == 1
        if yval(i, 1) == 0
          f_p = f_p + 1;
        else
          t_p = t_p + 1;
        end
      % True negative and false negative
      else
        if yval(i, 1) == 0
          t_n = t_n + 1;
        else
          f_n = f_n + 1;
        end
      end
    end
    
    precision = t_p/(t_p + f_p);
    recall = t_p/(t_p + f_n);
    
    F1 = 2*recall*precision/(recall + precision);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
