function [learn, val] = k_FoldCV_SPLIT(data, nfold, i)
    idx = mod(1:size(data,1), nfold);
    learn = data(idx == i, :);
    val = data(idx == i, :);
end
