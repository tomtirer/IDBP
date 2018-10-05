function X_hat = median_inpainting(Y,deletion_set)
% median estimator

[M,N] = size(Y);
Y0_median = Y;
Y0_median(deletion_set) = nan(size(deletion_set));
win_size = 0;
bitmap_NaN = isnan(Y0_median);
while sum(bitmap_NaN(:))>0
    Y0_median_prev = Y0_median;
    win_size = win_size + 1;
    indices_NaN = find(bitmap_NaN);
    for ind_nan=1:1:length(indices_NaN)
        [row_nan,col_nan] = ind2sub([M,N],indices_NaN(ind_nan));
        row_start = max([1,row_nan-win_size]);
        row_end = min([M,row_nan+win_size]);
        col_start = max([1,col_nan-win_size]);
        col_end = min([N,col_nan+win_size]);
        neighbors_NaN = Y0_median_prev(row_start:row_end,col_start:col_end);
        median_val = nanmedian(neighbors_NaN(:));
        Y0_median(row_nan,col_nan) = median_val;
    end
    bitmap_NaN = isnan(Y0_median);
end
X_hat = Y0_median;

