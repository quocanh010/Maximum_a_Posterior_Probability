%compute dct
function dct_v = compute_dct_vetor(x)
    x_dct = dct2(x);
    ind = reshape(1:numel(x_dct), size(x_dct)); %# indices of elements
    ind = fliplr( spdiags( fliplr(ind) ) );     %# get the anti-diagonals
    ind(:,1:2:end) = flipud( ind(:,1:2:end) );  %# reverse order of odd columns
    ind(ind==0) = [];                           %# keep non-zero indices
    dct_v = x_dct(ind); 
end
