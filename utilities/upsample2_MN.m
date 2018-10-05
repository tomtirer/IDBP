function y = upsample2_MN(x,K,siz)
for i = 1:size(x,3)
    y(:,:,i) = upsample(upsample(x(:,:,i),K)',K)';
end
if size(y,1)>siz(1)
    y(end-(size(y,1)-siz(1)-1):end,:,:)=[]; % make sure to keep original size after down-up
end
if size(y,2)>siz(2)
    y(:,end-(size(y,2)-siz(2)-1):end,:)=[]; % make sure to keep original size after down-up
end

end