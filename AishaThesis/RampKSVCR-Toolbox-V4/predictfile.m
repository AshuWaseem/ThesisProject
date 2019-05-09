xy = T;
nT=size(xy,1);
d= zeros(nT,length(model.b));
if nT >1000
    for j=1:nT/1000
        xnewk=xy(1000*(j-1)+1:1000*j,:);
        d(1000*(j-1)+1:1000*j,:) = Kernel( ker , xnewk',X',sigma)* model.Beta+ ones(size(xnewk,1),1)*model.b;
    end
    xnewk=xy(1000*j+1:nT,:);
    dpred(1000*j+1:nT,:)= Kernel( ker , xnewk',X',sigma)*model.Beta+ ones(size(xnewk,1),1)*model.b;
else
    d =  Kernel( ker , xy',X',sigma)*model.Beta+ ones(nT,1)*model.b;
end

% pred(d<-threshold) = -1; pred(d >threshold) = 1;
pred = -(d<-threshold) + (d >threshold) ;
expLosses=zeros(size(pred,1),length(unique(y)));

for i=1:length(unique(y)),
    expLosses(:,i) = sum(pred == repmat(model.Code(i,:),size(pred,1),1),2);
end
[minVal,finalOutput] = max(expLosses,[],2);
pred = finalOutput;