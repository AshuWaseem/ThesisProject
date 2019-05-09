clear
load Demo-Dataset


%% generate data
 
%% classify
rho = 1;
c1 =5;
c2 =5;
epsilon = 0.3;
threshold = 0; 
Ins_t = 2;% it should be larger than the epsilon
Hinge_s = -0.8;

%ker = 'linear';
ker = 'rbf';
sigma = 1/10;%sigma = 1/200;

index = randperm(2300);
NSLKDDSample2300(isnan(NSLKDDSample2300)==1)=1;
NSLKDDSample2300 = NSLKDDSample2300(index,:);
unique(NSLKDDSample2300(:,end))
TrainData = NSLKDDSample2300(1:1500,:);
TestData = NSLKDDSample2300(1501:end,:);
X = TrainData(:,1:end-1); y =TrainData(:,end);
T=TestData(:,1:end-1); actual =TestData(:,end);


result=[];

classes=unique(y);
model = RampLossKSVCR(X,y,c1,c2,epsilon,rho, Ins_t, Hinge_s, ker, sigma );
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
pred = classes(pred);

%[confus,ClassLabel] = confusionmat(actual,pred)
[confus,accuracy,numcorrect,precision,recall,F] = compute_accuracy_F(actual,pred,classes);



