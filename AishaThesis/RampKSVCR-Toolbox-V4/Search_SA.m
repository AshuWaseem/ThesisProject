



ObjectiveFunction = @rampKSCVR;
x0 = [1 4 4 0 10 -5];   % Starting point


lb = [-10 -10 -10 -10 -10 -10];
ub = [10 10 10 10 10 10];

rng default % For reproducibility
[x,fval,exitFlag,output] = simulannealbnd(ObjectiveFunction,x0,lb,ub);

function res=rampKSCVR(x0)
% it should be larger than the epsilon



load Demo-Dataset
rho = x0(1);
c1 =x0(2);
c2 =x0(3);
% epsilon = 0.3;
threshold = x0(4);
Ins_t = x0(5);% it should be larger than the epsilon
Hinge_s = x0(6);

%% generate data
 
%% classify



index = randperm(2300);

NSLKDDSample2300(isnan(NSLKDDSample2300)==1)=1;

NSLKDDSample2300 =NSLKDDSample2300(index,:);
unique(NSLKDDSample2300(:,end))
TrainData = NSLKDDSample2300(1:1500,:);
TestData = NSLKDDSample2300(1501:end,:);
X = TrainData(:,1:end-1); y =TrainData(:,end);
T=TestData(:,1:end-1); TestLabel =TestData(:,end);


result=[];


%  ker = 'linear';
ker = 'rbf';
sigma = 2/10;%sigma = 1/200;
epsilon=0:0.025:0.5;
L=unique(y);
for ii=1:length(epsilon)
model = RampLossKSVCR(X,y,c1,c2,epsilon(ii),rho, Ins_t, Hinge_s, ker, sigma );
%f = TestPrecisionNonLinear(par,X, y,X, y, ker,epsilon,sigma);


%%%predit part%%%%%
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
idx = finalOutput;


idx = L(idx);
accurcy(ii)= mean(idx==TestLabel);
MAE(ii) =  mean(abs(idx-TestLabel));

end


avgaccuracy=sum(accurcy)/length(accurcy);

res=1-avgaccuracy;
fileID = fopen('allExecutions_SA_2.txt','a');
tw2=num2str(x0);
tw2=strcat(tw2,'= ');
tw2=strcat(tw2,num2str(res));
tw2=strcat(tw2,', ');

tw2=strcat(tw2,num2str(avgaccuracy));
tw2=strcat(tw2,'\n');
fprintf(fileID,tw2);
fclose(fileID);

end

