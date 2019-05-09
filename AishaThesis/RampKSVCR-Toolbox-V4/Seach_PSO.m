



ObjectiveFunction = @rampKSCVR;
x0 = [1 4 4 0 10 -5];   % Starting point


lb = [-10 -10 -10 -10 -10 -10];
ub = [10 10 10 10 10 10];

rng default % For reproducibility
options = optimoptions('particleswarm','SwarmSize',10,'MaxIterations',100);


IntCon=0;
nonlcon=[];



for i=22:40
    
  %  x0=generateRandomValues(x0);
    
    mkdir (num2str(i));
    
    movefile ('allExecutions_PSO_1.txt',num2str(i),'f');
  [x,fval,exitflag] = particleswarm(ObjectiveFunction,6,lb,ub,options);

end

function x0=generateRandomValues(x0)

a = -10;
b = 10;
x0(1) = (b-a).*rand(1,1) + a;

x0(2) = (b-a).*rand(1,1) + a;
x0(3) = (b-a).*rand(1,1) + a;
x0(4) = (b-a).*rand(1,1) + a;
x0(5) = (b-a).*rand(1,1) + a;
x0(6) = (b-a).*rand(1,1) + a;
end
function res=rampKSCVR(x0)
% it should be larger than the epsilon



load Demo-Dataset
rho = x0(1);
c1 =x0(2);
c2 =x0(2);
epsilon = 0.3;
threshold = x0(4);
Ins_t = x0(5);% it should be larger than the epsilon


if Ins_t<0
   Ins_t=abs(Ins_t);
   if Ins_t<0.3
       Ins_t=Ins_t+0.3;
   end
end
if Ins_t<0.3
       Ins_t=Ins_t+0.3;
   end
Hinge_s = x0(6);

%% generate data
 
%% classify


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


avgaccuracy=accuracy;

res=1-avgaccuracy;
fileID = fopen('allExecutions_PSO_1.txt','a');
tw2=num2str(x0);
tw2=strcat(tw2,'= ');
tw2=strcat(tw2,num2str(res));
tw2=strcat(tw2,', ');

tw2=strcat(tw2,num2str(avgaccuracy));
tw2=strcat(tw2,'\n');
fprintf(fileID,tw2);
fclose(fileID);

end

