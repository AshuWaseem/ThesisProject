clear 

load Demo-Dataset
index = randperm(2300);
NSLKDDSample2300(isnan(NSLKDDSample2300)==1)=1;
NSLKDDSample2300 = NSLKDDSample2300(index,:);
unique(NSLKDDSample2300(:,end))
Train = NSLKDDSample2300(1:1500,:);
Test = NSLKDDSample2300(1501:end,:);

Algorithm = RampKSVCR();
name = 'RampKSVCR';

%%%%Grid search%%%%
obj.parameters.C = 10.^(-3:1:3); %C1=C2=C
% kernel width
obj.parameters.k = 10.^(-3:1:3); %gamma
obj.parameters.e = 0.1;
obj.parameters.t = 2 %%2.^(-3:1:3); %[2^-3, 2^-2,...,2^3]
obj.parameters.s = -0.8  %%-2.^(-3:1:3); 
obj.parameters.threshold=0;
obj.nOfFolds = 2;
obj.method = Algorithm;
obj.cvCriteria = ACC;

%%%%Evaluate the model on a fixed parameter%%%%
% obj.parameters.C = 5; %C1=C2=C
% % kernel width
% obj.parameters.k = 0.1; %gamma
% obj.parameters.e = 0.3;
% obj.parameters.t = 2; %[2^-3, 2^-2,...,2^3]
% obj.parameters.s = -0.8; 
% obj.parameters.threshold=0;
% obj.nOfFolds=2;
% obj.method = Algorithm;
% obj.cvCriteria = ACC;


% Train = load(fname1); txt or csv files fname1=filename.txt
% Test = load(fname2);

train.patterns = Train(:,1:end-1);
train.targets = Train(:,end);
test.patterns = Test(:,1:end-1);
test.targets = Test(:,end);

[train, test] = preProcessData(train,test);

% Running the algorithm
param = crossValide(obj,train);

info = Algorithm.runAlgorithm(train,test,param.pram);
time = info.trainTime; 
[confus,accuracy,numcorrect,precision,recall,F] = compute_accuracy_F(test.targets,info.predictedTest,unique(train.targets))

