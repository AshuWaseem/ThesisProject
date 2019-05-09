clear

load Demo-Dataset

%% generate data

%0.58064      7.1566     -1.1963     0.11217     0.11564    -0.37019
%% classify
rho = 0.58064 ;
c1 =7.1566;
c2 =7.1566;
% epsilon = 0.3;
threshold = -11217;
Ins_t =       0.11564  ;    ;% it should be larger than the epsilon
Hinge_s =-2.4065 ;


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

subplot(1,2,1)
plot(epsilon,accurcy,'r-.','linewidth',2)
%legend('Mean Zero-one Error(MZE)')
legend('Accuracy')
ylabel('Accuracy')
xlabel('\epsilon')
set(findobj('FontSize',10),'FontSize',10);
% axis([0 0.5 0.5 0.55])
%  axis([0 0.5 0.8 1])
subplot(1,2,2)
plot(epsilon,MAE,'b-.','linewidth',2)
%legend('Mean Absolute Error(MAE)')
legend('MAE')
xlabel('\epsilon')
ylabel('Mean absolute error')
set(findobj('FontSize',10),'FontSize',10);
% axis([0 0.5 0.6 0.65])
%  axis([0 0.5 0.0 0.3])
