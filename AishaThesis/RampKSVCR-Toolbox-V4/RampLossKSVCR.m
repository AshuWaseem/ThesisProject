function model = RampLossKSVCR(X,Y,c1,c2,epsilon,rho, Ins_t, Hinge_s,ker,  sigma )
label = unique(Y);
p = length(label);
[n,m] = size(X);
nch =  nchoosek([1:p],2);
Code = zeros(p,size(nch,1));
K=Kernel(ker, X',X',sigma);
A = [ K + rho*eye(n)]\speye(n);
%A = invChol_mex(K + rho*speye(n));
Yk = zeros(n,size(nch,1)); 

% for k =1:size(nch,1)
%     Code(nch(k,:),k) = [-1,1];
%     i = nch(k,1); 
%     j =nch(k,2);
%     Yk= -(Y==label(i))+(Y==label(j)); 
%     [Betak, bk,Alphak] = SubRampLossKSVCR(K,A,Yk,c1,c2,n,epsilon,rho, Ins_t, Hinge_s);
%     Beta(:,k) = Betak;
%     Alpha(:,k) = Alphak;
%     b(k) = bk;
% end

for k =1:size(nch,1)
    Code(nch(k,:),k) = [-1,1];
    i = nch(k,1); 
    j =nch(k,2);
    Yk(:,k)= -(Y==label(i))+(Y==label(j)); 
end
[Beta, b,deta] = SubRampLossKSVCR(K,A,Yk,c1,c2,n,epsilon,rho, Ins_t, Hinge_s);

model.Beta = Beta;
model.b=b;
model.deta = deta;
model.Code = Code;

end

function [Beta, bk,deta] = SubRampLossKSVCR(H,A,Yk,c1,c2,n,epsilon,rho, Ins_t, Hinge_s)

N = size(Yk,2);
maxiters = 10;
Beta = zeros(n,N); 
bk = ones(1,N); 
i=1;
Z = zeros(n,N);  
U = zeros(n,N); 
val = 0;
while i<maxiters
    HB = H*Beta+ones(n,1)*bk;
    deta= -(Yk.*c2).*(Yk.*HB<Hinge_s & Yk~=0) + c1*(HB>Ins_t & Yk==0) - c1*(HB < - Ins_t & Yk==0);
    [Beta, bk,Z,U,history]= SubKSVOR(H,A,Yk,c1,c2,n,epsilon,rho,deta,Z,U) ; % Z,U are warmstarts
    i=i+1;
%     figure(2)
%     plot(1:length(history.objval), history.objval)
%     pause
     val = [val history.objval];
end

% figure(2)
% plot(1:length(val), val)
% pause
end


function [Beta, b,Z,U,history] = SubKSVOR(H,A,Yk,c1,c2,n,epsilon,rho,deta,Z,U)
%'traindata' is a training data matrix , each line is a sample vector
%'targets' is a label vector,should  start  from 1 to p

%Global constants and defaults
MAX_ITER = 50;
ABSTOL   = 1e-3;
RELTOL   = 1e-3;
v2 = sum(A,2);
i=1;
%  A = [ H + rho*eye(n),ones(n,1);ones(1,n) 0]\speye(n+1);
while i <=MAX_ITER
%    Beta = A(1:end-1,:)*[rho*(Z-U+deta);zeros(1,m)];
%    Alpha = [ H0 + rho*eye(n),yk;yk' 0 ]\[rho*(Z-U);0];
%    Alpha = Alpha(1:end-1);
    v1 = A*(Z-U);
    v = rho*sum(v1,1)/sum(v2); 
    Beta = v1 -v2*v - deta;
    Zold = Z; 
    Z= Beta+U;
    S = SoftThreshold(Z, epsilon/rho);
    for j=1:size(Yk,2)
         Z(:,j)= (Yk(:,j)~=0).*median([zeros(n,1),Z(:,j)+1/rho*Yk(:,j), c2*Yk(:,j)],2)+...
             (Yk(:,j)==0).*median([-c1*ones(n,1),S(:,j), c1*ones(n,1)],2);
    end
    %U- -update
    r = Beta- Z;
    U = U + r;
    val =0;
    for j = 1:size(Yk,2)
        val = val+ objective(H,Yk(:,j), Beta(:,j),  epsilon,deta(:,j)); 
    end
    history.objval(i)  =val;
    s = rho*(Z - Zold);
    history.r_norm(i) = norm(r);
    history.s_norm(i) = norm(s);
    history.eps_pri(i) = sqrt(n)*ABSTOL + RELTOL*max(norm(Beta), norm(Z));
    history.eps_dual(i)= sqrt(n)*ABSTOL + RELTOL*norm(rho*U);
    if  history.r_norm(i) < history.eps_pri(i) && ...
            history.s_norm(i) < history.eps_dual(i);
        break
    end
    i = i+1;
end
%    Beta = Z;

%  HB = -H* (Beta+deta); 
%   b0= (HB - epsilon).*(Yk==0& Beta>0 & Beta<c1)+ (HB + epsilon).*(Yk==0&Beta>-c1 & Beta<0)...
%       +(HB +Yk).*(Yk~=0&Yk.*Beta>0 & Yk.*Beta<c2);
%   b = mean(b0);
e = 1e-2;
for j = 1:size(Yk,2)
   bk_c1 =  -H(Yk(:,j)==0&(e<Beta(:,j)&Beta(:,j)<c1-e), :) * (Beta(:,j)+deta(:,j)) + epsilon;
   bk_c2 = - H(Yk(:,j)==0&(Beta(:,j)<e&Beta(:,j)>-c1-e), :) * (Beta(:,j)+deta(:,j)) - epsilon;
   bk_l =  -H(Yk(:,j)>0&(Beta(:,j)>e & Beta(:,j)<c2-e), :) * (Beta(:,j)+deta(:,j)) + 1;
   bk_r = - H(Yk(:,j)<0& (Beta(:,j)>-c2+e & Beta(:,j)<e), :) * (Beta(:,j)+deta(:,j))- 1;
   b(j) = mean([bk_c1;bk_c2;bk_l;bk_r]);
end
  Beta = Beta + deta; 
  
end

function Znew = SoftThreshold(Zold,kappa)

Znew = (Zold + kappa).*(Zold<- kappa)+(Zold - kappa).*(Zold> kappa) ;

end

function obj = objective(H,Yk, Beta,  epsilon,deta)
         obj = 0.5*(Beta+deta)'*H*(Beta+deta)+ epsilon*(Yk==0)'*abs(Beta) -(Yk~=0)'*(Yk.*Beta); 
end
