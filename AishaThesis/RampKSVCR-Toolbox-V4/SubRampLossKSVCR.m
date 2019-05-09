% function [Beta, bk, Alpha, objval,history] = SubRampLossKSVCR(H,A,Yk,c1,c2,n,epsilon,rho, Ins_t, Hinge_s)
% 
% maxiters = 100;
% deta = zeros(n,1);
% Beta = zeros(n,1); 
% bk = 1; 
% i=1;
% Z = zeros(n,1);  
% U = zeros(n,1); 
% 
% while i<maxiters
%     
%     deta(Yk==0) = -c1*(-H(Yk==0,:)*Beta+bk>Ins_t) + c1*(-H(Yk==0,:)*Beta+bk<-Ins_t);
%     deta(Yk~=0)= - c2*(Yk(Yk~=0).*(-H(Yk~=0,:)*Beta+bk)<Hinge_s);
%     [Beta, bk,Alpha, Z,U, val]= SubKSVOR(H,A,Yk,c1,c2,n,epsilon,rho,deta,Z,U) ; % Z,U are warmstarts
%     history{i}=val;
%     objval(i)=objective(H,Yk, Beta,  epsilon, deta);
%     i=i+1;
% end
% 
% end
% 
% 
% function [Beta, bk,Alpha, Z,U, objval] = SubKSVOR(H,A,Y,c1,c2,n,epsilon,rho,deta,Z,U)
% %'traindata' is a training data matrix , each line is a sample vector
% %'targets' is a label vector,should  start  from 1 to p
% 
% %Global constants and defaults
% MAX_ITER = 100;
% ABSTOL   = 1e-6;
% RELTOL   = 1e-3;
% 
% 
% i=1; 
% yk = 1*(Y<=0) -(Y>0);
% %  A = [ H + rho*eye(n),ones(n,1);ones(1,n) 0]\speye(n+1);
% while i <=MAX_ITER
%     Beta = A(1:end-1,:)*[rho*yk.*(Z-U+deta);0];
%     Alpha = yk.*Beta - deta;
% %    Alpha = [ H0 + rho*eye(n),yk;yk' 0 ]\[rho*(Z-U);0];
% %    Alpha = Alpha(1:end-1);
%     Zold = Z; 
%     Z= Alpha+U;
%     Z(Y~=0) = median([zeros(sum(Y~=0),1) , Z(Y~=0)+1/rho, c2*ones(sum(Y~=0),1) ],2);
%     Sk = SoftThreshold(Z(Y==0), epsilon/rho);
%     Z(Y==0) = median([-c1*ones(sum(Y==0),1),Sk, c1*ones(sum(Y==0),1)],2);
%     %U- -update
%     r = Alpha- Z;
%     U = U + r;
%     objval(i)  =  objective(H,Y, Beta,  epsilon,deta);
%     s = rho*(Z - Zold);
%     history.r_norm(i) = norm(r);
%     history.s_norm(i) = norm(s);
%     history.eps_pri(i) = sqrt(n)*ABSTOL + RELTOL*max(norm(Alpha), norm(Z));
%     history.eps_dual(i)= sqrt(n)*ABSTOL + RELTOL*norm(rho*U);
%     if  history.r_norm(i) < history.eps_pri(i) && ...
%             history.s_norm(i) < history.eps_dual(i);
%         break
%     end
%     i = i+1;
% end
%    bk_c1 =  H(Y==0&Alpha>0, :) * Beta + epsilon;
%    bk_c2 =  H(Y==0&Alpha<0, :) * Beta  - epsilon;
%    bk_l =  H(Y<0&Alpha>0, :) * Beta - 1;
%    bk_r =  H(Y>0&Alpha>0, :) * Beta + 1;
%    bk = mean([bk_c1;bk_c2;bk_l;bk_r]);
% end
% 
% function Znew = SoftThreshold(Zold,kappa)
% 
% Znew = (Zold + kappa).*(Zold<- kappa)+(Zold - kappa).*(Zold> kappa);
% 
% end
% 
% function obj = objective(H,Yk, Beta,  epsilon,deta)
%     yk = 1*(Yk<=0) -(Yk>0);
%     Alpha = yk.*Beta - deta;
%     obj = 0.5*Beta'*H*Beta+ epsilon* (Yk==0)'*abs(Alpha) -(Yk~=0)'*Alpha;
% end



%%
function [Beta, bk] = SubRampLossKSVCR(H,A,Yk,c1,c2,n,epsilon,rho, Ins_t, Hinge_s)

N = size(Yk,2);
maxiters = 100;
Beta = zeros(n,N); 
bk = ones(1,N); 
i=1;
Z = zeros(n,N);  
U = zeros(n,N); 

while i<maxiters
    HB = H*Beta+ones(n,1)*bk;
    deta= -(Yk.*c2).*(Yk.*HB<Hinge_s & Yk~=0) + c1*(HB>Ins_t & Yk==0) - c1*(HB < - Ins_t & Yk==0);
    [Beta, bk,Z,U]= SubKSVOR(H,A,Yk,c1,c2,n,epsilon,rho,deta,Z,U) ; % Z,U are warmstarts
    i=i+1;
end

end


function [Beta, b,Z,U] = SubKSVOR(H,A,Yk,c1,c2,n,epsilon,rho,deta,Z,U)
%'traindata' is a training data matrix , each line is a sample vector
%'targets' is a label vector,should  start  from 1 to p

%Global constants and defaults
MAX_ITER = 100;
ABSTOL   = 1e-6;
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
    history.objval(i)  =  objective(H,Yk, Beta,  epsilon); 
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
 Beta = Beta + deta;
 HB = -H* Beta; 
  b0= (HB - epsilon).*(Yk==0& Beta>0 & Beta<c1)+ (HB + epsilon).*(Yk==0&Beta>-c1 & Beta<0)...
      +(HB +Yk).*(Yk~=0&Yk.*Beta>0 & Yk.*Beta<c2);
  b = mean(b0);

end

function Znew = SoftThreshold(Zold,kappa)

Znew = (Zold + kappa).*(Zold<- kappa)+(Zold - kappa).*(Zold> kappa) ;

end
%%

function obj = objective(H,Yk, Beta,  epsilon,deta)
    yk = 1*(Yk<=0) -(Yk>0);
    Alpha = yk.*Beta - deta;
    obj = 0.5*Beta'*H*Beta+ epsilon* (Yk==0)'*abs(Alpha) -(Yk~=0)'*Alpha;
end