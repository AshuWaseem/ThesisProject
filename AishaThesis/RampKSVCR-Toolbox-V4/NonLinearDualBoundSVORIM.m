function par = NonLinearDualBoundSVORIM(traindata, targets, c1, c2, epsilon, rho, ker, sigma)
%'traindata' is a training data matrix , each line is a sample vector
%'targets' is a label vector,should  start  from 1 to p
model='EX';
% rho is the augmented Lagrangian parameter.

% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.

%Data preprocessing
[n, m] = size(traindata);
Lab=sort(unique(targets));
p=length(Lab); %  the number of total rank
l= zeros(1,p);
id={};
X=[];Y=[];
i=1;
Id = [];
while i<=p
    id{i}=find(targets==Lab(i));
    l(i)=length(id{i});
    X=[X;traindata(id{i},:)];
    Y=[Y;targets(id{i})];
    Id = [Id, id{i}];
    i=i+1;
end
[~,Id0]=sort(Id); 

 lc=cumsum(l);
 
 w = [];
 b = [];
 
 s = cell(1,p);
 r = cell(1,p);
 
K=Kernel(ker, X',X',sigma);
K=(K+K')/2;

nch =  nchoosek([1:p],2);
Code = zeros(p,size(nch,1));

for k =1:size(nch,1)
    Code(nch(k,:),k) = [-1,1];
    i = nch(k,1); j =nch(k,2);
    s{k} =lc(i)-l(i)+1:lc(i);
    r{k} = lc(j)-l(j)+1:lc(j);
    c{k} = [1:n];
    c{k}([lc(i)-l(i)+1:lc(i)  lc(j)-l(j)+1:lc(j)]) = [];
        Ak = X(c{k},:);
        Lk = X(s{k},:); 
        Rk =X(r{k},:);
        row=[c{k} c{k} s{k} r{k}];
        Hk = K(row,row);
        %    model= subSVOR(Ak,Hk,Lk,Rk,c1, c2, epsilon, rho);
        model = subSVOR_quadgrog(Ak, Hk, Lk,Rk,c1, c2, epsilon);
        
        P{k} = model.P;
        p0{k} = model.p;
        alpha{k} = model.alpha;
        alphax{k}= model.alphax;
        normw{k} = model.normw;
        time(k) = model.time;
        b(k,1) = model.b;
        Id1= [s{k} c{k} r{k}];
        SVs{k}= model.SVs(Id1);
end
   par.l= s;
   par.r = r;
   par.c= c;
   par.P = P;
   par.p = p0;
   par.alpha = alpha;
   par.alphax = alphax;
   par.normw = normw;
   par.time = time;
   par.b = b;
   par.X=X;
   par.maxtime = max(par.time);
    par.SVs =  SVs;
    par.Y=Y;
    par.Code = Code;
%    par.w = w;
%    par.b = b;

end


function par = subSVOR_quadgrog(Ak, H, Lk,Rk, c1, c2, epsilon)

t_start = tic;
%Global constants and defaults
QUIET    = 0;

m = size(Ak,2);
lk = size(Ak,1);
rk1 = size(Lk,1); 
rk2 =  size(Rk,1);
rk = rk1+rk2;
%ADMM solver
mP=2*lk +rk;    %dimension of Phi
mG=4*lk + 2*rk;  %dimension of Gamma
mU=mG+1;                                %dimension of U
mp1 = 1 : lk;        
mp2 = lk+1 : 2*lk;
mp3 = 2*lk+1: mP;

c= zeros(mU,1);
c([mp1 mp2]+1) = c1;
c(mp3+1) = c2;

q = ones(mP,1);
q(mp1) = epsilon;
q(mp2) = epsilon;
q(mp3) = -1;

p = ones(mP,1);
p(mp2) = -1;
p(mP-rk2+1:mP) =  -1;

% H = [Ak; -Ak; Bk{1}; -Bk{2}]*[Ak; -Ak; Bk{1}; -Bk{2}]';   %linear Kernel 
% Qk=[Ak; Ak; Lk; Rk];

% H= Kernel(ker, Qk',Qk',sigma);
% H=(H'+H)/2+1;
H0 = (H+1).*(p*p');

% % options = optimoptions('quadprog',...
% %     'Algorithm','interior-point-convex','Display','off');
options = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','MaxIter',200,'Display','off');
% % x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
A = []; b = []; f = q; Aeq =[]; beq = []; lb = zeros(mP,1); ub = c(2:mP+1); x0 = []; 
 P = quadprog(H0,f,A,b,Aeq,beq,lb,ub,x0,options);

 % diagnostics, reporting, termination checks
 
 par.P= P;
 par.p= p;
 par.alpha = P(mp1);
 par.alphax = P(mp2);
 P3=P(mp3);
 par.SVs = [P3(1:rk1);abs(P(mp1)-P(mp2));P3(rk1+1:rk1+rk2)];
%  par.normw = sqrt(P'*(H0.*(p'*p))*P);
  par.normw =1;
 bk =(p'*P);
 par.b =bk;

%  switch ker 
%      case 'linear'
%   par.w = [Ak; -Ak; Bk{1}; -Bk{2}]'*P;
%   b1 = Ak(P(mp1)~=0,:)* par.w-epsilon;   
%   b2 = Ak(P(mp2)~=0,:)* par.w+epsilon;
%   par.b = mean([b1;b2]);
%  end

 
 if  ~QUIET
       par.time = toc(t_start);
 end

end

function [par, history]  = subSVOR(Ak,H, Lk,Rk, c1, c2, epsilon, rho)
%'traindata' is a training data matrix , each line is a sample vector
%'targets' is a label vector,should  start  from 1 to p

% rho is the augmented Lagrangian parameter.

% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.

t_start = tic;

%Data preprocessing

%Global constants and defaults
QUIET    = 0;
MAX_ITER = 200;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;
ABSTOL   = 1e-6;
RELTOL   = 1e-3;

lk = size(Ak,1);
rk1 = size(Lk,1); 
rk2 =  size(Rk,1);
rk = rk1+rk2;

%ADMM solver
mP=2*lk +rk;    %dimension of Phi
mG=4*lk + 2*rk;  %dimension of Gamma
mU=mG;                                %dimension of U
P = zeros(mP,1);                       %Phi={ w,b, xi, xi*}
G = zeros(mG,1);                       %Gamma={eta,eta*,delta, phi, phi*}
U = zeros(mU,1);                       %U- -update

mp1 = 1 : lk;        
mp2 = lk+1 : 2*lk;
mp3 = 2*lk+1: mP;

c= zeros(mU,1);
c([mp1 mp2]) = c1;
c(mp3) = c2;

q = ones(mP,1);
q(mp1) = epsilon;
q(mp2) = epsilon;
q(mp3) = -1;

p = ones(mP,1);
p(mp2) = -1;
p(mP-rk2+1:mP) =  -1;

% H = [Ak; -Ak; Bk{1}; -Bk{2}]*[Ak; -Ak; Bk{1}; -Bk{2}]';   %linear Kernel 

%  Qk=[Ak; Ak; Lk; Rk];

% H0= Kernel(ker, Qk',Qk',sigma);%Kernel Matrix of Bound SVM 
% H = (H0+1).*(p*p');
% H0= Kernel(ker, Qk',Qk',sigma);%Kernel Matrix of Bound SVM 
H = (H+1).*(p*p');

k=1; 
while k <=MAX_ITER
    %Phi={ w,b, xi, xi*}-update
    V = U + B(mP,G) - c;
    br = - q - rho * AtX(mP,V);
    [P, niters] = cgsolve(H, br, rho);
    
    %Gamma={eta,eta*,delta, phi, phi*}-update with relaxation
    Gold = G;
    G = pos(Bt(mP,c-AX(P)-U));
    
    %U- -update
    r = AX(P) + B(mP,G) - c;
    U = U + r;
    %     history.objval(k)  = objective(H,P,q);
    s = rho*AtX(mP,B(mP,G - Gold));
    history.r_norm(k) = norm(r);
    history.s_norm(k) = norm(s);

    history.eps_pri(k) = sqrt(mU)*ABSTOL + RELTOL*max([norm(AX(P)), norm(B(mP,G)), norm(c)]);
    history.eps_dual(k)= sqrt(mP)*ABSTOL + RELTOL*norm(rho*AtX(mP,U));
    
    if  history.r_norm(k) < history.eps_pri(k) && ...
         history.s_norm(k) < history.eps_dual(k);
         break
    end
    k = k+1;
end
 
 if  ~QUIET
       par.time = toc(t_start);
 end
 
 par.P= P;
 par.p= p;
 par.alpha = P(mp1);
 par.alphax = P(mp2);
% par.normw = sqrt(P'*(H0.*(p'*p))*P);
par.normw=1;
bk =(p'*P);

par.b =bk;
 
 if  ~QUIET
       par.time = toc(t_start);
 end
 
end

% function obj = objective(H,P,q)
%     obj = 1/2 * vHv(H,P) + q'*P;
% end

function [x, niters] = cgsolve(H, b,rho,tol, maxiters)
% cgsolve : Solve Ax=b by conjugate gradients
%
% Given symmetric positive definite sparse matrix A and vector b, 
% this runs conjugate gradient to solve for x in A*x=b.
% It iterates until the residual norm is reduced by 10^-6,
% or for at most max(100,sqrt(n)) iterations

n = length(b);

if (nargin < 4) 
    tol = 1e-6;
    maxiters = max(100,sqrt(n));
elseif(nargin < 5)
    maxiters = max(100,sqrt(n));
end

normb = norm(b);
x = zeros(n,1);
r = b;
rtr = r'*r;
d = r;
niters = 0;
while sqrt(rtr)/normb > tol  &&  niters < maxiters
    niters = niters+1;
%     Ad = A*d; 
    Ad = AtAX(H, d,rho);
    alpha = rtr / (d'*Ad);
    x = x + alpha * d;
    r = r - alpha * Ad;
    rtrold = rtr;
    rtr = r'*r;
    beta = rtr / rtrold;
    d = r + beta * d;
end
end

%Ad = A*d
function Ad = AtAX(H, d,rho)
Ad = H*d +2* rho*d;
end

function F = AtX(mP,V)
F = V(1:mP)+V(mP+1:end);
end

function h = AX(P)
h = [P;P];
end


function h = vHv(H,d)
h = d'*(H*d) ;
end

function Bv= B(mP,v)
Bv(:,1) = [v(1:mP);-v(mP+1:end)];
end

function Btd = Bt(mP,d)
Btd = [d(1:mP);-d(mP+1:end)];
end

function A = pos(A)
A(A<0)=0;
end





