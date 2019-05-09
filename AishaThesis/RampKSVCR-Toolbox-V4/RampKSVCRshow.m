clear

%% generate data
 
% prettySpiral = 0;
%  
% if ~prettySpiral
%     generate some random gaussian like data
%     rand('state', 0);
%     randn('state', 0);
%     N= 50;
%     D= 2;
%  
%     X1 = mgd(N, D, [4 3], [2 -1;-1 2]);
%     X2 = mgd(N, D, [1 1], [2 1;1 1]);
%     X3 = mgd(N, D, [3 -3], [1 0;0 4]);
%  
%     X= [X1; X2; X3];
%     X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
%     y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
%  
%     scatter(X(:,1), X(:,2), 20, y)
%  
% else
%     generate twirl data!
%  
%     N= 50;
%     t = linspace(0.5, 2*pi, N);
%     x = t.*cos(t);
%     y = t.*sin(t);
%  
%     t = linspace(0.5, 2*pi, N);
%     x2 = t.*cos(t+2);
%     y2 = t.*sin(t+2);
%  
%     t = linspace(0.5, 2*pi, N);
%     x3 = t.*cos(t+4);
%     y3 = t.*sin(t+4);
%  
%     X= [[x' y']; [x2' y2']; [x3' y3']];
%     X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
%     y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
%  
%     scatter(X(:,1), X(:,2), 20, y)
% end


load XandY;

%% classify
rho = 1;
c1 =15;
c2 =15;
epsilon = 0.3;
threshold = 0;
Ins_t = .2;
Hinge_s = -.4;

result=[];
%  ker = 'linear';
 ker = 'rbf';
sigma = 2/10;%sigma = 1/200;
model = RampLossKSVCR(X,y,c1,c2,epsilon,rho, Ins_t, Hinge_s, ker, sigma );
%f = TestPrecisionNonLinear(par,X, y,X, y, ker,epsilon,sigma);
L=unique(y);


%% Plot the figure
contour_level = [-epsilon,0, epsilon];
xrange = [-1.5 1.5];
yrange = [-1.5 1.5];
% step size for how finely you want to visualize the decision boundary.
inc = 0.005;
% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x1, x2] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x1);

xy = [x1(:) x2(:)]; % make (x,y) pairs as a bunch of row vectors.

% set up the domain over which you want to visualize the decision
% boundary
% d = [];
% for k=1:max(y)
%     par.normw{k}=1;
%     d(:,k) =  decisionfun(xy, par, X,y,k,epsilon, ker,sigma)';
% end
% [~,idx] = min(abs(d)/par.normw{k},[],2);
% nd=max(y);
nd = (max(y)*(max(y)-1)/2);
d = []; pred=zeros(size(xy,1),nd);
% for k=1:nd
%     par.normw{k}=1;
%     d(:,k) =  decisionfun(xy, par, X,y,k,epsilon, ker,sigma)';
% end
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
%pred(d<-threshold) = -1; pred(d >threshold) = 1;
pred = -(d<-threshold) + (d >threshold) ;
expLosses=zeros(size(pred,1),max(y));

for i=1:max(y),
    expLosses(:,i) = sum(pred == repmat(model.Code(i,:),size(pred,1),1),2);
end
[minVal,finalOutput] = max(expLosses,[],2);
idx = finalOutput;
idx = L(idx);
plt = 2; %1, just show the decison region with different colors; 2, show the decision hyperlane between class 1 and class 3
switch plt
    case 1
        % reshape the idx (which contains the class label) into an image.
        decisionmap = reshape(idx, image_size);
         imagesc(xrange,yrange,decisionmap);
        % plot the class training data.
        hold on;
        set(gca,'ydir','normal');
        cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
        colormap(cmap);
        plot(X(y==1,1), X(y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
        plot(X(y==2,1), X(y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
        plot(X(y==3,1), X(y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
        hold on;
        % title(sprintf('%d trees, Train time: %.2fs, Test time: %.2fs\n', opts.numTrees, timetrain, timetest));
    case 2
        %% show SVs
        r=2;
 %       C= c1*(y==r)+c2*(y~=r);
        color = {[.9 .3 .3],[.3 .9 .3],[.3 .3 .9]};
 %       SVs = (sum((abs(model.Beta)>1e-2),2)>0);
        SVs = (abs(model.Beta(:,r))>1e-2) ;
%        model.Beta =  model.Beta - model.deta;
%        SVs = (y<r).*(model.Beta(:,r)<-1e-3) +  (y==r).*(abs(model.Beta(:,r))>1e-4) + (y>r).*(model.Beta(:,r)>1e-3) ;
        for i=1:max(y)
            % show the SVs using biger marker
            plot(X(y==i&SVs==1,1),X(y==i&SVs==1,2), 'o', 'MarkerFaceColor', color{i}, 'MarkerEdgeColor','k'); 
            hold on
            % plot the points of not SVs
            plot(X(y==i&SVs~=1,1),X(y==i&SVs~=1,2), 'o', 'MarkerFaceColor', color{i}, 'MarkerEdgeColor',color{i});  
        end
        hold on;
        title(sprintf('Ratio of SVs is %.2f\n', mean(SVs)));
        color = {'r-','g-','b*','r.','go','b*'};
        color1 = {'r-','g--','b*','r.','go','b*'};
        contour_level1 = [-epsilon, 0, epsilon];
        contour_level2 = [-epsilon, 0, epsilon];
        contour_level0 = [-1,0,1];
        % for k = 1:nd
        for k=r
            decisionmapk = reshape(d(:,k), image_size);
            contour(x1,x2, decisionmapk, [-1 1], color1{k},'LineWidth',0.5);
            contour(x1,x2, decisionmapk, [contour_level(1) contour_level(1)], color{k});
            contour(x1,x2, decisionmapk, [contour_level(2) contour_level(2) ], color{k},'LineWidth',2);
            contour(x1,x2, decisionmapk, [contour_level(3) contour_level(3) ], color{k});
            contour(x1,x2, decisionmapk, [contour_level0(3) contour_level0(3) ], color1{k},'LineWidth',0.5);
        end
end