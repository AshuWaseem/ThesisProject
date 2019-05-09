function f = decisionfun(xnew,model, X,y,k,epsilon, ker,sigma)

 mod='dual';
%  mod='prim';
%f = (-(model.p{k}.*model.P{k})'*Kernel(ker,Qk',xnew',sigma)- model.b(k))/model.normw{k};
switch mod
    case 'dual'
        Ak = model.X(model.c{k},:);
        Lk = model.X(model.l{k},:);
        Rk = model.X(model.r{k},:);
        
        Qk=[Ak; Ak; Lk; Rk];
        nT=size(xnew,1);
        if nT >1000
            for j=1:nT/1000
                xnewk=xnew(1000*(j-1)+1:1000*j,:);
                f(1000*(j-1)+1:1000*j) = -(model.p{k}.*model.P{k})'*Kernel(ker,Qk',xnewk',sigma)- model.b(k);
            end
            xnewk=xnew(1000*j+1:nT,:);
            f(1000*j+1:nT)=-(model.p{k}.*model.P{k})'*Kernel(ker,Qk',xnewk',sigma)- model.b(k);
        else
            f = -(model.p{k}.*model.P{k})'*Kernel(ker,Qk',xnew',sigma)- model.b(k);
        end
case 'prim'
    nT=size(xnew,1);
    if nT >1000
        for j=1:nT/1000
            xnewk=xnew(1000*(j-1)+1:1000*j,:);
            f(1000*(j-1)+1:1000*j) = model.P{k}'*Kernel(ker,model.X',xnewk',sigma)- model.b(k);
        end
        xnewk=xnew(1000*j+1:nT,:);
        f(1000*j+1:nT)=model.P{k}'*Kernel(ker,model.X',xnewk',sigma)- model.b(k);
    else
        f = model.P{k}'*Kernel(ker,model.X',xnew',sigma)- model.b(k);
    end
end

end