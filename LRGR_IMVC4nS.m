function [P_ini,Z_ini,obj] = LRGR_IMVC4nS(X,ind_folds,S_ini,Giv,W_ini,numClust,lambda1,lambda2,max_iter)

Sum_S = 0;
for iv = 1:length(S_ini)
    Z_ini{iv} = Giv{iv}*S_ini{iv}*Giv{iv}';
    Sum_S = Sum_S+Z_ini{iv};
end
Sum_S = (Sum_S+Sum_S')*0.5;
LS = diag(sum(Sum_S)) - Sum_S;
[P_ini,~] = eigs(LS,numClust,'sa');
for iv = 1:length(Z_ini)
    Qiv{iv} = Z_ini{iv}'*P_ini;
    W_lambda{iv} = 1./(W_ini{iv}+lambda1);
    Eiv{iv} = EuDist2(X{iv}',X{iv}',0);
end
Nv = length(S_ini);
Siv = S_ini;
clear S_ini
for iter = 1:max_iter
     % ---------- Siv ------ %
    for iv = 1:Nv
        ind_0 = find(ind_folds(:,iv)==0);
        linshi_B = Z_ini{iv};
        linshi_B(ind_0,:) = [];
        linshi_B(:,ind_0) = [];                
        linshi_S = linshi_B-0.5*lambda2*Eiv{iv};
        linshi_S2 = zeros(size(linshi_S));
        for in = 1:size(linshi_S,2)
            idx = [1:size(linshi_S,2)];
            idx(in) = [];    
            linshi_S2(idx,in) = EProjSimplex_new(linshi_S(idx,in)); 
        end
        Siv{iv} = linshi_S2;        
    end  
    % ----------- Z_ini --------- %
    for iv = 1:Nv
        linshi_GSG = Giv{iv}*Siv{iv}*Giv{iv}';
        linshi_Z = (linshi_GSG+lambda1*P_ini*Qiv{iv}').*W_lambda{iv};
        linshi_Z2 = zeros(size(linshi_Z));
        for in = 1:size(linshi_Z,2)
            idx = [1:size(linshi_Z,2)];
            idx(in) = [];
            linshi_Z2(idx,in) = EProjSimplex_new(linshi_Z(idx,in));      
        end
        Z_ini{iv} = linshi_Z2;
    end       
    
    % -------- P_ini ------- %
    linshi_H = 0;
    for iv = 1:Nv
        linshi_H = linshi_H+Z_ini{iv}*Qiv{iv};
    end
    linshi_H(isnan(linshi_H)) = 0;
    linshi_H(isinf(linshi_H)) = 0;
    [linshi_U,~,linshi_V] = svd(linshi_H,'econ');
    linshi_U(isnan(linshi_U)) = 0;
    linshi_U(isinf(linshi_U)) = 0;   
    linshi_V(isnan(linshi_V)) = 0;
    linshi_V(isinf(linshi_V)) = 0;
    P_ini = linshi_U*linshi_V';
    
    % --------- Q{iv} ---------- %
    for iv = 1:Nv
        Qiv{iv} = Z_ini{iv}'*P_ini;
    end    
    % ------- obj ------ %
    linshi_obj = 0;
    for iv = 1:Nv
        linshi_obj = linshi_obj+norm((Z_ini{iv}-Giv{iv}*Siv{iv}*Giv{iv}').*W_ini{iv},'fro')^2+lambda1*norm(Z_ini{iv}-P_ini*Qiv{iv}','fro')^2+lambda2*sum(sum(Eiv{iv}.*Siv{iv}));
    end
    obj(iter) = linshi_obj;
    if iter>5 & abs(obj(iter)-obj(iter-1)) < 1e-4
        iter
        break;
    end       
    
end