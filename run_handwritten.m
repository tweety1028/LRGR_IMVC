clear all
clc

Dataname = 'handwritten-5view';

pre_percentDel = [0.3,0.5,0.7];
para_k = 0;

percentDel = 0.3

para_k = 13
lambda1 = 0.01;
lambda2 = 0.01;
            acc = []
            rand('seed',5857)
f = 1;
Datanfold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datanfold);
ind_folds = folds{f}; 
clear new_folds KnnGraph
truthF = Y+1;
clear Y
numClust = length(unique(truthF));
if size(X{1},2)~=length(truthF)
    for iv = 1:length(X)
        X{iv} = X{iv}';
    end
end

for iv = 1:length(X)
    X1 = X{iv};
    X1 = NormalizeFea(X1,0);  % 0-column 1-row
    ind_0 = find(ind_folds(:,iv) == 0);
    ind_1 = find(ind_folds(:,iv) == 1);

    linshi_W = ones(length(truthF),length(truthF));
    linshi_W(:,ind_0) = 0;
    linshi_W(ind_0,:) = 0;
    W_ini{iv} = linshi_W;

    % ---------- knn ----------- %
    X1(:,ind_0) = [];
    options = [];
    options.NeighborMode = 'KNN';
    options.k = para_k;
%                     options.t = para_t;
    options.WeightMode = 'Binary';      % Binary  HeatKernel  Cosine
    Z1 = full(constructW(X1',options));
    Z1 = Z1- diag(diag(Z1));
    linshi_W = diag(ind_folds(:,iv));
    linshi_W(:,ind_0) = [];
    S_ini{iv} = max(Z1,Z1');
    Giv{iv} = linshi_W;
    Yiv{iv} = X1;
    clear Z1 linshi_W
end
clear X
X = Yiv;
clear Yiv

max_iter = 100;
[P_ini,Z_ini,obj] = LRGR_IMVC4nS(X,ind_folds,S_ini,Giv,W_ini,numClust,lambda1,lambda2,max_iter);

Fng = P_ini;
Fng = NormalizeFea(Fng,1);
Fng(isnan(Fng)) = 0;
%pre_labels    = kmeans(Fng,numClust,'emptyaction','singleton','replicates',20,'display','off');
%result_cluster = ClusteringMeasure(truthF, pre_labels)*100  


Sum_Z = 0;
nv = length(Z_ini);
for iv = 1:nv
    Sum_Z = Sum_Z+Z_ini{iv};
end
Sum_Z = (1/nv)*Sum_Z;
Sum_Z = (Sum_Z+Sum_Z')*0.5;

Dd = diag(sqrt(1./(sum(Sum_Z,1)+eps)));
An = Dd*Sum_Z*Dd;
An(isnan(An)) = 0;
An(isinf(An)) = 0;
try
    [Fng2, ~] = eigs(An,numClust,'largestabs');
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eigs:ARPACKroutineErrorMinus14'))
        opts.tol = 1e-3;
        [Fng2, ~] = eigs(An,numClust,'largestabs',opts.tol);
    else
        rethrow(ME);
    end
end
Fng2(isnan(Fng2))=0;
Fng2 = NormalizeFea(Fng2,1);
if sum(isnan(Fng2(:))) == 0
    pre_labels2 = kmeans(real(Fng2),numClust,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
    result_SC = ClusteringMeasure(truthF, pre_labels2)*100
end            

                  
                    

