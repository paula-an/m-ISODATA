function [groups, centr, prob, output] = misodata(p)
% MISODATA is an unsupervised clustering algorithm to capture scenarios 
% from historical series.
%
%   [ ... ] = MISODATA(p) partitions the points in the N-by-D data matrix
%    in the selected .csv file into clusters. Where N is the number of data 
%    points and D the dimension of the historical series. MISODATA captures
%    the scenarios from historical series according to p. p is a structure 
%    whose fields are:
%       'FileSerie' - Name of .csv file containing the historical series.
%       'dMin' - the minimum allowed distance between the centroids.
%       	Default value: 0.04
%       'max_std' - the maximum allowed standard deviation of scenarios. 
%       	Default value: 0.55*p.dMin 
%       'pMin' - scenario's minimum allowed probability of occurence. 
%       	Default value: 1e-10
%       'SimName' - Name of the simulation.
%       	Default value: 'none'
%       'Niter' - Maximum number of iterations.
%       	Default value: 100
%       'ResultsPath' - Path for saving results.
%       	Default value: []
%       'PlotFlag' - Flag to plot two-dimensional graphs.
%       	Default value: false
%       'SaveFlag' - Flag to save results.
%       	Default value: false
%       'nCluMax' - maximum number of clusters.
%       	Default value: internally computed.
%       'nClu0' - initial number of clusters
%       	Default value: 1.
%       'UseDCF' - Use of the dimensional correction factor
%           Default value: true
%       'Display' - off, final or iter
%       	Default value: iter.
%
%   [GROUPS, CENTR, PROB] = MISODATA(p) GROUPS returns the historical 
%    series data points location in the CENTR K-by-N centroids matrix. Where
%    K is the number of clusters obtained. PROB returns the probability of 
%    occurrence of the scenario associated with each centroid.
%
%   [GROUPS, CENTR, PROB, EXITFLAG] = MISODATA(p) returns an EXITFLAG
%    that describes the exit condition. Possible values of EXITFLAG and the
%    corresponding exit conditions are:
%
%       1  Iterative proccess converged.
%   	0  Number of iterations reached.
%
%   See also KMEANS, CLUSTER.

tic  % Measuring elapsed time
%% Reading the data
if ~any(strcmp(fieldnames(p), 'FileSerie'))
    % error
    error('ERROR: field p.FileSerie is mandatory in p structure!!!')
end
%
load(p.FileSerie, 'data')
[Nobs, nser] = size(data);  % Number of data points and number of series
%
% Normalizing data
base_data_1 = min(data, [], 1);
base_data_2 = max(data, [], 1);
for iser = 1:nser
    data(:, iser) = (data(:, iser)-base_data_1(iser))/(base_data_2(iser)-base_data_1(iser));
end
%
% Calculating unique observations in the series
data_unique = unique(data,'rows');
[Nobs_unique, ~] = size(data_unique);  % Number of unique observations
%
%% Parameters treatment
% Use of dimensional correction factor
if ~any(strcmp(fieldnames(p), 'UseDCF'))
    % default value
    p.UseDCF = true;
end
%
% Mminimum allowed distance between the centroids
if ~any(strcmp(fieldnames(p), 'dMin'))
    % default value
    p.dMin = 0.04;
end
%
% Applying the Dimensional correction factor
if p.UseDCF
    p.dMin = p.dMin*sqrt(nser);
end
%
% Maximum allowed standard deviation of scenarios
if ~any(strcmp(fieldnames(p), 'max_std'))
    % default value
    p.max_std = p.dMin*0.55;
elseif p.UseDCF
    % Applying the Dimensional correction factor
    p.max_std = p.max_std*sqrt(nser);  % Dimensional correction factor
end
%
% Scenario's minimum allowed probability of occurence
if ~any(strcmp(fieldnames(p), 'pMin'))
    % default value
    p.pMin = 1e-10;
end
%
% Simulation name
if ~any(strcmp(fieldnames(p), 'SimName'))
    % default value
    p.SimName = 'none';
end
%
% Number of iterations
if ~any(strcmp(fieldnames(p), 'Niter'))
    % default value
    p.Niter = 100;
end
%
% Path for results
if ~any(strcmp(fieldnames(p), 'ResultsPath'))
    % default value
    p.ResultsPath = [];  % Current folder
end
%
% Flag to plot two-dimensional graphs
if ~any(strcmp(fieldnames(p), 'PlotFlag'))
    % default value
    p.PlotFlag = false;
end
%
% Flag to save scenarios data as a .csv file
if ~any(strcmp(fieldnames(p), 'SaveFlag'))
    % default value
    p.SaveFlag = false;
end
%
% Flag to display results
if ~any(strcmp(fieldnames(p), 'Display'))
    % default value
    p.Display = 'iter';
end
%
% Maximum number of clusters.
if ~any(strcmp(fieldnames(p), 'nCluMax'))
    % default value
    p.nCluMax = round(1/p.pMin);
    if p.nCluMax > Nobs_unique
        p.nCluMax = Nobs_unique;
    end
else
    if p.nCluMax > Nobs_unique
        p.nCluMax = Nobs_unique;
        disp(['nCluMax greater than number of unique observations and was seted to ' num2str(Nobs_uniqe) '!!!'])
    end
end
%
% Initial number of clusters.
if ~any(strcmp(fieldnames(p), 'nClu0'))
    % default value
    p.nClu0 = 1;
else
    if p.nClu0 > p.nCluMax
        p.nClu0 = p.nCluMax;
        disp(['nClu0 greater than maximum number of clusters and was seted to ' num2str(p.nCluMax) '!!!'])
    elseif p.nClu0 < 1
        p.nClu0 = 1;
        disp('nClu0 smaller than 1 and was seted to 1!!!')
    end
end
%
% output structure
output = struct();
output.exitflag = 0;
output.iter = 0;
%
%% m-ISODATA main program
%
% Minimum number of data per cluster
nmin = round(p.pMin*Nobs);
if nmin == 0
    nmin = 1;
end
%
% Extracting initial clusters (as forecasted)
nClu = p.nClu0;
centrF = data_unique(randperm(Nobs_unique,nClu)', :);
%
% Historic of number of clusters
nClu_hist = zeros(p.Niter, 1);
%
% Stop criteria flags and counters
countSTOP = 0;
flagOSCI = 0;
nClu_old_s = nClu;  % For stop criteria purposes
nClu_old_m = nClu;  % For stop criteria purposes
iter = 0;
hist_desv = 1;
iter_odd = false;
%
% main loop
if strcmp(p.Display, 'iter')
    disp(' ')
end
while true
    %
    nClu_old = nClu;  % For stop criteria purposes
    %
    if countSTOP == 0
        %
        % Grouping data into clusters
        [groups, clucount] = GroupData(data, centrF);
        %
        % Erasing empty clusters
        if any(clucount==0)
            empty_idx = find(clucount==0)';
            c = 0;
            for idx = empty_idx
                groups(groups>idx-c) = groups(groups>idx-c)-1;
                c = c+1;
            end
            nClu = nClu-length(empty_idx);
            clucount(empty_idx) = [];
        end
        %
        % Updating stop criterion
        if nClu == nClu_old
            countSTOP = 1;
        else
            nClu_old = nClu;
            countSTOP = 0;
        end
    end
    
    if strcmp(p.Display, 'iter')
        disp(['iter: ' num2str(iter) ' | nClu: ' num2str(nClu)]);
    end
    
    % Verifying stop criteria
    if iter == p.Niter || countSTOP == 3 || (flagOSCI >= 4 && ~iter_odd) || (hist_desv < 0.05 && ~iter_odd)
        %
        %
        % Calculating centroids
        centr = CalcCentr(groups, data, clucount);
        if p.PlotFlag
            if nser == 2
                plot_groups(nClu, nser, data, groups, p)
            else
                disp('Warning: "plot_flag = true" only available for two-dimensional series')
            end
        end
        %
        % Defining exitflag
        output.iter = iter;
        if iter == p.Niter
            output.exitflag = 0;  % Número de iterações atingido
        else
            output.exitflag = 1;  % Parada por estabilização
        end
        %
        % Exit m-ISODATA program
        break
    end
    %
    % Updating iteration
    iter = iter+1;
    iter_odd = rem(iter, 2) ~= 0;
    nClu_hist(iter) = nClu;
    if iter>=10
        hist_desv = std(nClu_hist(iter-9:iter))/mean(nClu_hist(iter-9:iter));
    end
    %
    % Calculating centroids
    centr = CalcCentr(groups, data, clucount);
    %
    % Calculating distances between centroids and data and obtaining medoids
    [clu_desv, clu_bevec] = CalcDesv(data, groups, clucount);
    %
    % Verifying if iteration is odd or even
    if iter_odd && iter ~= p.Niter
        %
        %  Accessing split function
        [centrF, nClu] = Split(clu_desv, clu_bevec, groups, centr, nmin, p);
        %
        % Updating stop criterion
        if nClu == nClu_old_s
            flagOSCI = flagOSCI+1;
        else
            nClu_old_s = nClu;
            flagOSCI = 0;
        end
    else
        %
        %  Accessing merge function
        [centrF, nClu] = Merge(centr, nClu, p);
        %
        % Updating stop criterion
        if nClu == nClu_old_m
            flagOSCI = flagOSCI+1;
        else
            nClu_old_m = nClu;
            flagOSCI = 0;
        end
    end
    %
    % Updating stop criterion
    if nClu == nClu_old
        countSTOP = countSTOP+1;
    else
        countSTOP = 0;
    end
    %
end
%
%% Scenarios output
if strcmp(p.Display, 'iter')  || strcmp(p.Display, 'final')
    disp(['iter: ' num2str(iter)]);
end
%
% Returning data ranges
for iser = 1:nser
    centr(:, iser) = centr(:, iser)*(base_data_2(iser)-base_data_1(iser))+base_data_1(iser);
end
%
% Calculating the probabilities of occurrence
prob = clucount/Nobs;
%
% Saving data in the .csv file
if p.SaveFlag
    %
    % Scenarios table
    tbl = [centr prob];
    %
    % Sorting scenarios into decreasing probabilities
    [pB,I] = sort(tbl(:, nser+1),'descend');
    tblB = [tbl(I, 1:nser) pB];
    writetable(table(tblB),[p.ResultsPath p.SimName '.csv'],'WriteVariableNames', false)
end
%
%
end
%
%% Function to group data into clusters
function [groups, clucount] = GroupData(data, centrF)
[Nobs, ~] = size(data);
[nClu, ~] = size(centrF);
groups = zeros(Nobs, 1);
clucount = zeros(nClu, 1);
%
for iobs = 1:Nobs
    dmax = inf;
    for iclu = 1:nClu
        d = norm(data(iobs, :)-centrF(iclu, :));
        if d < dmax
            dmax = d;
            kclu = iclu;
        end
    end
    groups(iobs) = kclu;
    clucount(kclu) = clucount(kclu)+1;
end
%
%
end
%
%% Function to calculate distances between centroids and data
function [clu_desv, clu_bevec] = CalcDesv(data, groups, clucount)
[~, dim] = size(data);
nClu = length(clucount);
%
clu_desv = zeros(nClu, 1);
clu_bevec = zeros(nClu, dim);
%
% Calculating standard deviations
for iclu = 1:nClu
    if clucount(iclu) > 0
        idata = data(groups==iclu, :);
        %
        if clucount(iclu) > 1
            %
            % Covariance matrix
            data_cov = cov(idata, 1);
            %
            % Maximum calculated standard deviation of current cluster 
            clu_desv(iclu) = sqrt(max(diag(data_cov)));
            %
            % Eigenvectors and eigenvalues of covariance matrix
            [Evec, Eval] = eig(data_cov);
            %
            % Finding the eigenvector associated with the greatest eigenvalue
            max_Eval_idx = diag(Eval) == max(diag(Eval));
            if sum(max_Eval_idx) > 1
                new_Eval_idx = find(max_Eval_idx);
                max_Eval_idx = new_Eval_idx(1);
            end
            clu_bevec(iclu, :) = Evec(:, max_Eval_idx')';
        else
            clu_desv(iclu) = 0;
            clu_bevec(iclu, :) = zeros(1, dim);
        end
    end
end
%
%
end
%
%% Calculating centroids
function centr = CalcCentr(groups, data, clucount)
[~, dim] = size(data);
nClu = length(clucount);
centr = zeros(nClu, dim);
for iclu = 1:nClu
    idata = data(groups==iclu, :);
    if clucount(iclu) == 1
        centr(iclu, :) = idata;
    else
        centr(iclu, :) = mean(idata);
    end
end
%
%
end
%
%% Merge function
function [centrF, nClu] = Merge(centr, nClu, p)
merge_clu = zeros(round(nClu/2), 2);
merge_cont = 0;
is_merge = zeros(nClu, 1);
for iclu = 1:nClu
    if is_merge(iclu)
        continue
    end
    %
    dmax = inf;
    for jclu = 1:nClu
        if jclu == iclu
            continue
        end
        %
        d = norm(centr(iclu, :)-centr(jclu, :));
        if d<dmax
            dmax = d;
            jmerge = jclu;
        end
    end
    if dmax < p.dMin
        % Merging clusters
        if is_merge(jmerge) == 0 && jmerge > iclu
            merge_cont = merge_cont+1;
            merge_clu(merge_cont, :) = [iclu jmerge];
            is_merge(jmerge) = 1;
        else
            is_merge(iclu) = 1;
        end
    end
end
%
% Merging clusters
for imrg = 1:merge_cont
    iclu = merge_clu(imrg, 1);
    jclu = merge_clu(imrg, 2);
    centr(iclu, :) = (centr(iclu, :)+centr(jclu, :))/2;
end
%
centr(logical(is_merge), :) = [];
%
% Forecasted centroids
centrF = centr;
nClu = nClu-sum(is_merge);
%
%
end
%
%% Split function
function [centrF, nClu] = Split(clu_desv, clu_bevec, groups, centr, nmin, p)
[nClu, dim] = size(centr);
nc_counter = 0;  % Counter of new clusters
nc_max = p.nCluMax-nClu;  % Maximum number of new clusters
if nc_max > 2*nClu
    nc_max = nClu;
end
clusters_tmp = zeros(nc_max, dim);

if nc_max > 0
    idx_grupos = unique(groups);
    for iclu = 1:nClu
        idx = idx_grupos(iclu);
        Nparobs = sum(groups==idx);
        if clu_desv(iclu) > p.max_std && Nparobs > 2*nmin
            nc_counter = nc_counter+1;
            %
            % New clusters
            clusters_tmp(nc_counter, :) = centr(iclu, :)+clu_desv(iclu)*clu_bevec(iclu, :);
            centr(iclu, :) = centr(iclu, :)-clu_desv(iclu)*clu_bevec(iclu, :);
        end
        %
        if nc_counter == nc_max
            break
        end
    end
end
%
% Forecasted centroids
centrF = [centr; clusters_tmp(1:nc_counter, :)];
nClu = nClu+nc_counter;
%
%
end
%