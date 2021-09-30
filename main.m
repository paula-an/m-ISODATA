clc
clear
close all
%
% Operating System
if strcmp(computer, 'PCWIN64')
    folder_root = '';
else
    folder_root = '/';
end
rng(1);  % Seed for reproducibility
%
%% Simulation parameters
p = struct(...
    'SimName', 'mISODATA_demand-2wind',...  % Simulation name used to save the scenarios in a .csv file
    'FileSerie', [folder_root 'data/demand-2wind.mat'],...  % .csv file containing historical series
    'ResultsPath', [folder_root 'results/'],...  % Path to results folder
    'Display', 'off', ...  % No displays
    'PlotFlag', false,...  % Flag to plot 2-dimensional scenarios - false (default)
    'SaveFlag', true);  % Flag to save scenarios data as a .csv file
%
%% Selecting series
% Series from Merrick paper are 1 demand, 3 wind and 7 solar
% so, to extract 1 demand and 2 wind historical series we need to extract
% series from 1 to 3

%% Calling m-ISODATA
[groups, centr, prob, exitflag] = misodata(p);
%
if p.SaveFlag
    disp(['The results were saved in ' p.ResultsPath p.SimName '.mat'])
end