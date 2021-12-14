addpath('.');

close all
clear
clc
format long g

%% Parameters
num_objectives = 3;

runs = 30;

root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal';
% root_path = 'D:\Research\NFV_AG_Journal';

src_folder = fullfile(root_path, 'results');
out_folder = fullfile(root_path, 'processed', 'aggregate');

% Get list of all subfolders with obj files
all_folders = split(genpath(src_folder), ':');
% all_folders = split(genpath(src_folder), ';');
obj_folders = [];

for i = 1 : length(all_folders) - 1
    folder = all_folders{i};
    file_search = fullfile(folder, 'HV.out');
    items = dir(file_search);
    
    if ~isempty(items)
        seps = strfind(folder, filesep);
        folder = folder(1: seps(end)-1);
        
        if ~ismember(folder, obj_folders)
            obj_folders = [obj_folders, string(folder)];
        end
    end
end

for folder = obj_folders
    output = [];
    
    file_search = fullfile(folder, '*', 'HV.out');
    hv_files = dir(file_search);
    
    agg_hv = [];
    
    for i = 1 : length(hv_files)
        hv_file = hv_files(i);
        file = fullfile(hv_file.folder, hv_file.name);
        hvs = csvread(file);
        
        %         if i == 1
        %             output(:, 1) = hvs(:, 1);
        %         end
        
        agg_hv = [agg_hv, hvs(:, 2)];
    end
    
    output(:, 2) = mean(agg_hv, 2);
    output(:, 3) = std(agg_hv, 0, 2);
    output(:, 4) = min(agg_hv, [], 2);
    output(:, 5) = prctile(agg_hv, 25, 2);
    output(:, 6) = median(agg_hv, 2);
    output(:, 7) = prctile(agg_hv, 75, 2);
    output(:, 8) = max(agg_hv, [], 2);
    
    output = [["evaluations","mean","stdev","min","lq","median","uq","max"]; output];
    
    dest_folder = fullfile(out_folder, erase(folder, src_folder));
    if ~exist(dest_folder, 'dir')
        mkdir(dest_folder);
    end
    
    out_file = fullfile(dest_folder, 'aggregate.csv');
    
    writematrix(output, out_file);
end