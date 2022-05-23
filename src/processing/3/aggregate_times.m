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
    file_search = fullfile(folder, 'running_time.out');
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
    
    file_search = fullfile(folder, '*', 'running_time.out');
    time_files = dir(file_search);
    
    agg_times = [];
    
    for i = 1 : length(time_files)
        time_file = time_files(i);
        file = fullfile(time_file.folder, time_file.name);
        
        % Temporary bodge to deal with empty files
        try 
            time = csvread(file);
        catch
            continue
        end
        
        %         if i == 1
        %             output(:, 1) = hvs(:, 1);
        %         end
        
        agg_times = [agg_times, time];
    end
    
    output(:, 2) = mean(agg_times, 2);
    output(:, 3) = std(agg_times, 0, 2);
    output(:, 4) = min(agg_times, [], 2);
    output(:, 5) = prctile(agg_times, 25, 2);
    output(:, 6) = median(agg_times, 2);
    output(:, 7) = prctile(agg_times, 75, 2);
    output(:, 8) = max(agg_times, [], 2);
    
    output = [["evaluations","mean","stdev","min","lq","median","uq","max"]; output];
    
    dest_folder = fullfile(out_folder, erase(folder, src_folder));
    if ~exist(dest_folder, 'dir')
        mkdir(dest_folder);
    end
    
    out_file = fullfile(dest_folder, 'aggregate_times.csv');
    
    writematrix(output, out_file);
end