addpath('.');

close all
clear
clc
format long g

%% Parameters
num_objectives = 3;

runs = 30;

root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal/results/VLG/';
% root_path = 'D:\Research\NFV_PlacementModel_Journal';

topologies = ["DCell", "FatTree", "LeafSpine"];
dc_size = ["500", "1000", "2000", "4000", "8000", "16000", "32000", "64000"];

for i = 1:length(topologies)
    topology = topologies(i);
    
    fprintf('%s\n', topology);
    
    for j = 1:length(dc_size)
        size = dc_size(j);
        
        folder = fullfile(root_path, topology, size);
        
        file_search = fullfile(folder, '*', 'HV.out');
        hv_files = dir(file_search);
        
        agg_hv = [];
        
        for l = 1 : length(hv_files)
            hv_file = hv_files(l);
            file = fullfile(hv_file.folder, hv_file.name);
            hvs = csvread(file);
            
            agg_hv = [agg_hv, hvs(:, 2)];
        end
        
        [sig_test, h] = ranksum(agg_hv(1, :), agg_hv(2, :), 'tail', 'right');
        
        fprintf('%f, %i \n', sig_test, h);
    end
end