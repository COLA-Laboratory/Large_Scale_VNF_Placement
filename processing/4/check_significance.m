addpath('.');

close all
clear
clc
format long g

%% Parameters
num_objectives = 3;

runs = 30;

root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal/results/AlgorithmComparison/';
% root_path = 'D:\Research\NFV_PlacementModel_Journal';

topologies = ["DCell", "FatTree", "LeafSpine"];
dc_size = ["500", "1000", "2000", "4000", "8000", "16000"];

algs = ["PPLS_Simple", "PPLSD", "NSGA-II", "MOEAD", "IBEA"];

for i = 1:length(topologies)
    topology = topologies(i);
    
    fprintf('%s\n', topology);
    
    for j = 1:length(dc_size)
        size = dc_size(j);
        
        for k = 1:length(algs)
            alg = algs(k);
            
            folder = fullfile(root_path, topology, size, alg);
            
            file_search = fullfile(folder, '*', 'HV.out');
            hv_files = dir(file_search);
            
            agg_hv = [];
            
            for l = 1 : length(hv_files)
                hv_file = hv_files(l);
                file = fullfile(hv_file.folder, hv_file.name);
                hvs = csvread(file);
                
                agg_hv = [agg_hv, hvs(end, 2)];
            end
            
            out(:, k) = agg_hv;
        end
        
        fprintf('%s\n', size);
        for an = 2:length(algs)            
            [sig_test, h] = ranksum(out(:, 1), out(:, an), 'tail', 'right');
            
            fprintf('(%s,%s): %f, %i \n', algs(1), algs(an), sig_test, h);
        end
        fprintf('\n');
    end
end