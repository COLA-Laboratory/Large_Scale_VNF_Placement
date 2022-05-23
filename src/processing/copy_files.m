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

src_folder = fullfile(root_path, 'data', 'Model');
out_folder = fullfile(root_path, 'data', 'VLG');

topologies = ["DCell", "FatTree", "LeafSpine"];
sizes = ["500", "1000", "2000", "4000", "8000", "16000"];
models = ["UtilisationModel"];

for topology = topologies
    for model = models
        for i = 0:29
            for size = sizes
                
                in_folder = fullfile(src_folder, topology, size, num2str(i), model);
                vlg_folder = fullfile(out_folder, topology, size, num2str(i));
                
                if ~exist(vlg_folder, 'dir')
                    mkdir(vlg_folder )
                end
               
                in_file = fullfile(in_folder, '*.objs');
                copyfile(in_file, vlg_folder );
                
                in_file = fullfile(in_folder, 'running_time.out');
                copyfile(in_file, vlg_folder );
            end
        end
    end
end