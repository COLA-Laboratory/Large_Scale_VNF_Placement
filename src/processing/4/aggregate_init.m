addpath('.');

close all
clear
clc
format long g

%% Parameters
runs = 30;

root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal';
% root_path = 'D:\Research\NFV_AG_Journal';

src_folder = fullfile(root_path, 'processed', 'aggregate', 'Initialization');
out_folder = fullfile(root_path, 'processed', 'initialization');

topologies = ["DCell", "FatTree", "LeafSpine"];
sizes = ["500", "1000", "2000", "4000", "8000", "16000", "32000", "64000"];
init_strats = ["isa", "sa"];

for topology = topologies
    for init_strat = init_strats
        
        hv_out = [];
        for i = 1:length(sizes)
            size = sizes(i);
            folder = fullfile(src_folder, topology, size, init_strat);
            
            hv_file = fullfile(folder, "aggregate.csv");
            
            hv_agg = get_aggregate(hv_file);
            
            hv_out(i, :) = [size, hv_agg];
        end
        
        % Add headers
        header = ["size", "lq_diff","median","uq_diff"];
        
        hv_out = [header; hv_out];
        
        % Create files
        dest_folder = fullfile(out_folder, topology, init_strat);
        if ~exist(dest_folder, 'dir')
            mkdir(dest_folder);
        end
        
        hv_out_file = fullfile(dest_folder, "hv.csv");
        
        writematrix(hv_out, hv_out_file);
    end
end

function aggregate = get_aggregate(file)

contents = readmatrix(file);

median = contents(end, 6);
lq = contents(end, 5);
uq = contents(end, 7);

aggregate = [median - lq, median, uq - median];

end