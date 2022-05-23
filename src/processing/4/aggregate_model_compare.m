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

src_folder = fullfile(root_path, 'processed', 'aggregate', 'Model');
out_folder = fullfile(root_path, 'processed', 'model', 'comparison');

topologies = ["DCell", "FatTree", "LeafSpine"];
sizes = ["500", "1000", "2000", "4000", "8000", "16000"];
models = ["0.5","inf","ConstantModel","MM1Model","PercLenModel","UtilisationModel","UtilLatencyModel"];

for topology = topologies
    for model = models
        row = 1;
        
        hv_out = [];
        times_out = [];
        for size = sizes
            folder = fullfile(src_folder, topology, size, model);
            
            hv_file = fullfile(folder, "aggregate.csv");
            times_file = fullfile(folder, "aggregate_times.csv");
            
            hv_agg = get_aggregate(hv_file);
            times_agg = get_aggregate(times_file);
            
            hv_out(row, :) = [size, hv_agg];
            times_out(row, :) = [size, times_agg];
            
            row = row + 1;
        end
        
        % Add headers
        header = ["size", "lq_diff","median","uq_diff"];
        
        hv_out = [header; hv_out];
        times_out = [header; times_out];
        
        % Create files
        dest_folder = fullfile(out_folder, topology, model);
        if ~exist(dest_folder, 'dir')
            mkdir(dest_folder);
        end
        
        hv_out_file = fullfile(dest_folder, "hv.csv");
        time_out_file = fullfile(dest_folder, "time.csv");
        
        writematrix(hv_out, hv_out_file);
        writematrix(times_out, time_out_file);
    end
end

function aggregate = get_aggregate(file)

contents = readmatrix(file);

median = contents(end, 6);
lq = contents(end, 5);
uq = contents(end, 7);

% Fix logarithmic graph issues
if median == 0
    median = 0.1;
end

aggregate = [median - lq, median, uq - median];

end