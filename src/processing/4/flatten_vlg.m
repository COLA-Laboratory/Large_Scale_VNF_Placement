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

src_folder = fullfile(root_path, 'processed', 'aggregate', 'VLG');
out_folder = fullfile(root_path, 'processed', 'vlg');

topologies = ["DCell", "FatTree", "LeafSpine"];
sizes = ["500", "1000", "2000", "4000", "8000", "16000", "32000", "64000"];

for topology = topologies
    
    % Add headers
    header = ["size", "lq_diff","median","uq_diff"];
    
    init_hv_out = header;
    final_hv_out = header;
    times_out = header;
    
    for size = sizes
        
        hv_file = fullfile(src_folder, topology, size, "aggregate.csv");
        times_file = fullfile(src_folder, topology, size, "aggregate_times.csv");
        
        [init_agg, final_agg] = get_aggregate(hv_file);
        times_agg = get_aggregate(times_file);
        
        init_hv_out = [init_hv_out; size, init_agg];
        final_hv_out = [final_hv_out; size, final_agg];
        
        times_out = [times_out; size, times_agg];
    end
    
    % Create files
    dest_folder = fullfile(out_folder, topology);
    if ~exist(dest_folder, 'dir')
        mkdir(dest_folder);
    end
    
    init_hv_out_file = fullfile(dest_folder, "init_hv.csv");
    final_hv_out_file = fullfile(dest_folder, "final_hv.csv");
    time_out_file = fullfile(dest_folder, "time.csv");
    
    writematrix(init_hv_out, init_hv_out_file);
    writematrix(final_hv_out, final_hv_out_file);
    writematrix(times_out, time_out_file);
end

function [initial, final] = get_aggregate(file)

contents = readmatrix(file);

init_lq = contents(1, 5);
init_median = contents(1, 6);
init_uq = contents(1, 7);

fin_lq = contents(end, 5);
fin_median = contents(end, 6);
fin_uq = contents(end, 7);

initial = [init_median - init_lq, init_median, init_uq - init_median];
final = [fin_median - fin_lq, fin_median, fin_uq - fin_median];

end