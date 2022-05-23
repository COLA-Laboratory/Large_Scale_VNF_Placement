close all
clear
clc
format long g

root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal';
% root_path = 'D:\Research\NFV_PlacementModel_Journal';

src_folder = fullfile(root_path, 'data');
out_folder = fullfile(root_path, 'results');

num_runs = 30;

algs = ["MOEAD", "PPLSD", "PPLS_Simple", "IBEA", "NSGA-II"];

folders = ["AlgorithmComparison", "AlgCmpReduced" "Initialization", "Model", "VLG"];

for i = 1 : length(folders)
    folder = fullfile(src_folder, folders{i});
    l_0 = dir(folder);
    
    for j = 3 : length(l_0)
        l_1_p = fullfile(l_0(j).folder, l_0(j).name);
        l_1 = dir(l_1_p);
        
        for k = 3 : length(l_1)
            l_2_p = fullfile(l_1(k).folder, l_1(k).name);
            
            for r = 0 : num_runs                
                for alg = algs
                    move_folders(src_folder, l_2_p, num2str(r), out_folder);
                end
            end
        end
    end
end

% Get list of all subfolders with obj files
function move_folders(src_folder, curr_folder, run, out_folder)

a_folder = fullfile(curr_folder, run);

all_folders = split(genpath(a_folder), ':');
% all_folders = split(genpath(a_folder), ';');

obj_folders = [];

for i = 1 : length(all_folders) - 1
    folder = all_folders{i};
    file_search = fullfile(folder, 'HV.out');
    items = dir(file_search);
    
    if ~isempty(items)
        if ~ismember(folder, obj_folders)
            obj_folders = [obj_folders, string(folder)];
        end
    end
end

for folder = obj_folders
    substr_a = erase(curr_folder, src_folder);
    substr_b = erase(folder, a_folder);
    
    results_folder = fullfile(out_folder, substr_a, substr_b, num2str(run));
    
    hv_file = 'HV.out';
    hv_in = fullfile(folder, hv_file);
    
    time_file = 'running_time.out';
    time_in = fullfile(folder, time_file);
    
    if not(isfolder(results_folder))
        mkdir(results_folder)
    end
    
    status_hv = copyfile(hv_in, results_folder);
    status_tm = copyfile(time_in, results_folder);
end

end
