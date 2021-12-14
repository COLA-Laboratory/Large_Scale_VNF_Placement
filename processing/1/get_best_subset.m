addpath('.');

close all
clear
clc
format long g

%% Parameters
num_objectives = 3;
pop_size = 128;

nadir = [5.805159975122, 0.396598975478901, 206.308893730829];
utopian = [0.468939967725214, 1.74614755077407e-08, 56.0044519224511];
ref = zeros(1, num_objectives) + 1.000001;

runs = 30;

% root_path = 'D:\Research\NFV_AG_Journal\';
root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal/';
src_folder = fullfile(root_path, 'data', 'AlgorithmComparison');
out_folder = fullfile(root_path, 'data', 'AlgCmpReduced');

cd(src_folder);

file_search = dir(fullfile('**', '*.objs'));

for i = 1:length(file_search)
    file = file_search(i);
    objs = get_objs(file.folder, file.name);
    
    objs = filter_NDS(objs, objs);
    objs = unique(objs, 'rows');
    
    if size(objs, 1) > pop_size
        new_objs =  zeros(128, 3);
        
        scaled_objs = scale_objectives(file.name, objs, nadir, utopian);
        [idx, C] = kmeans(scaled_objs, 128);
        
        for j = 1:length(C)
            
            best = 0;
            min_dist = 10000000000;
            for k = 1:length(scaled_objs)
                dist = norm(scaled_objs(k, :) - C(j, :));
                
                if dist < min_dist
                    min_dist = dist;
                    best = k;
                end
            end
            
            new_objs(j, :) = objs(best, :);
        end
    else 
        new_objs = objs;
    end 
        
    test_directory = erase(file.folder, src_folder);
    out_directory = fullfile(out_folder, test_directory);
    
    if ~exist(out_directory, 'dir')
        mkdir(out_directory)
    end
    
    writematrix(new_objs, fullfile(out_directory, file.name), 'FileType', 'text');
end

function objectives = get_objs(folder, file_name)
full_path = fullfile(folder, file_name);

% Manually read CSV to handle 'Infeasible' values
fid = fopen(full_path);
lines = {};
tline = fgetl(fid);

objectives = [];
row = 1;

while ischar(tline)
    if contains(tline, 'Infeasible')
        tline = fgetl(fid);
        continue
    end
    
    try
        s = split(tline, ',');
        
        if length(s) ~= 3 || count(tline, '.') ~= 3
            tline = fgetl(fid);
            continue
        end
        
        objectives(row,1) = str2double(s{1});
        objectives(row,2) = str2double(s{2});
        objectives(row,3) = str2double(s{3});
        
        if objectives(row, 1) > 1000
            c = 2;
        end
        
    catch
        tline = fgetl(fid);
        continue
    end
    
    tline = fgetl(fid);
    row = row + 1;
end
fclose(fid);

end

function scaled_objs = scale_objectives(file_name, objectives, nadir, utopian)

num_services_idx = strfind(file_name, '_');
num_services = extractBetween(file_name, 1, num_services_idx - 1);
num_services = str2double(num_services{1});

scaled_objs = objectives;

scaled_objs(:, 3) = scaled_objs(:, 3) / num_services;

scaled_objs = (scaled_objs - utopian) ./ (nadir - utopian);

end