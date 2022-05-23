root_path = '/media/joebillingsley/Data/projects/NFV_AG_Journal/';
src_folder = fullfile(root_path, 'data', 'Model', 'DCell', '500');

objective_first = dir([src_folder '/**/UtilisationModel/50_11904.objs']);
objective_second = dir([src_folder '/**/0.5/50_11968.objs']);

for i = 1:length(objective_first)
    file_one = objective_first(i);
    file_two = objective_second(i);
    
    objs_one = get_objs(file_one.folder, file_one.name);  
    objs_two = get_objs(file_two.folder, file_two.name);  
    
    objs_one = unique(objs_one, 'rows');
    objs_two = unique(objs_two, 'rows');
    
    subplot(1,2,1);
    scatter3(objs_one(:, 1), objs_one(:, 2), objs_one(:, 3));
    xlabel('Latency'); 
    ylabel('Packet loss');
    zlabel('Energy');
    title('Utilisation');
    
    view(90,0);
    
    subplot(1,2,2);
    scatter3(objs_two(:, 1), objs_two(:, 2), objs_two(:, 3));
    xlabel('Latency'); 
    ylabel('Packet Loss');
    zlabel('Energy');
    title('Accurate');
    
    view(90,0);
    
    pause(10)
end

function objectives = get_objs(folder, file_name)
num_services_idx = strfind(file_name, '_');
num_services = extractBetween(file_name, 1, num_services_idx - 1);
num_services = str2double(num_services{1});

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
    catch
        tline = fgetl(fid);
        continue
    end
    
    tline = fgetl(fid);
    row = row + 1;
end
fclose(fid);

if ~isempty(objectives)
    objectives(:, 3) = objectives(:, 3) / num_services;
end

end