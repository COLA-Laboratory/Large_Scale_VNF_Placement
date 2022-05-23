clear
clc
format long g

% out_folder = 'D:\Research\NFV_AG_Journal\processed\cache_ub\';
out_folder = '/media/joebillingsley/Data/projects/NFV_AG_Journal/results/cache_ub/';

if ~exist(out_folder, 'dir')
    mkdir(out_folder);
end

utilisation = 0.95;
max_servers = 64001;

for table_size = 0.0:0.01:0.31
    all_feasible = [];
    
    for num_servers = 500:500:max_servers
        num_used = utilisation * num_servers;
        table_length = table_size * num_servers;
        
        p_fea = 1;
        
        for n = 0:num_used - 1
            p_placed = table_success(table_length, num_servers, n);
            p_fea = p_fea * p_placed;
        end
        
        all_feasible = [all_feasible; p_fea];
    end
    
    num_s_col = 500:500:max_servers;
    all_feasible = [num_s_col' all_feasible];
    
    file = fullfile(out_folder, [num2str(table_size), '.csv']);
    writematrix(all_feasible, file);
end

function p_placed = table_success(num_picks, num_servers, num_used)

if num_picks > num_used
    p_placed = 1;
elseif num_used == num_servers
    p_placed = 0;
else
    p_acc = 1;
    
    for i = 0:num_picks-1
        p_not_found = (num_used - i) / (num_servers - i);
        p_acc = p_acc * p_not_found;
    end
    
    p_placed = 1 - p_acc;
end

end