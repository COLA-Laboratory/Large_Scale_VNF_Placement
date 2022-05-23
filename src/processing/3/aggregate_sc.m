clear
clc
format long g

% base_folder = 'D:\Research\NFV_AG_Journal\';
base_folder = '/media/joebillingsley/Data/projects/NFV_AG_Journal';

src_folder = fullfile(base_folder, 'data', 'SolutionConstruction');
out_folder = fullfile(base_folder, 'processed', 'solution_construction');

num_servers = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000];
min_size = zeros(length(num_servers), 1);

topologies = ["FatTree", "DCell", "LeafSpine"];
for topo = topologies
    
    bfs_rows = [num_servers', min_size];
    sbfs_rows = [num_servers', min_size];
    
    for i = 1:length(num_servers)
        
        bfs_done = false;
        sbfs_done = false;
        
        ns = num_servers(i);
        for table_size = 0:100
            file = fullfile(src_folder, topo, num2str(ns), [num2str(table_size), '.csv']);
            data = readmatrix(file);
            
            if data(1) > 99900 && ~sbfs_done
                sbfs_rows(i, 2) = table_size;
                sbfs_done = true;
            end
            
            if data(2) > 99900 && ~bfs_done
                bfs_rows(i, 2) = table_size;
                bfs_done = true;
            end
            
            if bfs_done && sbfs_done
                break;
            end
        end
    end
    
    bfs_file = fullfile(out_folder, 'BFS', append(topo, '.csv'));
    sbfs_file = fullfile(out_folder, 'SBFS', append(topo, '.csv'));
    
    writematrix(bfs_rows, bfs_file);
    writematrix(sbfs_rows, sbfs_file);
end



