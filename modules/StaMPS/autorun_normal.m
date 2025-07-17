% Read project config file
function value = read_conf_value(filename)
    % Open the file for reading
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open the file.');
    end
    
    % Read the file line by line
    value = '';
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, 'CURRENT_RESULT')
            parts = strsplit(line, '=');
            if numel(parts) > 1
                value = strtrim(parts{2}); % Trim spaces from the value
                break; % Stop after finding the first match
            end
        end
    end
    
    % Close the file
    fclose(fid);
    
    % Display the extracted value
    if isempty(value)
        WARNING('CURRENT_RESULT not found in the file.');
    else
        fprintf('Extracted value: %s\n', value);
    end
end

% Running StaMPS
function process_patch_folders(current_result)
    % Get list of all folders in the current directory
    files = dir();
    folders = files([files.isdir]);
    
    % Loop through each folder that contains 'PATCH_' in its name
    for i = 1:length(folders)
        folder_name = folders(i).name;
        if contains(folder_name, 'PATCH_')
            chdir(strcat(current_result, '/', folder_name))
            fprintf('Processing folder: %s\n', folder_name);
            % Stamps 1: 
            fprintf("Step 1: Load initial gamma\n");
            getparm();
            setparm('n_cores', 30);
            setparm('plot_scatterer_size', 30);
            stamps(1,1);
            ps_info();
            fprintf("\n")
            % Stamps 2:
            fprintf("Step 2: Calculate coherence\n");
            setparm('max_topo_err', 10)
            setparm('gamma_change_convergence', 0.005);
            setparm('filter_grid_size', 50);
            stamps(2,2);
            fprintf('\n');
            % Stamps 3:
            fprintf("Step 3: Select PS\n");
            setparm('select_method', 'PERCENT');
            setparm('percent_rand', 80);
            stamps(3,3);
            fprintf('\n');
            % Stamps 4:
            fprintf("Step 4: Weed PS\n");
            setparm('weed_zero_elevation', 'n');
            setparm('weed_neighbours', 'n');
            stamps(4,4);
            fprintf('\n');
            % Stamps 5:
            fprintf("Step 5: Phase correction\n");
            setparm('merge_resample_size', 10);
            stamps(5,5, 'y');
            fprintf('\n');
            fprintf('Step 6: Phase unwrapping');
            setparm('unwrap_time_win', 24);
            setparm('unwrap_gold_n_win', 32);
            setparm('unwrap_grid_size', 10);
            stamps(6,6, 'y');
            fprintf('\n');
            % Stamps 7:
            aps_linear();
            fprintf("Step 7: Phase unwrapping correction\n");
            setparm('scla_deramp', 'y');
            stamps(7, 7, 'y');
            stamps(6, 6, 'y');
            stamps(7, 7, 'y');
            fprintf('\n');
            % Stamps 8:
            fprintf("Step 8: Atmospheric correction\n");
            setparm('scn_time_win', 30);
            stamps(8,8, 'y');
            fprintf('\n');
            % Export to csv
            fprintf("Exporting results...");
            chdir(current_result);
        end
    end
end

try
    clear();
    current_result = read_conf_value('../../config/project.conf');
    chdir(current_result);

    % Change priority of StaMPS
    folder_to_move = '../../modules/StaMPS/matlab';
    if contains(path, folder_to_move)
        rmpath(folder_to_move);
    end

    % Add it back to the top of the path
    addpath(folder_to_move, '-begin');

    process_patch_folders(current_result);
catch ME
    fprintf(ME.message);
    exit();
end