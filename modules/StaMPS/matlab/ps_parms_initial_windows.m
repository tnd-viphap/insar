%PS_PARMS_INITIAL Initialize parms to default values for PS processing
%
%   Andy Hooper, Jan 2008

parms=struct('Created',datetime("today"));

parms.small_baseline_flag='n'; % PS ifgs with single masters

function value = read_project_conf(filename, key)
    value = '';
    fid = fopen(filename, 'r');
    if fid == -1
        fprintf('Error: Could not open %s\n', filename);
        return;
    end
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, key)
            tokens = strsplit(line, '=');
            if length(tokens) == 2
                value = strtrim(tokens{2});
            end
            break;
        end
    end
    fclose(fid);
end

project_conf = "../../snap2stamps/bin/project.conf";
result_folder = strcat(read_project_conf(project_conf, "CURRENT_RESULT"), '/parms');

processor_file = strcat(read_project_conf(project_conf, "CURRENT_RESULT"), '/processor.txt');
if exist(processor_file,'file')~=2
    processor_file = ['..' filesep processor_file];
    if exist(processor_file,'file')~=2
        processor_file = ['..' filesep processor_file];
        if exist(processor_file,'file')~=2
            processor_file = ['..' filesep processor_file];
        end
    end
end
if exist(processor_file,'file')==2
    processor=textscan(processor_file,'%s');
    parms.insar_processor=strtrim(processor{1});
else
    parms.insar_processor='doris';
end

save(result_folder,'-struct','parms')


ps_parms_default_windows
