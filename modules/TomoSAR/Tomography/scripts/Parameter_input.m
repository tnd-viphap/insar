function value = read_conf_value(filename, key)
    % Open the file for reading
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open the file.');
    end
    
    % Read the file line by line
    value = '';
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, key)
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

%%%%%%%%%%%%%%%%%%%%%%%%% For SHP analysis  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% CalWin - based on azimuth and range spacing to have rough square window in real world 
% row x col is expected to much greater than number of images to better invert covariance matrix.  
% i.e., for 100 Sentinel-1 images, 7x25 or 9x35 is a good one.
CalWin = [7 25]; % - [row col]  

% These follow parameters can be good for most areas.
Alpha = 0.05;  % significance level. 0.05 for 5% significance.
BroNumthre = 20; % less than 20 is likely PS
Cohthre = 0.25; % threshold to select DS in which its phase variance is mostly less than 20 degree.    
Cohthre_slc_filt = 0.05; % less than 0.05 is mostly water 

%%%%%%%%%%%%%%%%%%%%%%%%% For ComSAR analysis  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% PSDSInSAR is heavy on memory use due to full covariance estimation. A rough approximation for
% PSDSInSAR RAM requirement is 1.5*Nslc*Nslc*Nline*Nwidth/2.7e8 (GB)
% ComSAR is much friendly Big Data processing. A rough approximation for
% ComSAR RAM requirement is 0.3*Nslc*Nslc*Nline*Nwidth/2.7e8 (GB)
% i.e., 200 images of 500x2000 size, 220 GB is for PSDS, but for ComSAR it requires only 45 GB.
COMSAR_fetch = read_conf_value('../../../../config/project.conf', 'COMSAR');
if strcmpi(COMSAR_fetch, 'true') || strcmp(COMSAR_fetch, '1')
    ComSAR_flag = true;
    fprintf("-> ComSAR Enabled")
else
    ComSAR_flag = false;
    fprintf("-> PSDS Enabled")
end
miniStackSize = read_conf_value('../../../../config/project.conf', 'MINISTACK');
miniStackSize = str2num(miniStackSize);
Unified_fetch = read_conf_value('../../../../config/project.conf', 'UNIFIED');
if strcmpi(Unified_fetch, 'true') || strcmp(Unified_fetch, '1')
    Unified_flag = true;
    fprintf("-> Unified ComSAR Enabled")
else
    Unified_flag = false;
    fprintf("-> Unified ComSAR Disabled")
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

InSAR_processor = 'snap'; % snap or isce 
switch InSAR_processor
    case 'snap' % 
        % Define path - expect the SNAP export STAMPS structure
        % check out a tutorial here https://youtu.be/HzvvJoDE8ic 
        InSAR_path = read_conf_value('../../../../config/project.conf', 'CURRENT_RESULT');
        reference_date = split(InSAR_path, '/');
        size_split=length(reference_date);
        reference_date = split(reference_date{size_split}, '_');
        reference_date = reference_date{2};

        file_par = [InSAR_path,'/rslc/',reference_date,'.rslc.par'];
        par_getline = regexp(fileread(file_par),['[^\n\r]+','zimuth_lines','[^\n\r]+'],'match');
        nlines  = str2num([par_getline{1}(15:end)]);

        slcstack = ImgRead([InSAR_path,'/rslc'],'rslc',nlines,'cpxfloat32');
        interfstack = ImgRead([InSAR_path,'/diff0'],'diff',nlines,'cpxfloat32');       
    case 'isce'
        % Define path - expect the 'make_single_reference_stack_isce' structure
        % check out a tutorial  - to be prepare 
        InSAR_path = 'X:\Accra\isce\INSAR_20180103';
        reference_date = '20180103';
        
        nlines = load([InSAR_path,'/len.txt']);        
        slcslist = load([InSAR_path,'/slcs.list']);    
        [slcstack,  interfstack] = ImgRead_isce(InSAR_path,nlines,str2num(reference_date),slcslist);      
    otherwise
        disp('not yet support')
end

warning('off','all')
