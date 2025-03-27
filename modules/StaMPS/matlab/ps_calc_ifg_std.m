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
      if startsWith(line, 'MAX_PERP')
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

function []=PS_calc_ifg_std
% PS_CALC_IFG_STD() calculate std for each ifg
%
%   Andy Hooper, June 2006
%
%   ======================================================
%   09/2006 AH: small baselines added 
%   02/2010 AH: More informative info displayed
%   ======================================================

fprintf('\nEstimating noise standard deviation (degrees)...\n')

small_baseline_flag=getparm('small_baseline_flag');

load psver
psname=['ps',num2str(psver)];
phname=['ph',num2str(psver)];
pmname=['pm',num2str(psver)];
bpname=['bp',num2str(psver)];
ifgstdname=['ifgstd',num2str(psver)];

ps=load(psname);
pm=load(pmname);
bp=load(bpname);

if exist([phname,'.mat'],'file')
    phin=load(phname);
    ph=phin.ph;
    clear phin
else
    ph=ps.ph;
end

if strcmpi(small_baseline_flag,'y')
  bperp_values = ps.bperp;
else
  bperp_values = [ps.bperp(1:ps.master_ix-1), 0, ps.bperp(ps.master_ix:end)];
end
max_bperp = read_conf_value('../../snap2stamps/bin/project.conf');
max_bperp = str2double(max_bperp);

n_ps=length(ps.xy);
master_ix=sum(ps.master_day>ps.day)+1;

if strcmpi(small_baseline_flag,'y')
    ph_diff=angle(ph.*conj(pm.ph_patch).*exp(-j*(repmat(pm.K_ps,1,ps.n_ifg).*bp.bperp_mat)));    
else
    bperp_mat=[bp.bperp_mat(:,1:ps.master_ix-1),zeros(ps.n_ps,1,'single'),bp.bperp_mat(:,ps.master_ix:end)];
    ph_patch=[pm.ph_patch(:,1:master_ix-1),ones(n_ps,1),pm.ph_patch(:,master_ix:end)];
    ph_diff=angle(ph.*conj(ph_patch).*exp(-j*(repmat(pm.K_ps,1,ps.n_ifg).*bperp_mat+repmat(pm.C_ps,1,ps.n_ifg))));
end

ifg_std=[sqrt(sum(ph_diff.^2)/n_ps)*180/pi]';
if strcmpi(small_baseline_flag,'y')
  for i=1:ps.n_ifg
    fprintf('%3d %s_%s %3.2f\n',i,datestr(ps.ifgday(i,1)),datestr(ps.ifgday(i,2)),ifg_std(i))
  end
else
  for i=1:ps.n_ifg
    fprintf('%3d %s %3.2f\n',i,datestr(ps.day(i)),ifg_std(i))
  end
end
fprintf('\n')

% Calculate mean standard deviation and find interferograms with std <= mean
mean_std = mean(ifg_std);
low_std_indices = find(ifg_std <= mean_std);
high_std_indices = find(ifg_std > mean_std);

fprintf('Mean standard deviation: %3.2f degrees\n', mean_std);
fprintf('Interferograms with standard deviation <= mean:\n');
if strcmpi(small_baseline_flag,'y')
    for i = low_std_indices'
        fprintf('%3d %s_%s %3.2f\n',i,datestr(ps.ifgday(i,1)),datestr(ps.ifgday(i,2)),ifg_std(i))
    end
else
    for i = low_std_indices'
        fprintf('%3d %s %3.2f\n',i,datestr(ps.day(i)),ifg_std(i))
    end
end
fprintf('\n')

% Find interferograms with |bperp| >= 150
toremove_bperp_indices = find(abs(bperp_values) >= max_bperp);

% Append low bperp indices to drop_ifg_index
drop_ifg_index = unique([low_std_indices; toremove_bperp_indices]);

% Set the updated drop_ifg_index parameter
setparm('drop_ifg_index', drop_ifg_index);
fprintf('Set drop_ifg_index parameter with %d interferograms (including those with |bperp| ≤ 150)\n', length(drop_ifg_index));

save(ifgstdname,'ifg_std'); 
end
    
    

