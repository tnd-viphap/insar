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
  
  % Open the file for reading
  bperp_values = ps.bperp;
  fid = fopen('../../../modules/snap2stamps/bin/project.conf');
  if fid == -1
      error('Cannot open the file.');
  end
  
  % Read the file line by line
  max_bperp = '';
  while ~feof(fid)
      line = fgetl(fid);
      if startsWith(line, 'MAX_PERP')
          parts = strsplit(line, '=');
          if numel(parts) > 1
              max_bperp = strtrim(parts{2}); % Trim spaces from the value
              break; % Stop after finding the first match
          end
      end
  end
  
  % Close the file
  fclose(fid);
  
  % Display the extracted value
  if isempty(max_bperp)
      WARNING('MAX_PERP not found in the file.');
  else
      fprintf('Extracted value: %s\n', max_bperp);
  end
  max_bperp = str2double(max_bperp);
  % Find interferograms with |bperp| >= 150
  toremove_bperp_indices = find(abs(bperp_values) >= max_bperp);
  
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
  high_std_indices = find(ifg_std > mean_std);
  
  fprintf('Mean standard deviation: %3.2f degrees\n', mean_std);
  fprintf('Interferograms with standard deviation > mean:\n');
  if strcmpi(small_baseline_flag,'y')
      for i = high_std_indices'
          fprintf('%3d %s_%s %3.2f\n',i,datestr(ps.ifgday(i,1)),datestr(ps.ifgday(i,2)),ifg_std(i))
      end
  else
      for i = high_std_indices'
          fprintf('%3d %s %3.2f\n',i,datestr(ps.day(i)),ifg_std(i))
      end
  end
  fprintf('\n')
  
  % Append high bperp indices to drop_ifg_index
  drop_ifg_index = unique([high_std_indices; toremove_bperp_indices]);
  
  % Convert drop_ifg_index to a list and print
  fprintf('drop_ifg_index values:\n');
  list_str = '';
  for i = 1:length(drop_ifg_index)
      if i == 1
          list_str = num2str(drop_ifg_index(i));
      else
          list_str = [list_str, ' ', num2str(drop_ifg_index(i))];
      end
  end
  fprintf('%s\n\n', list_str);
  
  % Set the updated drop_ifg_index parameter
  setparm('drop_ifg_index', list_str);
  fprintf('Set drop_ifg_index parameter with %d interferograms (including those with |bperp| ≤ 150)\n', length(drop_ifg_index));
  
  save(ifgstdname,'ifg_std'); 
  end