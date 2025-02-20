function ps_export_gis(filename,lon_rg,lat_rg,ortho)
    if ~exist('ps_plot_ts_v-dao.mat')
    ps_plot('v-dao', 'a_linear', 'ts'); %Time series variabls
    close(gcf)
    end
    if ~exist('ps_plot_v-dao.mat')
    ps_plot('v-dao', 'a_linear', -1); %Mean Velocity variabls
    close(gcf)
    end

    load('ps_plot_v-dao','ph_disp') %dimension [#ps 1]
    load('ps_plot_ts_v-dao','ph_mm')   %dimension [#ps #ifg]
    load('ps_plot_ts_v-dao','day')   %dimension [#ifg 1]
    load('ps2','lonlat')  %dimension [#ps 2]
    load('lonlat_err','lonlat_err')
    load('dem_err','dem_err')
    load('ps2','xy')  %%dimension [#ps 3] id despoints PS
    load ('hgt2','hgt') %dimension [#ps 1]
    load('pm2','coh_ps') % [#ps 1]






    %decouper selon une zone
    if ~isempty(lon_rg) && ~isempty(lat_rg)
        ix = lonlat(:,1) >= min(lon_rg) & lonlat(:,1) <= max(lon_rg) & ...
            lonlat(:,2) >= min(lat_rg) & lonlat(:,2) <= max(lat_rg);
    else
        ix = true(size(lonlat, 1), 1); % If lon_rg or lat_rg is empty, select all points
    end
    xy=xy(ix,:);
    lonlat=lonlat(ix,:);
    lonlat_err=lonlat_err(ix,:);
    hgt=hgt(ix,:);
    dem_err=dem_err(ix,:);
    coh_ps=coh_ps(ix,:);
    ph_disp=ph_disp(ix,:);
    ph_mm=ph_mm(ix,:);



    %DEM ORTHO CORRECTION %%%%%%%%%%%%%%%
    if ~isempty(ortho) && strcmp(ortho,'ortho')
      lonlat=lonlat-lonlat_err; 
      hgt=hgt+dem_err; 
    end

    %Les variables a exporter 
    ids=sprintfc('PS_%d',xy(:,1)); %[#ps 1] %Cell of STRING
    clear xy

    clear lonlat_err
    lon=sprintfc('%.8f',lonlat(:,1)); %[#ps 1] %Cell of STRING
    lat=sprintfc('%.8f',lonlat(:,2)); %[#ps 1] %Cell of STRING
    clear lonlat
    clear dem_err
    hgt=sprintfc('%.1f',hgt); %[#ps 1] %Cell of STRING
    coh=sprintfc('%.1f',coh_ps); %[#ps 1] %Cell of STRING
    clear coh_ps
    vlos=sprintfc('%.1f',ph_disp); %[#ps 1] %Cell of STRING
    clear ph_disp
    days=sprintfc('D%s',datestr(day,'yyyymmdd')); %[1 #ifg] %Cell of STRING
    clear day
    d_mm0=bsxfun(@minus, ph_mm, ph_mm(:,1)); %First Measure as reference
    clear ph_mm
    d_mm=num2cell(d_mm0);
    %d_mm=sprintfc('%.1f',d_mm0) %Crach
    clear d_mm0
    
    %dem_err=
    
    
    header=['CODE';'LON';'LAT';'HEIGHT';'COHERENCE';'VLOS';days]'; % [1 6+#IFG] %Cell of STRING
    clear days
    data=[ids,lon,lat,hgt,coh,vlos,d_mm]; %[#ps 6+#IFG];
    clear ids lon lat hgt coh vlos d_mm
    gis_data = cell2table(data,'VariableNames',header);
    clear data header
    writetable(gis_data,filename)
    clear gis_data





    %vlos_sd=
    %shapewrite();
    %dlmwrite('header.csv',header, 'delimiter', ',', 'precision', 9);
    %dlmwrite('data.csv',data, 'delimiter', ',', 'precision', 9);
    %xlswrite('data_excel.csv',gis_data)
    % date conversion day-693960 for excel ||| date conversion to str = datestr(DT,dateform)
    % matlabt ordinal date to python = matlab-366
    %days=cellstr(strcat(char('D'*ones(length(day),1)),datestr(day,'yyyymmdd'))); %[1 #ifg]
    %d_mm=sprintfc('%.3f',d_mm0); %[#ps #ifg] %crash
    %shape=struct('Geometry', 'Point','X',num2cell(lon),'Y',num2cell(lat),'V',num2cell(V));
    %shapewrite(shape,filename);
    %repmap : Repeat copies of array
end
