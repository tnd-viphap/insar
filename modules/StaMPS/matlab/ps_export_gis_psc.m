function ps_export_gis(filename,lon_rg,lat_rg)

    load('ps1','lonlat')  %dimension [#ps 2]
    load('ps1','xy')  %%dimension [#ps 3] id despoints PS
    load ('hgt1','hgt') %dimension [#ps 1]
    load('pm1','coh_ps') % [#ps 1]
    load('da1','D_A') % [#ps 1]





    %decouper selon une zone
    ix=lonlat(:,1)>=min(lon_rg)&lonlat(:,1)<=max(lon_rg)&lonlat(:,2)>=min(lat_rg)&lonlat(:,2)<=max(lat_rg);
    xy=xy(ix,:);
    lonlat=lonlat(ix,:);
    hgt=hgt(ix,:);
    coh_ps=coh_ps(ix,:);
    D_A=D_A(ix,:);


    %Les variables a exporter 
    ids=sprintfc('PS_%d',xy(:,1)); %[#ps 1] %Cell of STRING
    clear xy
    lon=sprintfc('%.8f',lonlat(:,1)); %[#ps 1] %Cell of STRING
    lat=sprintfc('%.8f',lonlat(:,2)); %[#ps 1] %Cell of STRING
    clear lonlat
    hgt=sprintfc('%.1f',hgt); %[#ps 1] %Cell of STRING
    coh=sprintfc('%.1f',coh_ps); %[#ps 1] %Cell of STRING
    clear coh_ps
    D_A=sprintfc('%.1f',D_A); %[#ps 1] %Cell of STRING


    
    header={'CODE','LON','LAT','HEIGHT','COHERENCE','D_A'}; % [1 6+#IFG] %Cell of STRING
    clear days
    data=[ids,lon,lat,hgt,coh,D_A]; %[#ps 6+#IFG];
    clear ids lon lat hgt coh D_A
    gis_data = cell2table(data,'VariableNames',header);
    clear data header
    writetable(gis_data,filename)
    clear gis_data

end
