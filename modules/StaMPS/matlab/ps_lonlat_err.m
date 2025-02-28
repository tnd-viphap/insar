function ps_lonlat_err(current_result)
    load('la2.mat','la');  %incidence angle in radians
    load(strcat(current_result, '/parms.mat'),'heading'); %azimuth clockwise from the north
    re=getparm('earth_radius_below_sensor');
    load('dem_err.mat','dem_err');
    theta=(180-heading)*pi/180;
    %disp(theta);
    Dx=dem_err.*cot(la)*cos(theta);
    Dy=dem_err.*cot(la)*sin(theta);
    Dlon=acos(1-(Dx.^2)/(2*re.^2))*180/pi;
    %disp(Dlon)
    Dlat=acos(1-(Dy.^2)/(2*re.^2))*180/pi;
    lonlat_err=[Dlon Dlat];
    save('lonlat_err','lonlat_err')
    %disp(lonlat_err)
end
