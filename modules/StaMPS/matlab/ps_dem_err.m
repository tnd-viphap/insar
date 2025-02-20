function ps_dem_error()
    slant_range=getparm('range_pixel_spacing');
    near_range=getparm('near_range_slc');
    re=getparm('earth_radius_below_sensor');
    rs=getparm('sar_to_earth_center');
    lambda=getparm('lambda');
    load('ps2.mat','ij');
    load('scla2','K_ps_uw');
    range_pixel=ij(:,3);
    K=K_ps_uw;
    rg=near_range+range_pixel*slant_range;
    alpha=pi-acos((rg.^2+re^2-rs^2)/2./rg/re);
    dem_err=-K*lambda.*rg.*sin(alpha)/4/pi;
    save('dem_err','dem_err')
    disp(dem_err)
end
