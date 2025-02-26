export STAMPS=/mnt/d/VIPHAP/insar/modules/StaMPS
export SAR=
export GETORB_BIN=
export SAR_ODR_DIR=
#export SAR_PRC_DIR
export VOR_DIR=
export INS_DIR=
export DORIS_BIN=
export TRIANGLE_BIN=/mnt/d/VIPHAP/insar/modules/triangle/bin
export SNAPHU_BIN=/mnt/d/VIPHAP/insar/modules/snaphu-1.4.2/bin

export ROI_PAC="$SAR/ROI_PAC"
#####################################
# ROI_PAC VERSION 3 
#####################################
export INT_BIN="$ROI_PAC/INT_BIN"
export INT_SCR="$ROI_PAC/INT_SCR"
#####################################

#####################################
# ROI_PAC VERSION 2.3 and before 
#####################################
#set MACH=`uname -s`
#if ($MACH == "HP-UX") then
#  export ARCHC=HP
#else if ($MACH == "IRIX") then
#  export ARCHC=SGI
#else if ($MACH == "SunOS") then
#  export ARCHC=SUN
#else if ($MACH == "Linux") then
#  export ARCHC=LIN
#else if ($MACH == "Darwin") then
#  export ARCHC=MAC
#fi
#export INT_LIB="$ROI_PAC/LIB/$ARCHC"
#export INT_BIN="$ROI_PAC/BIN/$ARCHC"
#export FFTW_LIB="$SAR/FFTW/$ARCHC""_fftw_lib"
#####################################

#####################################
# shouldn't need to change below here
#####################################

export MY_BIN="$INT_BIN"
export MATLABPATH=$STAMPS/matlab:`echo $MATLABPATH`
export DORIS_SCR="$STAMPS/DORIS_SCR"

# Needed for ROI_PAC (a bit different to standard)

### use points not commas for decimals, and give dates in US english
export LC_NUMERIC="en_US.UTF-8"
export LC_TIME="en_US.UTF-8"


export MY_SAR="$SAR"
export OUR_SCR="$MY_SAR/OUR_SCR"
export MY_SCR="$STAMPS/ROI_PAC_SCR"

export SAR_TAPE="/dev/rmt/0mn"

export PATH=${PATH}:$STAMPS/bin:$MY_SCR:$INT_BIN:$INT_SCR:$OUR_SCR:$DORIS_SCR:$GETORB_BIN:$DORIS_BIN:$TRIANGLE_BIN:$SNAPHU_BIN



