#     Copyright (C) 2015  Bekaert David - University of Leeds
#     Email: eedpsb@leeds.ac.uk or davidbekaert.com
# 
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

# Give the correct path to the APS toolbox
export APS_toolbox="/media/viphap/88b9ea56-f0c1-45dd-b6dc-bf46504a647b/insar/modules/TRAIN/"
# export PYTHONPATH="/home/viphap/insar/modules/TRAIN/python_modules/"
# full path to the get_modis.py file
# export get_modis_filepath="/home/viphap/insar/modules/TRAIN/python_packages/oscar-client-python/get_modis.py"

#####################################
# shouldn't need to change below here
#####################################

case ":${MATLABPATH}:" in
::)     export MATLABPATH="${APS_toolbox}/matlab";;
*:${APS_toolbox}/matlab:*)  : ;;
*)      export MATLABPATH="${APS_toolbox}/matlab:${MATLABPATH}";;
esac

export APS_toolbox_scripts="${APS_toolbox}/scripts"
export APS_toolbox_bin="${APS_toolbox}/bin"
export PATH="${PATH}:${APS_toolbox_bin}:${PYTHONPATH}"
