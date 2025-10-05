from pci.saringestaoi import saringestaoi
from pci import algo
from pci.exceptions import PCIException
import os
import re
# -------------------------
# Input / Output Parameters
# -------------------------
# sar_dictionary = [
#     "F:\Waterloo Related Files\NASA Hackathon\S1A_IW_GRDH_1SDV_20250819_suez\S1A_IW_GRDH_1SDV_20250819T154854_20250819T154919_060607_078A08_8896.SAFE\manifest.safe",
#     "F:\Waterloo Related Files\NASA Hackathon\S1A_IW_GRDH_1SDV_20250305_Jiangnan\S1A_IW_GRDH_1SDV_20250305T095528_20250305T095553_058168_072F9F_0FD9.SAFE\manifest.safe"
# ]
sar_input = r"F:\Waterloo Related Files\NASA Hackathon\S1A_IW_GRDH_1SDV_20251003_antwerp\S1A_IW_GRDH_1SDV_20251003T055837_20251003T055902_061257_07A400_1BF0.SAFE\manifest.safe"
output_ingest = r"./s4_ingest.pix"
output_ortho = r"./s4_ortho.pix"
#output_png = r"./s1_ortho.png"
dem_file = r"C:\PCI Geomatics\CATALYST Professional\etc\gmted2010.jp2"
def bump_filename(filename):
    match = re.search(r'(s\d+)_', filename)
    if match:
        num = int(match.group(1)[1:]) + 1
        bumped = re.sub(r's\d+_', f's{num}_', filename, count=1)
        return bumped
    return filename

if os.path.exists(output_ingest) or os.path.exists(output_ortho):
    output_ingest = bump_filename(output_ingest)
    output_ortho = bump_filename(output_ortho)
# ------------------------------------------------
# 1. Ingest SAR (Area of Interest optional)
# ------------------------------------------------
# If you don't yet have an AOI shapefile, just skip the 'aoi' parameter
# from pci.saringestaoi import saringestaoi
# fili = sar_input
# filo = output_ingest
# calibtyp = 'sigma'
# dbiw = []
# maskfile = ''
# mask = []
# fillop = ''
# filedem = ''
# dbec = []
# poption = 'aver'
# dblayout = ''
# saringestaoi(fili, filo, calibtyp, dbiw, maskfile, mask, fillop, filedem, dbec, poption, dblayout)

try:
    algo.saringestaoi(fili = sar_input, filo = output_ingest, calibtyp = 'sigma', dbiw = [], maskfile = '', mask = [], fillop = '', filedem = '', dbec = [], poption = 'aver', dblayout = '')
except PCIException as e:
    print('\n*** PCIException: {}'.format(e))
except Exception as e:
    print('\n*** Exception: {}'.format(e))

try:
    algo.ortho(mfile= output_ingest, filo = output_ortho, ftype = "PIX", foptions = "TILED256", filedem = dem_file)
except PCIException as e:
    print('\n*** PCIException: {}'.format(e))
except Exception as e:
    print('\n*** Exception: {}'.format(e))
# from pci.felee import felee
      

from pci.fspec import fspec


file	=	output_ortho
dbic	=	[1]	# input radar image
dboc	=	[1]	# output filtered radar image
flsz	=	[7,7]	# specifies a 7 x 7 filter
ftyp	=	'GAMMA'	# use GAMMA filter
mask	=	[]	# default, process entire image
nlook	=	[10]	# number of looks
damp	=	[]	# default damping factor 1.0
imagefmt	=	'AMP'	# Amplitude image format
cthresh	=	[]	# not used with GAMMA filter
ethresh	=	[]	# not used with GAMMA filter
gthresh	=	[]	# not used with GAMMA filter

fspec( file, dbic, dboc, flsz, ftyp, mask, nlook, damp, imagefmt, cthresh, ethresh, gthresh )
       
try:
    algo.felee(file=output_ortho, dbic = [1], dboc = [2], flsz = [], mask = [], nlook = [1.0], damp = [1.0], imagefmt = 'AMP')
except PCIException as e:
    print('\n*** PCIException: {}'.format(e))
except Exception as e:
    print('\n*** Exception: {}'.format(e))


