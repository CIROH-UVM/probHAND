# coding: utf-8

### Filename: driver.py
### Language: Python v2.7
### Created by: Jesse Gourevitch

import os
from osgeo import ogr
from osgeo import gdal
import platform
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import hashlib

import constants, utils

import gdal_merge



def mosaic_rasters(dem_dir,dem_list,output_fn,pixel_size,nd_value,merge_type="average"):
    """
    NOTE: If the input files are overlapping, the values for any overlapping 
    pixels will be overwritten as new images are added to the tiled result

    Parameters
    ----------
    dem_dir : TYPE
        DESCRIPTION.
    dem_list : TYPE
        DESCRIPTION.
    output_fn : TYPE
        DESCRIPTION.
    pixel_size : TYPE
        DESCRIPTION.
    nd_value : TYPE
        DESCRIPTION.
    merge_type : TYPE, optional
        DESCRIPTION. The default is "average".

    Returns
    -------
    None.

    """
    
    #Check if the output target file exits & remove it if it does
    if os.path.exists(output_fn):
        os.remove(output_fn)
    
    #Convert the raster cell size to a string (required to add to the argv_list)
    pixel_size = str(pixel_size)
    
    #Build the list of arguments to use when calling the gdal_merge function
    if nd_value == None:
        argv_list = ['-o',output_fn,'-ps',pixel_size,pixel_size]
    else:
        nd_value = str(nd_value)
        argv_list = ['-a_nodata', nd_value, '-o',output_fn,'-ps',pixel_size,pixel_size]
    
    #Append the DEM filenames to the arg_v command list
    for file in dem_list:
        #If dem_dir == None, then the filenames in the dem_list are absolute
        #paths including the .tiff file name.  Otherwise, they are the .tiff 
        #only and the path needs to be included.
        if dem_dir == None:
            argv_list.append(file)
        else:
            argv_list.append(os.path.join(dem_dir,file))
       
    #Add the commands to sys.argv and call the gdal_merge function
    sys.argv[1:] = argv_list
    gdal_merge.main()
    
def generate_huc8_stream_network(paths_dict,f):
    
    #output file
    huc8_stream_uri_1m = paths_dict['stream_huc8_uri'] #after resampling from constants.cell_size to 1m
    
    #Initialize the paths and file uris
    huc8_dem_derivatives_dir = paths_dict['temp_flow_acc_dir']
    huc8_stream_fn = huc8_stream_uri_1m.split('\\')[-1].split('.')[0]
    huc8_stream_uri_5m = os.path.join(huc8_dem_derivatives_dir, '{}_5m.tif'.format(huc8_stream_fn))
    flowdir_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'flowdir_buff.tif') #output of DinfFlowDir
    slope_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'slope_buff.tif') #output of DinfFlowDir
    huc8_dem_buff_filled_uri = os.path.join(huc8_dem_derivatives_dir, 'huc8_dem_buff_filled.tif') #output of PitRemove
    flowacc_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'flowacc_buff.tif') #output of AreaDinf
    huc8_dem_buffered_uri = paths_dict['huc8_dem_uri']
    dem_dir = paths_dict['dem_dir']
    hand_uri = os.path.join(huc8_dem_derivatives_dir, 'HAND.tif')
        
    
    f.write('      Creating HUC8 DEM\n')
    if os.path.exists(huc8_dem_buffered_uri):
        f.write('         HUC8 buffered DEM exists, using existing\n\n')
    else:
        f.write('         HUC8 buffered DEM exists, mosaicing HUC12 DEMs.  Input Files:\n\n')
        dem_list = [file for file in os.listdir(dem_dir) if file[-3:] == 'tif']
        for dem in dem_list:
            f.write('    {}\n'.format(dem))
        nd_value = 0
        mosaic_rasters(dem_dir,dem_list,huc8_dem_buffered_uri,constants.pixel_size,nd_value)
        
    if os.path.exists(huc8_stream_uri_5m):
        f.write('      HUC8 stream network exists, using existing\n\n')
    else:
        f.write('      HUC8 stream network missing, generating network\n\n')

        f.write('      Pitfilling HUC8 DEM\n')
        if os.path.exists(huc8_dem_buff_filled_uri):
            f.write('         Pit-filled HUC8 dem exists, using existing file\n\n')
        else:
            f.write ('         Removing pits from buffered DEM\n\n')
            args = (constants.num_cores, huc8_dem_buffered_uri, huc8_dem_buff_filled_uri)
            # f.write('   Removing Pits\n')
            f.write('         mpiexec -n %s PitRemove -z "%s" -fel "%s"\n\n' %args)
            os.system('mpiexec -n %s PitRemove -z "%s" -fel "%s"' %args)
            
        f.write('      Calculate slope & d-infinity flow direction - Start\n')
        if os.path.exists(flowdir_buff_uri):
            f.write('         HUC8 flow direction file exists, using existing file\n\n')
        else:
            f.write( '         Calculating D-inf flow direction and slope from buffered DEM\n')
            args = (constants.num_cores, huc8_dem_buff_filled_uri, flowdir_buff_uri, slope_buff_uri)
            
            f.write('         mpiexec -n %s DinfFlowDir -fel "%s" -ang "%s" -slp "%s"\n\n' %args)
            os.system('mpiexec -n %s DinfFlowDir -fel "%s" -ang "%s" -slp "%s"' %args)
    
        f.write('      Calculate Flow Accumulation - Start\n')
        ### Calculate flow accumulation
    
        if os.path.exists(flowacc_buff_uri):
            f.write('         Flow accumulation raster exists, using existing\n\n')
        else:
            f.write ('         Calculating flow accumulation from D-infinity flow direction\n')
            args = (constants.num_cores, flowdir_buff_uri, flowacc_buff_uri)
            
            f.write('         Call to AreaDinf\n')
            
            f.write('         mpiexec -n %s AreaDinf -ang "%s" -sca "%s" -nc\n\n' %args)
            os.system('mpiexec -n %s AreaDinf -ang "%s" -sca "%s" -nc' %args)
    
        ### Maximum flow accumulation is greater than threshold, adjust it
        f.write('         Calls to GDAL\n')
        f.write(f'            GDAL call: flowacc_buff_raster = gdal.Open({flowacc_buff_uri})\n')
        flowacc_buff_raster = gdal.Open(flowacc_buff_uri)
        f.write('            GDAL call: flowacc_buff_band = flowacc_buff_raster.GetRasterBand(1)\n')
        flowacc_buff_band = flowacc_buff_raster.GetRasterBand(1)
        f.write('            GDAL Call: flowacc_buff_stats = flowacc_buff_band.GetStatistics(True, True)\n')
        flowacc_buff_stats = flowacc_buff_band.GetStatistics(True, True)
        max_flowacc = flowacc_buff_stats[1]
    
        thresh = constants.threshold_flow
        if constants.adjust_threshold:
            thresh = float(round(constants.pixel_size*thresh/(constants.pixel_size**2)))
        while max_flowacc < thresh:
            thresh *= 0.1
            
        # if constants.generate_headwater_mask:
        #     f.write('Generating headwater mask raster\n')
        #     generate_headwater_mask(paths_dict,ws_uri)
    
        f.write('      Delineating Stream Network\n')
        ### Delineate stream network based on flow accumulation threshold
        if os.path.exists(huc8_stream_uri_5m):
            f.write(f'         Stream network raster exists for threshold: {constants.threshold_flow}, using existing raster\n\n')
        else:
            f.write ('         Delineating stream network from flow accumulation raster\n\n')
            args = (constants.num_cores, flowacc_buff_uri, huc8_stream_uri_5m, str(thresh))
            
            f.write('         mpiexec -n %s Threshold -ssa "%s" -src "%s" -thresh %s\n\n' %args)
            os.system('mpiexec -n %s Threshold -ssa "%s" -src "%s" -thresh %s' %args)
            
    #upsample stream raster from 5m to 1m
    pixel_size = constants.pixel_size
    nd_value = -999
    method='max'
    utils.resample_raster(huc8_stream_uri_5m, huc8_stream_uri_1m, pixel_size, method)

    ### Generaet HUC-8 level HAND ###
    args = (constants.num_cores, flowdir_buff_uri, huc8_dem_buff_filled_uri, huc8_stream_uri_5m, hand_uri)
    os.system('mpiexec -n %s DinfDistDown -ang "%s" -fel "%s" -src "%s" -dd "%s" -m v -nc' % args)
    

# def generate_headwater_mask(paths_dict,f):
    
    
#     #output file
#     headwater_mask_dir = paths_dict['headwater_mask_dir']
#     huc8_stream_uri_1m = paths_dict['stream_huc8_uri'] #after resampling from constants.cell_size to 1m
    
    
#     huc8_dem_derivatives_dir = paths_dict['temp_flow_acc_dir']
#     flowdir_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'flowdir_buff.tif') #output of DinfFlowDir
#     slope_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'slope_buff.tif') #output of DinfFlowDir
#     huc8_dem_buff_filled_uri = os.path.join(huc8_dem_derivatives_dir, 'huc8_dem_buff_filled.tif') #output of PitRemove (FEL)
#     flowacc_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'flowacc_buff.tif') #output of AreaDinf
#     huc8_dem_buffered_uri = paths_dict['huc8_dem_uri']
#     dem_dir = paths_dict['dem_dir']
    
    
#     output_point_uri = paths_dict['watershed_output_uri']
    
#     """
#     mpiexec -n 8 PitRemove logan.tif
#         input: DEM
#         output: fel
#     mpiexec -n 8 D8Flowdir -p loganp.tif -sd8 logansd8.tif -fel loganfel.tif
#         input: fel.tif
#         outputs: p, sd8
#     mpiexec -n 8 DinfFlowdir -ang loganang.tif -slp loganslp.tif -fel loganfel.tif 
#         input: fel.tif
#         output: ang.tif, slp.tif
#     mpiexec -n 8 AreaD8 -p loganp.tif -ad8 loganad8.tif
#         input: p.tif
#         output: ad8.tif
#     mpiexec -n 8 AreaDinf -ang loganang.tif -sca logansca.tif ==> 
#         input: ang.tif
#         output: sca.tif
#     mpiexec -n 8 Aread8 -p loganp.tif -o loganoutlet.shp -ad8 loganad8o.tif 
#         inputs: p.tif, outlet.tif
#         output: ad80.tif
#     mpiexec -n 8 Gridnet -p loganp.tif -plen loganplen.tif -tlen logantlen.tif -gord logangord.tif 
#         input: p.tif
#         output: plen.tif, tlen.tif, gord.tif
#     mpiexec -n 8 PeukerDouglas -fel loganfel.tif -ss loganss.tif 
#         input: fel.tif
#         output: ss.tif
#     mpiexec -n 8 Aread8 -p loganp.tif -o loganoutlet.shp -ad8 loganssa.tif -wg loganss.tif
#         inputs: p.tif, shp.tif
#         outputs: ssa.tif, ss.tif
#     mpiexec -n 8 Dropanalysis -p loganp.tif -fel loganfel.tif -ad8 loganad8.tif -ssa loganssa.tif -drp logandrp.txt -o loganoutlet.shp -par 5 500 10 0 
#         inputs: p.tif, fel.tif, ad8.tif, ssa.tif, outlet.shp
#         output: drp.txt
#     mpiexec -n 8 Threshold -ssa loganssa.tif -src logansrc.tif -thresh 300
#         inputs: ssa.tif, threshold from drp.txt (the smallest threshold with a Tstatistic that is less than 2)
#         output: src.tif
#     mpiexec -n 8 Streamnet -fel loganfel.tif -p loganp.tif -ad8 loganad8.tif -src logansrc.tif -ord loganord3.tif -tree logantree.dat -coord logancoord.dat -net logannet.shp -w loganw.tif -o loganoutlet.shp
#         inputs: fel.tif, p.tif, ad8.tif, src.tif, outlet.shp
#         outputs: tree.dat, coord.dat, net.shp, w.tif, 
#     """
    
    
#     cellsize = utils.get_raster_cellsize(flowacc_buff_uri)
#     utils.clip_raster(flowacc_buff_uri, ws_uri, flowacc_uri, cellsize)
    
    
#     array, gt, ndval = utils.raster2array(flowacc_uri)
#     proj = utils.get_raster_projection(flowacc_uri)
    
    
#     thresh = constants.headwater_threshold
#     if constants.adjust_threshold:
#         thresh = float(round(cellsize*thresh/(cellsize**2)))
    
#     max_flowacc = np.nanmax(array)
#     while max_flowacc < thresh:
#         old_thresh = thresh
#         thresh *= 0.1
#         print('WARNING: threshold is greater than max flow accumulation of {}'.format(max_flowacc))
#         print('   Reducing threshold from {} to {}'.format(old_thresh,thresh))
        
#     nodata_mask = np.where(np.isnan(array),1,0)
    
#     array = np.where(array<thresh,1,0)
    
#     array = np.where(nodata_mask == 1,np.nan,array)
    
#     raster_output_settings = {
#         'gt':gt,
#         'ndval':ndval,
#         'proj':proj,
#         'redo_ndval':True}
    
#     raster_template_uri = None
#     utils.array2raster(array, raster_template_uri, headwater_mask_uri,raster_output_settings)
    

def generate_hand(paths_dict):
    """Create the HAND surface. This is a raster that represents the height 
        above the nearest drainage (i.e., the bottom of the channel). The 
        model uses three tools found in TauDEM.

    Parameters:
        paths_dict (dict): dictionary containing data paths and parameter values

    Returns:
        None
    """
    
    HAND_log_uri = paths_dict['HAND_log_uri']

    with open(HAND_log_uri,'w') as f:
        
        f.write('Checking file hashes & deleting outdated derivative files')
        utils.check_file_hashes(paths_dict,f)
    
        ### Initialize paths to input data files
        dem_buffered_uri = paths_dict['dem_buff_uri'] #HUC12 DEM
        # huc8_dem_buffered_uri = paths_dict['huc8_dem_uri']
        # ws_buff_uri = paths_dict['ws_buff_uri']
        ws_uri = paths_dict['ws_uri']
        huc8_stream_uri_1m = paths_dict['stream_huc8_uri']
        huc8_hand_uri = os.path.join(paths_dict['huc8_derivatives_dir'], 'HAND.tif')


        ### Initialize path to DEM derivatives directory
        dem_derivates_dir = paths_dict['dem_derivates_dir']
        
            
        ### output URIs        
        hand_uri = paths_dict['hand_uri']
        flowdir_uri = os.path.join(dem_derivates_dir, 'flowdir.tif')
        flowdir_buff_uri = os.path.join(dem_derivates_dir, 'flowdir_buff.tif')
        dem_filled_uri = os.path.join(dem_derivates_dir, 'dem_filled.tif')
        dem_filled_buff_uri = os.path.join(dem_derivates_dir, 'dem_filled_buff.tif')
        stream_uri = os.path.join(dem_derivates_dir, 'stream_network.tif')
        slope_uri = os.path.join(dem_derivates_dir, 'slope.tif')
        
        #testing files
        flowacc_buff_uri = os.path.join(dem_derivates_dir,'temp_flow_accumulation.tif')
        temp_stream_network = os.path.join(dem_derivates_dir,'temp_stream_network.tif')

        ### If HUC8 HAND does not exist, generate it, so that it can be imputed later ###
        if not os.path.exists(huc8_hand_uri):
            print('HUC-8 level HAND for imputing needed.  Generating HUC-8 derivatives')
            generate_huc8_stream_network(paths_dict, f)


        ### If HAND layer already exists, return None
        if os.path.exists(hand_uri):
            f.write('HAND layer already exists for the values in settings.txt\n')
            f.write('   Exiting HAND creation')
            return None
            
    
        ### Clip HUC8 stream raster to watershed extent
    
        f.write('Get HUC12 stream raster\n')
        if os.path.exists(stream_uri):
            f.write('   Stream raster for HUC12 already exists, using existing\n')
        else:
            
            if os.path.exists(huc8_stream_uri_1m):
                f.write('   HUC-8 level stream raster exists for the current flow accumulation threshold\n')
                f.write('      Clipping HUC12 stream raster from HUC8 stream raster\n\n')
            else:
                f.write('   HUC-8 level stream raster does not exist for the current flow accumulation threshold\n')
                f.write('      Recalculating HUC8 stream raster\n\n')
                generate_huc8_stream_network(paths_dict,f)
            
            print ('Clipping stream raster to watershed extent')
            f.write('   Clipping the stream Raster\n')
            f.write(f'   utils.clip_raster({huc8_stream_uri_1m}, {ws_uri}, {stream_uri}, 1.0)\n\n')
            utils.clip_raster(huc8_stream_uri_1m, ws_uri, stream_uri, 1.0)
            
            
        f.write('Pitfilling HUC12 DEM\n')
        if os.path.exists(dem_filled_buff_uri):
            f.write('   Pit-filled HUC8 dem exists, using existing file\n\n')
        else:
            f.write ('   Removing pits from buffered DEM\n\n')
            args = (constants.num_cores, dem_buffered_uri, dem_filled_buff_uri)
            # f.write('   Removing Pits\n')
            f.write('   mpiexec -n %s PitRemove -z "%s" -fel "%s"\n\n' %args)
            os.system('mpiexec -n %s PitRemove -z "%s" -fel "%s"' %args)
            
            
                 
        f.write('      Calculate slope & d-infinity flow direction\n')
        if os.path.exists(flowdir_buff_uri):
            f.write('         HUC12 flow direction file exists, using existing file\n\n')
        else:
            f.write( '         Calculating D-inf flow direction and slope from buffered DEM\n')
            args = (constants.num_cores, dem_filled_buff_uri, flowdir_buff_uri, slope_uri)
            
            f.write('         mpiexec -n %s DinfFlowDir -fel "%s" -ang "%s" -slp "%s"\n\n' %args)
            os.system('mpiexec -n %s DinfFlowDir -fel "%s" -ang "%s" -slp "%s"' %args)

        utils.clip_raster(flowdir_buff_uri, ws_uri, flowdir_uri, 1.0)
        utils.clip_raster(dem_filled_buff_uri, ws_uri, dem_filled_uri, 1.0)
            
            
    
        ### D-Infinity Distance Down
        
        f.write('Generating HAND Layer - Start\n')
        print ('Creating HAND layer')
        # f.write('   Populating args variable\n')
        args = (constants.num_cores, flowdir_uri, dem_filled_uri, stream_uri, hand_uri)
        
        
        f.write('    mpiexec -n %s DinfDistDown -ang "%s" -fel '
                  '"%s" -src "%s" -dd "%s" -m v -nc\n\n' %args)
        
        os.system('mpiexec -n %s DinfDistDown -ang "%s" -fel '
                  '"%s" -src "%s" -dd "%s" -m v -nc' %args)
            
        f.write('Done trying to generate HAND')
        if not os.path.exists(hand_uri):
            print ('\n***** ERROR: HAND LAYER WAS NOT CREATED *****\n')
            f.write('\n***** ERROR: HAND LAYER WAS NOT CREATED *****\n')
            sys.exit()
        
    # f.close()

    # sys.exit()
    return None


def resample_data(paths_dict, cellsize):
    
    ### Initialize paths to input data files
    hand_uri = paths_dict['hand_uri']
    lulc_uri = paths_dict['lulc_uri']
    ws_uri = paths_dict['ws_uri']

    ### Clip or resample HAND layer
    hand_cellsize = utils.get_raster_cellsize(hand_uri)
    hand_resampled_uri = hand_uri[:-4] + '_resampled%dm.tif' %cellsize
    if not os.path.exists(hand_resampled_uri) and hand_cellsize != cellsize:
        if cellsize == 7.4:
            hand_lcb_7point4m_uri = paths_dict['hand_lcb_7point4m_uri']
            utils.clip_raster(hand_lcb_7point4m_uri, ws_uri, hand_resampled_uri, 7.4)

        if cellsize == 5.0:
            hand_huc8_5m_uri = paths_dict['hand_huc8_5m_uri']
            utils.clip_raster(hand_huc8_5m_uri, ws_uri, hand_resampled_uri, 5.0)
        
        else:
            print ('Resampling HAND layer to lower resolution')
            method = 'bilinear'
            utils.resample_raster(hand_uri, hand_resampled_uri, cellsize, method)
        
    if hand_cellsize != cellsize:
        paths_dict['hand_uri'] = hand_resampled_uri

    ### Resample LULC layer to lower resolution 
    lulc_cellsize = utils.get_raster_cellsize(lulc_uri)
    lulc_resampled_uri = lulc_uri[:-4] + '_resampled%dm.tif' %cellsize
    if not os.path.exists(lulc_resampled_uri) and lulc_cellsize != cellsize:
        print ('Resampling LULC layer to lower resolution')
        method = 'near'
        utils.resample_raster(lulc_uri, lulc_resampled_uri, cellsize, method)
    
    if lulc_cellsize != cellsize:
        paths_dict['lulc_uri'] = lulc_resampled_uri

    return paths_dict


def impute_hand(paths_dict, cellsize):
    
    hand_1m_uri = paths_dict['hand_uri']
    hand_1m_imputed_uri = hand_1m_uri[:-4] + '_imputed1m.tif'
    
    if cellsize == 1.0 and not os.path.exists(hand_1m_imputed_uri):
        print ('Imputing missing values in 1m HAND layer using 7.4m HAND layer')

        print ('\tReading 5m HAND raster to array')
        hand_5m_uri = os.path.join(paths_dict['huc8_derivatives_dir'], 'HAND.tif')
        hand_5m_array, hand_5m_gt, _ = utils.raster2array(hand_5m_uri)

        print ('\tReading 1m HAND raster to array')
        hand_1m_array, hand_1m_gt, _ = utils.raster2array(hand_1m_uri) 

        print ('\tReading 1m LULC raster to array')
        lulc_1m_uri = paths_dict['lulc_uri']
        lulc_1m_array, lulc_1m_gt, _ = utils.raster2array(lulc_1m_uri)

        print ('\tAligning LULC array and HAND array')
        if lulc_1m_array.shape != hand_1m_array.shape:
            lulc_1m_array = utils.align_arrays(
                lulc_1m_array, lulc_1m_gt, hand_1m_array, hand_1m_gt)

        print ('\tFinding indices of missing values')
        ndval_rows, ndval_cols = np.where((np.isnan(hand_1m_array)) & 
                                            (~np.isnan(lulc_1m_array)))
        
        print ('\tGetting coordinates for indices of missing values')
        # Don't need to account for cellsize, since cellsize is 1.0
        x = hand_1m_gt[0] + ndval_cols 
        y = hand_1m_gt[3] - ndval_rows 

        print ('\tGetting values indices in 7m HAND array')
        c_5m = np.floor((x - hand_5m_gt[0]) / hand_5m_gt[1]).astype(int)
        r_5m = np.floor((hand_5m_gt[3] - y) / hand_5m_gt[1]).astype(int)
        
        print ('\tInserting imputed values into 1m array')
        imputed_val = hand_5m_array[r_5m, c_5m]
        hand_1m_array[ndval_rows, ndval_cols] = imputed_val

        print ('\tExporting imputed array to raster')
        utils.array2raster(hand_1m_array, hand_1m_uri, hand_1m_imputed_uri)
    paths_dict['hand_uri'] = hand_1m_imputed_uri

    return paths_dict


def generate_theissen(paths_dict, cellsize):
    
    ### Initialize paths to data files
    hand_uri = paths_dict['hand_uri']
    stream_uri = paths_dict['stream_uri']
    streampoints_dir = paths_dict['streampoints_dir']
    streampoints_uri = os.path.join(streampoints_dir, 'stream_points.shp')
    watershed_uri = paths_dict['ws_uri']
    thiessen_dir = paths_dict['thiessen_dir']
    thiessen_shp_uri = os.path.join(thiessen_dir, 'thiessen.shp')
    thiessen_raster_uri = os.path.join(thiessen_dir, 'thiessen_%dm.tif' %cellsize)

    ### Convert stream line vertices to points
    if not os.path.exists(streampoints_uri):
        print ('Converting stream line vertices to points')
        utils.line2points(stream_uri, streampoints_uri, 'Code')

    ### Calculate Theissen polygons
    if not os.path.exists(thiessen_shp_uri):
        print ('Calculating Thiessen polygons for stream vertice points')
        utils.thiessen_polygons(streampoints_uri, watershed_uri, thiessen_dir, hand_uri)

    return None


def extract_dem2points(paths_dict):
    
    ### Initialize paths to data files
    streampoints_dir = paths_dict['streampoints_dir']
    streampoints_uri = os.path.join(streampoints_dir, 'stream_points.shp')
    dem_uri = paths_dict['dem_buff_uri']

    ### Extract DEM values using stream vertice points
    dem_extract_uri = os.path.join(streampoints_dir, 'stream_points_DEMvalues.shp')
    if not os.path.exists(dem_extract_uri):
        print ('Extracting DEM values using stream vertice points')
        dem_array, dem_gt, _ = utils.raster2array(dem_uri)
        spoints_df = gpd.read_file(streampoints_uri)
        x = np.array([p.x for p in spoints_df['geometry']])
        y = np.array([p.y for p in spoints_df['geometry']])
        rows = ((dem_gt[3] - y) / dem_gt[1]).astype(int)
        cols = ((x - dem_gt[0]) / dem_gt[1]).astype(int)

        values = dem_array[rows, cols]
        spoints_df['RASTERVALU'] = values

        spoints_df = spoints_df[['Code', 'RASTERVALU', 'geometry']]
        spoints_df.to_file(dem_extract_uri, driver='ESRI Shapefile', 
            geometry=spoints_df['geometry'])

    return None


def read_data2mem(paths_dict, cellsize):
    
    ### Initialize paths to sub-directories
    streampoints_dir = paths_dict['streampoints_dir']

    ### Initialize paths to input data files
    ws_uri = paths_dict['ws_uri']
    hand_uri = paths_dict['hand_uri']
    lulc_uri = paths_dict['lulc_uri']
    dem_uri = paths_dict['dem_buff_uri']
    stream_uri = paths_dict['stream_uri']
    ratingcurve_uri = paths_dict['ratingcurve_uri']
    streamstats_uri = paths_dict['streamstats_uri']

    thiessen_dir = paths_dict['thiessen_dir']
    thiessen_raster_uri = os.path.join(thiessen_dir, 'thiessen_%dm.tif' %cellsize)

    ### Read HAND layer to Numpy array
    hand_array, hand_gt, _ = utils.raster2array(hand_uri)
    
    ### Read Thiessen layer to Numpy array
    thiessen_array, thiessen_gt, _ = utils.raster2array(thiessen_raster_uri)
    if thiessen_array.shape != hand_array.shape:
        thiessen_array = utils.align_arrays(
            thiessen_array, thiessen_gt, hand_array, hand_gt)

    ### Read stream points DBF to Pandas dataframe
    dem_extract_uri = os.path.join(streampoints_dir, 'stream_points_DEMvalues.shp')
    streampoints_df = gpd.read_file(dem_extract_uri)
    streampoints_df = streampoints_df.rename(columns={'RASTERVALU': 'ELEV'})

    ### Read LULC layer to Numpy array
    lulc_array, lulc_gt, _ = utils.raster2array(lulc_uri)
    if lulc_array.shape != hand_array.shape:
        lulc_array = utils.align_arrays(lulc_array, lulc_gt, hand_array, hand_gt)

    ### Read stream layer to Geopandas dataframe
    stream_df = gpd.read_file(stream_uri)
    
    #make sure that the code column is of type float
    stream_df['Code'] = stream_df['Code'].astype(float)

    ### Read stream stats CSV to Pandas dataframe
    streamstats_df = pd.read_csv(streamstats_uri)

    ### Create data_dict to store data variables
    data_dict = {}

    ### Put variables in data_dict
    data_dict['hand_array'] = hand_array
    data_dict['hand_gt'] = hand_gt
    data_dict['thiessen_array'] = thiessen_array
    data_dict['streampoints_df'] = streampoints_df
    data_dict['lulc_array'] = lulc_array
    data_dict['stream_df'] = stream_df
    data_dict['streamstats_df'] = streamstats_df

    return data_dict


