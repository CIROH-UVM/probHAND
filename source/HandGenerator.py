"""HandGenerator Class Module."""

import os
from osgeo import ogr
from osgeo import gdal
import platform
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import hashlib

import subprocess

import constants, utils

import gdal_merge


class HandGenerator:
    """ A class to handle generation of HAND maps from DEMs.

    Given an input digital elevation model, perform spatial analyses and output a height above nearest drainage map.
    """

    def __init__(self, paths_dict, taudem_fcn_fmt='lower'):

        self.paths = paths_dict
        self.set_taudem_fcn_fmts(taudem_fcn_fmt)
    
        self.generate_hand()


    def set_taudem_fcn_fmts(self, fmt):
        self.taudem_fcns = dict()
        if fmt.lower() == 'pascal':
            self.taudem_fcn_fmts = 'pascal'
            self.taudem_fcns['areadinf'] = 'AreaDinf'
            self.taudem_fcns['dinfflowdir'] = 'DinfFlowDir'
            self.taudem_fcns['dinfdistdown'] = 'DinfDistDown'
            self.taudem_fcns['pitremove'] = 'PitRemove'
            self.taudem_fcns['threshold'] = 'Threshold'
        elif fmt.lower() == 'lower':
            self.taudem_fcn_fmts = 'lower'
            
            self.taudem_fcns['areadinf'] = 'areadinf'
            self.taudem_fcns['dinfflowdir'] = 'dinfflowdir'
            self.taudem_fcns['dinfdistdown'] = 'dinfdistdown'
            self.taudem_fcns['pitremove'] = 'pitremove'
            self.taudem_fcns['threshold'] = 'threshold'
        else:
            raise ValueError(f"expected 'pascal' or 'lower' for taudem_fcn_fmt; got {fmt}")

    def generate_hand(self):
        """Create the HAND surface. This is a raster that represents the height 
        above the nearest drainage (i.e., the bottom of the channel). The 
        model uses three tools found in TauDEM.
        """

        with open(self.paths['HAND_log_uri'],'w') as f:
            
            #f.write('Checking file hashes & deleting outdated derivative files')
            #utils.check_file_hashes(self.paths,f)
        
            ### Initialize paths to input data files
            dem_buffered_uri = self.paths['dem_buff_uri'] #HUC12 DEM
            ws_uri = self.paths['ws_uri']
            huc8_stream_uri_1m = self.paths['stream_huc8_uri']
            huc8_hand_uri = os.path.join(self.paths['huc8_derivatives_dir'], 'HAND.tif')

            ### Initialize path to DEM derivatives directory
            dem_derivates_dir = self.paths['dem_derivates_dir']
                
            ### output URIs        
            hand_uri = self.paths['hand_uri']
            flowdir_uri = os.path.join(dem_derivates_dir, 'flowdir.tif')
            flowdir_buff_uri = os.path.join(dem_derivates_dir, 'flowdir_buff.tif')
            dem_filled_uri = os.path.join(dem_derivates_dir, 'dem_filled.tif')
            dem_filled_buff_uri = os.path.join(dem_derivates_dir, 'dem_filled_buff.tif')
            stream_uri = os.path.join(dem_derivates_dir, 'stream_network.tif')
            slope_uri = os.path.join(dem_derivates_dir, 'slope.tif')

            ### If HUC8 HAND does not exist, generate it, so that it can be imputed later ###
            if not os.path.exists(huc8_hand_uri):
                print('HUC-8 level HAND for imputing needed.  Generating HUC-8 derivatives')
                self.generate_huc8_stream_network(f)


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
                    self.generate_huc8_stream_network(f)
                
                print ('Clipping stream raster to watershed extent')
                f.write('   Clipping the stream Raster\n')
                f.write(f'   utils.clip_raster({huc8_stream_uri_1m}, {ws_uri}, {stream_uri}, 1.0)\n\n')
                utils.clip_raster(huc8_stream_uri_1m, ws_uri, stream_uri, 1.0)
                
                
            f.write('Pitfilling HUC12 DEM\n')
            if os.path.exists(dem_filled_buff_uri):
                f.write('   Pit-filled HUC8 dem exists, using existing file\n\n')
            else:
                f.write ('   Removing pits from buffered DEM\n\n')
                fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['pitremove']} -z {dem_buffered_uri} -fel {dem_filled_buff_uri}'
                f.write('   ' + fcn_str + '\n\n')
                subprocess.run(fcn_str.split())
                
                
                    
            f.write('      Calculate slope & d-infinity flow direction\n')
            if os.path.exists(flowdir_buff_uri):
                f.write('         HUC12 flow direction file exists, using existing file\n\n')
            else:
                f.write( '         Calculating D-inf flow direction and slope from buffered DEM\n')
                fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['dinfflowdir']} -fel {dem_filled_buff_uri} -ang {flowdir_buff_uri} -slp {slope_uri}'
                f.write('   ' + fcn_str + '\n\n')
                subprocess.run(fcn_str.split())

            utils.clip_raster(flowdir_buff_uri, ws_uri, flowdir_uri, 1.0)
            utils.clip_raster(dem_filled_buff_uri, ws_uri, dem_filled_uri, 1.0)
                
                
        
            ### D-Infinity Distance Down
            
            f.write('Generating HAND Layer - Start\n')
            print ('Creating HAND layer')
            fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['dinfdistdown']} -ang {flowdir_uri} -fel {dem_filled_uri} -src {stream_uri} -dd {hand_uri} -m v -nc'
            f.write('   ' + fcn_str + '\n\n')
            subprocess.run(fcn_str.split())
                
            f.write('Done trying to generate HAND')
            if not os.path.exists(hand_uri):
                print ('\n***** ERROR: HAND LAYER WAS NOT CREATED *****\n')
                f.write('\n***** ERROR: HAND LAYER WAS NOT CREATED *****\n')
                sys.exit()

        return None

    def mosaic_rasters(self, dem_dir, dem_list, output_fn, pixel_size, nd_value, merge_type="average"):
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
        if nd_value is None:
            argv_list = ['-o',output_fn,'-ps',pixel_size,pixel_size]
        else:
            nd_value = str(nd_value)
            argv_list = ['-a_nodata', nd_value, '-o',output_fn,'-ps',pixel_size,pixel_size]
        
        #Append the DEM filenames to the arg_v command list
        for file in dem_list:
            #If dem_dir is None, then the filenames in the dem_list are absolute
            #paths including the .tiff file name.  Otherwise, they are the .tiff 
            #only and the path needs to be included.
            if dem_dir is None:
                argv_list.append(file)
            else:
                argv_list.append(os.path.join(dem_dir,file))
        
        #Add the commands to sys.argv and call the gdal_merge function
        sys.argv[1:] = argv_list
        gdal_merge.main()
    
    def generate_huc8_stream_network(self, f):
        
        #output file
        huc8_stream_uri_1m = self.paths['stream_huc8_uri'] #after resampling from constants.cell_size to 1m
        
        #Initialize the paths and file uris
        huc8_dem_derivatives_dir = self.paths['temp_flow_acc_dir']
        huc8_stream_fn = huc8_stream_uri_1m.split('\\')[-1].split('.')[0]
        huc8_stream_uri_5m = os.path.join(huc8_dem_derivatives_dir, '{}_5m.tif'.format(huc8_stream_fn))
        flowdir_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'flowdir_buff.tif') #output of DinfFlowDir
        slope_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'slope_buff.tif') #output of DinfFlowDir
        huc8_dem_buff_filled_uri = os.path.join(huc8_dem_derivatives_dir, 'huc8_dem_buff_filled.tif') #output of PitRemove
        flowacc_buff_uri = os.path.join(huc8_dem_derivatives_dir, 'flowacc_buff.tif') #output of AreaDinf
        huc8_dem_buffered_uri = self.paths['huc8_dem_uri']
        dem_dir = self.paths['dem_dir']
        hand_uri = os.path.join(huc8_dem_derivatives_dir, 'HAND.tif')
            
        
        f.write('      Creating HUC8 DEM\n')
        if os.path.exists(huc8_dem_buffered_uri):
            f.write('         HUC8 buffered DEM exists, using existing\n\n')
        else:
            f.write('         HUC8 buffered DEM exists, mosaicing HUC12 DEMs.  Input Files:\n\n')
            dem_list = [file for file in os.listdir(dem_dir) if file[-3:] == 'tif']
            for dem in dem_list:
                f.write('    {}\n'.format(dem))
            nd_value = -9999
            self.mosaic_rasters(dem_dir,dem_list,huc8_dem_buffered_uri,constants.pixel_size,nd_value)
            
        if os.path.exists(huc8_stream_uri_5m):
            f.write('      HUC8 stream network exists, using existing\n\n')
        else:
            f.write('      HUC8 stream network missing, generating network\n\n')

            f.write('      Pitfilling HUC8 DEM\n')
            if os.path.exists(huc8_dem_buff_filled_uri):
                f.write('         Pit-filled HUC8 dem exists, using existing file\n\n')
            else:
                f.write ('         Removing pits from buffered DEM\n\n')
                fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['pitremove']} -z {huc8_dem_buffered_uri} -fel {huc8_dem_buff_filled_uri}'
                f.write('   ' + fcn_str + '\n\n')
                subprocess.run(fcn_str.split())
                
            f.write('      Calculate slope & d-infinity flow direction - Start\n')
            if os.path.exists(flowdir_buff_uri):
                f.write('         HUC8 flow direction file exists, using existing file\n\n')
            else:
                f.write( '         Calculating D-inf flow direction and slope from buffered DEM\n')
                fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['dinfflowdir']} -fel {huc8_dem_buff_filled_uri} -ang {flowdir_buff_uri} -slp {slope_buff_uri}'
                f.write('   ' + fcn_str + '\n\n')
                subprocess.run(fcn_str.split())
        
            f.write('      Calculate Flow Accumulation - Start\n')
            ### Calculate flow accumulation
        
            if os.path.exists(flowacc_buff_uri):
                f.write('         Flow accumulation raster exists, using existing\n\n')
            else:
                f.write ('         Calculating flow accumulation from D-infinity flow direction\n')
                fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['areadinf']} -ang {flowdir_buff_uri} -sca {flowacc_buff_uri}'
                f.write('   ' + fcn_str + '\n\n')
                subprocess.run(fcn_str.split())
        
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
                
            f.write('      Delineating Stream Network\n')
            ### Delineate stream network based on flow accumulation threshold
            if os.path.exists(huc8_stream_uri_5m):
                f.write(f'         Stream network raster exists for threshold: {constants.threshold_flow}, using existing raster\n\n')
            else:
                f.write ('         Delineating stream network from flow accumulation raster\n\n')
                fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['threshold']} -ssa {flowacc_buff_uri} -src {huc8_stream_uri_5m} -thresh {str(thresh)}'
                f.write('   ' + fcn_str + '\n\n')
                subprocess.run(fcn_str.split())
                
        #upsample stream raster from 5m to 1m
        pixel_size = 1.0 # constants.pixel_size
        method='max'
        utils.resample_raster(huc8_stream_uri_5m, huc8_stream_uri_1m, pixel_size, method)

        ### Generaet HUC-8 level HAND ###
        fcn_str = f'mpiexec -n {constants.num_cores} {self.taudem_fcns['dinfdistdown']} -ang {flowdir_buff_uri} -fel {huc8_dem_buff_filled_uri} -src {huc8_stream_uri_5m} -dd {hand_uri} -m v -nc'
        f.write('   ' + fcn_str + '\n\n')
        subprocess.run(fcn_str.split())

        return None


