# coding: utf-8

### Filename: utils.py
### Language: Python v2.7
### Created by: Jesse Gourevitch

import os
from osgeo import ogr
from osgeo import osr
import sys
from osgeo import gdal
import scipy
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import hashlib

from datetime import datetime

import constants, paths

from generate_thiessen import Thiessen
    
    
def check_file_hashes(paths_dict,f):
    
    ### Extract current HUC12 ID from paths dict
    
    ws_id = paths_dict['ws_id']
    huc12_id = paths_dict['huc12_id']
    threshold_flow = paths_dict['threshold']
    
    ### Initialize paths to input data files
    dem_dir = paths_dict['dem_dir']
    stream_uri = paths_dict['stream_uri']
    reach_type = paths_dict['reach_type']
    
    ### Hash file directory
    hash_dir = paths_dict['hash_dir']
    huc12_DEM_hashfile_uri = os.path.join(hash_dir,f'{ws_id}_{huc12_id}_{threshold_flow}.txt')
    huc12_stream_hashfile_uri = os.path.join(hash_dir,f'{ws_id}_{huc12_id}_{reach_type}.txt')
    DEM_hashfile_uri = os.path.join(hash_dir,'huc8_dem_hashes.txt')
           
    
    ### HUC8 level files that depend on the HUC12 DEM files
    ### Delete if any of the HUC12 DEM files change
    huc8_dem_dir = paths_dict['huc8_DEM_dir']
    huc8_temp_dem_derivates_dir = paths_dict['temp_flow_acc_dir']
    huc8_streamline_dir = paths_dict['huc8_streamline_dir']
    
    
    ### HUC12 level files 
    ### Delete if the current DEM changed or new DEM added
    HAND_dir = paths_dict['hand_dir']
    thiessen_dir = paths_dict['thiessen_dir']
    stream_points_dir = paths_dict['streampoints_dir']
    dem_derivates_dir = paths_dict['dem_derivates_dir']
    
    
    ### Get the hashes of the HUC12 DEMs
    hash_list,dem_list = get_dem_hash_list(dem_dir)
    dem_hash_dict = dict(list(zip(dem_list,hash_list)))
    
    all_dems_match,dem_added = check_dem_hashes(hash_list,DEM_hashfile_uri)
    
    if all_dems_match:
        f.write('All DEM hashes match existing hashes & no DEMs were added or removed\n')
        f.write('   Using existing HUC8 DEM derivatives\n\n')
    else:
        f.write('All DEM hash(es) changed\n')
        f.write('   Removing existing HUC8 DEM derivatives & writing new HUC8 hash file\n')
        #DELETE HUC8-LEVEL DEM DERIVATIVES
        clear_folder(huc8_temp_dem_derivates_dir,f)
        clear_folder(huc8_streamline_dir,f)
        clear_folder(huc8_dem_dir,f)
        f.write('\n')
        write_hash_file(hash_list,dem_list,DEM_hashfile_uri)
    
    
    current_dem_hash = dem_hash_dict[f'{ws_id}_{huc12_id}.tif']
    current_streamline_hash = hashlib.md5(open(os.path.join(stream_uri),'rb').read()).hexdigest()
    hash_list = [current_dem_hash,current_streamline_hash]
    dem_filename = f'{ws_id}_{huc12_id}.tif'
    streamline_filename = stream_uri.split('\\')[-1]
    file_list = [dem_filename,streamline_filename]
    
    
    dem_hash_matches,streamline_hash_matches = check_huc12_hashes(current_dem_hash,
                                                                  dem_filename,
                                                                  current_streamline_hash,
                                                                  streamline_filename,
                                                                  huc12_DEM_hashfile_uri,
                                                                  huc12_stream_hashfile_uri)
    
    if (not dem_hash_matches) | (not streamline_hash_matches):
        f.write('Either the current DEM hash or the stream polyline hash changed\n')
        f.write('   Clearing HUC12 streampoints folder & write new huc12 hash file\n')
        #CLEAR THE STREAMPOINTS DIRECTORY
        clear_folder(stream_points_dir,f)
        f.write('\n')
        
        if not streamline_hash_matches:
            f.write('Stream polyline modified\n')
            f.write('   Clearing Thiessen polygon directory\n')
            #CLEAR THE THIESSEN DIRECTORY
            clear_folder(thiessen_dir,f)
            f.write('\n')
            write_hash_file([current_streamline_hash],[streamline_filename],huc12_stream_hashfile_uri)
            
        if (not dem_hash_matches) | dem_added:
            f.write('Either the HUC12 DEM hash does not match or a DEM was added or removed\n')
            f.write('   Clearing HUC12 DEM derivatives folder\n')
            #DELETE HUC12 DEM Derivatives
            clear_folder(dem_derivates_dir,f)
            pattern = f'{ws_id}_{huc12_id}'
            clear_folder(HAND_dir,f,pattern)
            f.write('\n')
            write_hash_file([current_dem_hash],[dem_filename],huc12_DEM_hashfile_uri)
    
    else:
        f.write('Current DEM hash and the streamline hash are unchanged\n\n')



def check_huc12_hashes(current_dem_hash,dem_filename,current_streamline_hash,streamline_filename,huc12_DEM_hashfile_uri,huc12_stream_hashfile_uri):
    if os.path.isfile(huc12_DEM_hashfile_uri):
        with open(huc12_DEM_hashfile_uri,'r') as f:
            stored_DEM_hashes = f.readlines()
            stored_DEM_hashes = [(line.strip().split(':')[0],line.strip().split(':')[1]) for line in stored_DEM_hashes]
    
        stored_DEM_hashes = dict(stored_DEM_hashes)
        
        if dem_filename in stored_DEM_hashes.keys():
            dem_hash_matches = stored_DEM_hashes[dem_filename] == current_dem_hash
        else:
             dem_hash_matches = False
    else:
        dem_hash_matches = False
        
    if os.path.isfile(huc12_stream_hashfile_uri):
        with open(huc12_stream_hashfile_uri,'r') as f:
            stored_streamline_hashes = f.readlines()
            stored_streamline_hashes = [(line.strip().split(':')[0],line.strip().split(':')[1]) for line in stored_streamline_hashes]
    
        stored_streamline_hashes = dict(stored_streamline_hashes)
        
        if streamline_filename in stored_streamline_hashes.keys():
            streamline_hash_matches = stored_streamline_hashes[streamline_filename] == current_streamline_hash
        else:
             streamline_hash_matches = False
    else:
        streamline_hash_matches = False
             
    #     if streamline_filename in stored_streamline_hashes.keys():    
    #         streamline_hash_matches = stored_streamline_hashes[streamline_filename] == current_streamline_hash
    #     else:
    #         streamline_hash_matches = False
    # else:
    #     dem_hash_matches = False
    #     streamline_hash_matches = False
        
    return dem_hash_matches,streamline_hash_matches
        
        
def check_dem_hashes(hash_list,hashfile_uri):
       
    if os.path.isfile(hashfile_uri):
        with open(hashfile_uri,'r') as f:
            stored_hashes = f.readlines()
            stored_hashes = [line.strip().split(':')[1] for line in stored_hashes]
            
        stored_in_current = np.all([stored_hash in hash_list for stored_hash in stored_hashes])
        current_in_stored = np.all([current_hash in stored_hashes for current_hash in hash_list])
        lens_match = len(hash_list) == len(stored_hashes)
        dem_added = len(hash_list) > len(stored_hashes)
        
        exact_match = current_in_stored & stored_in_current & lens_match
    else:
        exact_match = False
        dem_added = True
    
    return exact_match,dem_added

def write_hash_file(hash_list,file_list,hashfile_uri):
    
    with open(hashfile_uri,'w') as f:
        for item in list(zip(file_list,hash_list)):
            f.write('{}:{}\n'.format(item[0],item[1]))


def clear_folder(path,f,pattern=None):
    if pattern == None:
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path,file))]
    else:
        files = [file for file in os.listdir(path) if pattern in file]
        
    for file in files:
        os.remove(os.path.join(path,file))
        if not f == None:
            f.write('   Removing: {}\n'.format(os.path.join(path,file)))

def delete_derivative_files(file_path,f):
    path_parts = file_path.split('\\')
    file_root = path_parts[-1].split('.')[0]
    path = r'{}\{}'.format(path_parts[0],path_parts[1])
    for part in path_parts[2:-1]:
        path = os.path.join(path,part)
    delete_list = [file for file in os.listdir(path) if file_root in file]
    for file in delete_list:
        os.remove(os.path.join(path,file))
        if not f == None:
            f.write('   Removing: {}\n'.format(os.path.join(path,file)))
    

    
def get_dem_hash_list(dem_dir):
    dem_list = [file for file in os.listdir(dem_dir) if file[-3:] == 'tif']
    hash_list = [hashlib.md5(open(os.path.join(dem_dir,dem),'rb').read()).hexdigest() for dem in dem_list]
    return hash_list,dem_list



def clip_raster(input_uri, template_uri, output_uri, cellsize):

    if template_uri.split('.')[-1] == 'tif':

        raster = gdal.Open(template_uri, 0)
        gt = raster.GetGeoTransform()
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + gt[1] * raster.RasterXSize
        ymin = ymax + gt[5] * raster.RasterYSize
        bbox = '%s %s %s %s' %(str(xmin), str(ymin), str(xmax), str(ymax))

        proj = get_raster_projection(template_uri)
        srs = osr.SpatialReference(wkt=proj)

        args = (input_uri, output_uri, bbox, srs, srs, cellsize, cellsize)
        command = 'gdalwarp "%s" "%s" -te %s -te_srs %s -t_srs %s -tr %d %d' %args

    if template_uri.split('.')[-1] == 'shp':
        args = (input_uri, output_uri, template_uri, cellsize, cellsize)
        command = 'gdalwarp "%s" "%s" -cutline "%s" -crop_to_cutline -tr %d %d' %args
    
    # print(command)
    os.system(command)

    return None


def resample_raster(input_uri, output_uri, cellsize, method):
    args = (str(cellsize), str(cellsize), method, input_uri, output_uri)
    command = 'gdalwarp -tr %s %s  -r %s "%s" "%s"' %args
    os.system(command)

    return None


def get_raster_cellsize(input_uri):
    raster = gdal.Open(input_uri)
    gt = raster.GetGeoTransform()
    cellsize = gt[1]

    return cellsize


def array2raster(array, raster_template_uri, raster_uri,raster_output_settings = None):
    """Convert array to raster.

    Parameters:
        array (numpy array): a numpy array of data contained within the first 
            band of the raster_uri
        gt (list): a dataset geotransform list
        proj (string): WKT describing the GeoTIFF projection
        raster_uri (string): path to GeoTIFF

    Returns:
        None
    """
    
    if raster_template_uri == None:
        #extract settings values from dict
        gt = raster_output_settings['gt']
        ndval = raster_output_settings['ndval']
        proj = raster_output_settings['proj']
        redo_ndval = raster_output_settings['redo_ndval']
    else:
        redo_ndval = True
        ### Get template raster properties
        _, gt, ndval = raster2array(raster_template_uri)
    
        stem = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        raster_proj_uri = os.path.join(stem, 'Data/LCB/HAND/LCB_HAND_7point4m.tif')
        proj = get_raster_projection(raster_template_uri)

    ### Create raster
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(raster_uri, array.shape[1], array.shape[0], 1, 
        gdal.GDT_Float32, options=['BIGTIFF=IF_SAFER', 'TFW=YES', 'COMPRESS=LZW'])
    raster.SetGeoTransform((gt[0], gt[1], 0, gt[3], 0, gt[5]))
    band = raster.GetRasterBand(1)
    
    ### Write data to raster 
    if redo_ndval:
        array = np.where(np.isnan(array), ndval, array)
    band.WriteArray(array)
    band.SetNoDataValue(ndval)
    band.ComputeStatistics(True)
    raster.SetProjection(proj)
    
    ### Close and clean up dataset
    band.FlushCache()
    gdal.Dataset.__swig_destroy__(raster)
    band = raster = None

    return None


def raster2array(raster_uri):
    """Convert raster to array.

    Parameters:
        raster_uri (string): path to GeoTIFF

    Returns:
        array (numpy array): a numpy array of data contained within the first 
            band of the raster_uri
        gt (list): a dataset geotransform list
    """
    
    ds = gdal.Open(raster_uri)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    ndval = band.GetNoDataValue()
    array = band.ReadAsArray()
    array = np.where(array == ndval, np.nan, array)
    
    return array, gt, ndval


def get_raster_projection(raster_uri):
    """Get raster's projection.

    Parameters:
        raster_uri (sting): raster path URI

    Returns:
        projection (string): raster projection
    """
    
    ds = gdal.Open(raster_uri)
    projection = ds.GetProjectionRef()

    return projection 


def shp2raster(shp_uri, raster_template_uri, output_uri, burn_field=None):
   
    ### Get shapefile data
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp_uri)
    lyr = ds.GetLayer()

    ### Get raster properties
    array, gt, ndval = raster2array(raster_template_uri)
    cols, rows = array.shape
    proj = get_raster_projection(raster_template_uri)

    ### Create raster
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(output_uri, rows, cols, 1, gdal.GDT_Int32, 
        options=['BIGTIFF=IF_SAFER'])
    raster.SetGeoTransform(gt)
    band = raster.GetRasterBand(1)
    ndval = np.nan
    band.SetNoDataValue(ndval)
    # temp = gdal.FillNodata(targetBand=band,maskBand=None,maxSearchDist=5,smoothingIterations=0)
    raster.SetProjection(proj)
    if burn_field != None:
        gdal.RasterizeLayer(
        	raster, [1], lyr, options=['ATTRIBUTE=%s' % burn_field])
    else:
        gdal.RasterizeLayer(raster, [1], lyr, burn_values=[1])
    band.ComputeStatistics(True)

    # Close and clean up dataset
    band.FlushCache()
    gdal.Dataset.__swig_destroy__(raster)
    band = raster = None

    return None



def delete_shp(shp_uri):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_uri):
         driver.DeleteDataSource(shp_uri)

    return None


def align_arrays(array, gt, aoi_array, aoi_gt, ndval=np.nan):
    """Align array to area-of-interest (AOI) by padding the perimeter with no 
        data values (ndval) on sides where array is too small or clipping the 
        perimeter on sides where the array is too large.

    Parameters:
        array (numpy array): array to be aligned
        gt (list): array GDAL geotransform list
        aoi_array (numpy array): AOI array to use for alignment
        aoi_gt (list): AOI array GDAL geotransform list

    Optional parameters:
        ndval (int): no data value; default set to 9999

    Returns:
        array (numpy array):
    """

    ### Get rows and cols
    rows = array.shape[0]
    cols = array.shape[1]
    aoi_rows = aoi_array.shape[0]
    aoi_cols = aoi_array.shape[1]
    breakhere=1

    ### Adjust top
    top = int((aoi_gt[3] - gt[3]) / gt[1])
    if top < 0:
        array = array[-top:, :]
    else:
        array = np.lib.pad(array, ((top, 0), (0, 0)), 'constant', 
                           constant_values=ndval)

    ### Adjust bottom
    bottom = int(aoi_rows - (rows+top))
    if bottom < 0:
        array = array[0:aoi_rows, :]
    else:
        array = np.lib.pad(array, ((0, bottom), (0, 0)), 'constant', 
                           constant_values=ndval)

    ### Adjust left
    left = int((gt[0] - aoi_gt[0]) / gt[1])
    if left < 0:
        array = array[:, -left:]
    else:
        array = np.lib.pad(array, ((0, 0), (left, 0)), 'constant', 
                           constant_values=ndval)

    ### Adjust right
    right = int(aoi_cols - (cols+left))
    if right < 0:
        array = array[:, 0:aoi_cols]
    else:
        array = np.lib.pad(array, ((0, 0), (0, right)), 'constant', 
                           constant_values=ndval)

    return array


def line2points(line_uri, points_uri, id_field): 
    ### Initialize ESRI Shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    ### Read line layer
    line_ds = driver.Open(line_uri, 0)
    line_lyr = line_ds.GetLayer()

    ### Create points datasource and layer objects
    points_ds = driver.CreateDataSource(points_uri)
    #points_lyr = points_ds.CreateLayer(points_uri, geom_type=ogr.OFTInteger)
    points_lyr = points_ds.CreateLayer(points_uri, line_lyr.GetSpatialRef(), geom_type=ogr.wkbPoint)

    ### Create ID field within points layer
    field = ogr.FieldDefn(id_field, ogr.OFTInteger)
    points_lyr.CreateField(field)

    ### Iterate through features in line layer
    for feat in line_lyr:
        ### Get line feature data
        feat_code = int(feat.GetField('Code'))
        line_str = feat.GetGeometryRef()

        if line_str.GetGeometryName() == 'LINESTRING':
            for pt in line_str.GetPoints():
                ### Get point coordinates
                pt_obj = ogr.Geometry(ogr.wkbPoint)
                pt_obj.AddPoint(pt[0], pt[1])

                ### Add point feature to points layer
                feat_def = points_lyr.GetLayerDefn()
                point_feat = ogr.Feature(feat_def)
                point_feat.SetGeometry(pt_obj)
                point_feat.SetField(id_field, feat_code)
                points_lyr.CreateFeature(point_feat)
                point_feat = None
        elif line_str.GetGeometryName() == 'MULTILINESTRING':
            for sub_line in line_str:
                for pt in sub_line.GetPoints():
                    ### Get point coordinates
                    pt_obj = ogr.Geometry(ogr.wkbPoint)
                    pt_obj.AddPoint(pt[0], pt[1])

                    ### Add point feature to points layer
                    feat_def = points_lyr.GetLayerDefn()
                    point_feat = ogr.Feature(feat_def)
                    point_feat.SetGeometry(pt_obj)
                    point_feat.SetField(id_field, feat_code)
                    points_lyr.CreateFeature(point_feat)
                    point_feat = None

    return None


def thiessen_polygons(points_uri, polygon_uri, output_dir, hand_uri):
    # Static variables
    unique_field = 'Code'

    # Get output raster resolution
    hand_data = gdal.Open(hand_uri)
    transform = {'transform': hand_data.GetGeoTransform(),
                 'x_size': hand_data.RasterXSize,
                 'y_size': hand_data.RasterYSize}

    processor = Thiessen()
    processor.generate_voronoi(points_uri, polygon_uri, output_dir, unique_field, transform)

    return None


def get_huc12list(ws, ws_id):
    
    ### Get list of HUC12s from "Buffered_DEM" directory
    huc12_set = set()
    stem = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for fn in os.listdir(os.path.join(stem, 'Data', ws, 'Buffered_DEM')):
        huc12_set.add(fn.split('.')[0])

    ### Only add back HUC12s with the ws_id in the fn
    huc12_list = [fn for fn in huc12_set if ws_id in fn]

    ### Only add back HUC12s that are not in the exclude_list
    exclude_list = constants.exclude_list
    huc12_list = [fn for fn in huc12_list if fn not in exclude_list]
    
    ### Remove HUC12 with trailing zeros (e.g. WIN_05010)
    huc12_list = [fn for fn in huc12_list if int(fn[-1]) > 0]

    ### Sort HUC12 list
    huc12_list = sorted(huc12_list)

    return huc12_list


def generate_rvs(n):

    np.random.seed(0)

    ### Create parameter PDFs
    mannings_pdf = stats.norm(constants.mannings_mean, constants.mannings_std)

    low_grade_xsarea_pdf = stats.norm(constants.low_grade_xsarea_mean, constants.low_grade_xsarea_std)
    high_grade_xsarea_pdf = stats.norm(constants.high_grade_xsarea_mean, constants.high_grade_xsarea_std)

    xa = constants.low_grade_slope_mean - constants.low_grade_slope_std
    xb = constants.low_grade_slope_mean + constants.low_grade_slope_std
    a = (xa - constants.low_grade_slope_mean) / constants.low_grade_slope_std
    b = (xb - constants.low_grade_slope_mean) / constants.low_grade_slope_std
    low_grade_slope_pdf = stats.truncnorm(a, b, loc=constants.low_grade_slope_mean, scale=constants.low_grade_slope_std)

    xa = constants.high_grade_slope_mean - constants.high_grade_slope_std
    xb = constants.high_grade_slope_mean + constants.high_grade_slope_std
    a = (xa - constants.high_grade_slope_mean) / constants.high_grade_slope_std
    b = (xb - constants.high_grade_slope_mean) / constants.high_grade_slope_std
    high_grade_slope_pdf = stats.truncnorm(a, b, loc=constants.high_grade_slope_mean,
                                           scale=constants.high_grade_slope_std)

    discharge_pdf_dict = {}
    for Q in constants.discharge_std_dict.keys():
        lower_se = constants.discharge_std_dict[Q][0]
        upper_se = constants.discharge_std_dict[Q][1]
        Q_mean = (lower_se + upper_se) / 2.0
        Q_std = (upper_se - lower_se) / 2.0
        a = (lower_se - Q_mean) / Q_std
        b = (upper_se - Q_mean) / Q_std
        discharge_pdf_dict[Q] = stats.truncnorm(a, b, loc=Q_mean, scale=Q_std)

    ### Create random sample for parameter values
    mannings_rvs = mannings_pdf.rvs(n-1)
    low_grade_xsarea_rvs = low_grade_xsarea_pdf.rvs(n-1)
    high_grade_xsarea_rvs = high_grade_xsarea_pdf.rvs(n - 1)
    low_grade_slope_rvs = low_grade_slope_pdf.rvs(n-1)
    high_grade_slope_rvs = high_grade_slope_pdf.rvs(n - 1)
    
    discharge_rvs_dict = {}
    for Q in discharge_pdf_dict.keys():
        discharge_rvs_dict[Q] = discharge_pdf_dict[Q].rvs(n-1)

    ### Insert values at beginning of rvs parameter arrays
    mannings_rvs = np.insert(mannings_rvs, 0, 0)
    low_grade_xsarea_rvs = np.insert(low_grade_xsarea_rvs, 0, 0)
    high_grade_xsarea_rvs = np.insert(high_grade_xsarea_rvs, 0, 0)
    low_grade_slope_rvs = np.insert(low_grade_slope_rvs, 0, 0)
    high_grade_slope_rvs = np.insert(high_grade_slope_rvs, 0, 0)

    for Q in discharge_rvs_dict.keys():
        discharge_rvs_dict[Q] = np.insert(discharge_rvs_dict[Q], 0, 0)

    ### Put RVS into dictionary
    rvs_dict = {
        'mannings_rvs': mannings_rvs,
        'low_grade_slope_rvs': low_grade_slope_rvs,
        'high_grade_slope_rvs': high_grade_slope_rvs,
        'low_grade_xsarea_rvs': low_grade_xsarea_rvs,
        'high_grade_xsarea_rvs': high_grade_xsarea_rvs,
        'discharge_rvs_dict': discharge_rvs_dict
        }

    return rvs_dict


def print_elapsedtime(t0):
    t1 = datetime.now()
    delta_t = t1 - t0
    mins, secs = divmod(delta_t.days * 86400 + delta_t.seconds, 60)
    print ('\nTime elapsed: %s minutes, %s seconds' %(mins, secs))

    return None


def rvs2dict(i, rvs_dict):
    ### Initialize parameters RVS
    mannings_rvs = rvs_dict['mannings_rvs']
    low_grade_xsarea_rvs = rvs_dict['low_grade_xsarea_rvs']
    high_grade_xsarea_rvs = rvs_dict['high_grade_xsarea_rvs']
    discharge_rvs_dict = rvs_dict['discharge_rvs_dict']
    low_grade_slope_rvs = rvs_dict['low_grade_slope_rvs']
    high_grade_slope_rvs = rvs_dict['high_grade_slope_rvs']

    params_dict = {
        'mannings_rvs': mannings_rvs[i] + 1,
        'low_grade_xsarea_rvs': low_grade_xsarea_rvs[i] + 1,
        'high_grade_xsarea_rvs': high_grade_xsarea_rvs[i] + 1,
        'low_grade_slope_rvs': low_grade_slope_rvs[i] + 1,
        'high_grade_slope_rvs': high_grade_slope_rvs[i] + 1,
        'discharge_rvs_Q2': discharge_rvs_dict['Q2'][i] + 1,
        'discharge_rvs_Q5': discharge_rvs_dict['Q5'][i] + 1,
        'discharge_rvs_Q10': discharge_rvs_dict['Q10'][i] + 1,
        'discharge_rvs_Q25': discharge_rvs_dict['Q25'][i] + 1,
        'discharge_rvs_Q50': discharge_rvs_dict['Q50'][i] + 1,
        'discharge_rvs_Q100': discharge_rvs_dict['Q100'][i] + 1,
        'discharge_rvs_Q200': discharge_rvs_dict['Q200'][i] + 1,
        'discharge_rvs_Q500': discharge_rvs_dict['Q500'][i] + 1,
        }

    return params_dict

def streamstats_qaqc():
    stem = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(stem, 'Data')
    ws_list = ['Lake_Champlain', 'Lamoille_River', 'Mettawee_River', 
               'Missisquoi_River', 'Otter_Creek', 'Winooski_River']

    for ws in ws_list:
        ss_uri = os.path.join(data_dir, ws, 'Stream_Stats/Stream_Stats.csv')
        ss_df = pd.read_csv(ss_uri)

        nhd_dir = os.path.join(data_dir, ws, 'NHD')
        nhd_dirlist = os.listdir(nhd_dir)
        nhd_dirlist = [fn.split('.')[0] for fn in nhd_dirlist]
        nhd_dirlist = list(set(nhd_dirlist))

        for nhd_fn in nhd_dirlist:
            nhd_uri = os.path.join(nhd_dir, nhd_fn+'.shp')
            nhd_df = gpd.read_file(nhd_uri)

            for reach in nhd_df['Code']:
                if not (ss_df['Name']==reach).any(): 
                    print (nhd_fn[:-4], int(reach))

    return None


