### Flag to only generate the HAND layer
only_generate_HAND = False

### Number of parallel cores (used in calls to mpiexec)
num_cores = 3

### Adjust threshold
'''
The Dinfinity area calculation in TauDEM 5.3.7 reports "specific catchment
area", which is the area divided by the contour length (contour length
estimated as the cell size) so for a square cell, the value reported is the
number of cells multiplied by the length of one cell face (cell size).  This 
flag converts the threshold to "specific catchment area" by dividing by the 
cell size
'''
adjust_threshold = True

### Watershed ID dictionary
ws_id_dict = {
	'Lake_Champlain': 'LKC',
	'Lamoille_River': 'LAM',
	'Mettawee_River': 'MET',
	'Missisquoi_River': 'MSQ',
	'Otter_Creek': 'OTR',
	'Winooski_River': 'WIN'
	}

### HUC8 code dictionary
huc8_code_dict = {
	'LKC': '04150408',
	'LAM': '04150405',
	'MET': '04150401',
	'MSQ': '04150407',
	'OTR': '04150402',
	'WIN': '04150403'
	}

### HUC12s to exclude
exclude_list = ['LKC_0101', 'LKC_0102', 'LKC_0103', 'LKC_0104',
				'LKC_0205', 'LKC_0206', 'LKC_0302', 'LKC_0303', 'LKC_0304', 
				'LKC_0601', 'LKC_0602', 'LKC_0802', 'LKC_1001', 'LKC_1002',
				'LKC_1004', 'LKC_1005', 'LKC_1006', 'LKC_1007', 
				'LKC_1203', 'LKC_1505', 'LKC_1506', 'LKC_1507', 'LKC_1602', 
				'LKC_1603', 'LKC_1604', 'MSQ_0104', 'MSQ_0104', 'MSQ_0105', 
				'MSQ_0201', 'MSQ_0202', 'MSQ_0203', 'MET_0205', 'MET_0307']

### Threshold flow accumulation; unit = pixels (or m^2); equals 9 sq miles
# threshold_flow = 23309893.0 #23,309,893 m^2
sq_meters_per_sq_mile = 2589988.0 #2,589,988.0 m^2 1 sq mile
sq_miles = 2
threshold_flow = float(round(sq_miles*sq_meters_per_sq_mile))

### Pixel size to use when tiling the HUC8 raster
pixel_size = 5

### Stage list to generate rating curve
stage_list = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 50]

### Flood recurrance interval list
interval_list = [2, 5, 10, 25, 50, 100, 200, 500]

### Dictionary converting LULC codes into Manning's n coefficients
### Based on the calibrated Truehart model
### See email from Rebecca on July 31, 2019
lulc2mannings_dict = {
	'1':  0.12,  # Deciduous
	'2':  0.13,  # Coniferous
	'3':  0.06,  # Herbaceous
	'4':  0.10,  # Shrub
	'5':  0.05,  # Water
	'6':  0.12,  # Emergent wetland
	'7':  0.10,  # Scrub/shrub wetland
	'8':  0.13,  # Forested wetland
	'9':  0.065, # Crops
	'10': 0.06,  # Pasture
	'11': 0.06,  # Hay
	'12': 0.04,  # Barren
	'13': 0.05,  # Buildings
	'14': 0.13,  # Other
	'15': 0.13
	}

### Manning's n mean and standard deviation
### Rebecca gave Jesse these values on July 30, 2019
mannings_mean = 0
mannings_std = 0.25

### Slope mean and standard deviation
### Emailed from Rebecca on November 30, 2021
### Subject line: FW: HAND Update
low_grade_slope_mean = 0.1
low_grade_slope_std = 0.66
high_grade_slope_mean = -0.25
high_grade_slope_std = 0.45

### Cross-sectional area mean and standard deviation
### Rebecca computed these values from LiDAR analysis - July 30, 2019
### Updated November 30, 2021.  Subject line: FW: HAND Update
low_grade_xsarea_mean = 0.25
low_grade_xsarea_std = 0.1
high_grade_xsarea_mean = 0.16
high_grade_xsarea_std = 0.1

### Values provided by Rebecca
discharge_std_dict = {
	'Q2': [-0.326, 0.484],
	'Q5': [-0.324, 0.480],
	'Q10': [-0.332, 0.497],
	'Q25': [-0.342, 0.520],
	'Q50': [-0.354, 0.549],
	'Q100': [-0.369, 0.584],
	'Q200': [-0.389, 0.637], # Linear interpolation between Q100 and Q500
	'Q500': [-0.408, 0.690],
	}

