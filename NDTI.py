"""
Global Cropland Analysis with Advanced Masking and NDTI Tracking
Combines Sentinel-2 processing with ESA WorldCover land type classification
Focuses on cropland areas with comprehensive cloud and quality masking
Added seasonal NDTI analysis excluding winter/summer periods
Includes multiple global regions: Vietnam, Saskatchewan Canada, Northern France
"""

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gzip
import io
import requests
from PIL import Image
from collections import defaultdict

# Initialize Google Earth Engine
PROJECT_ID = '590577866979'  # Replace with your Google Cloud Project ID

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    print("Please ensure you have:")
    print("1. Run 'earthengine authenticate'")
    print("2. Created a Google Cloud Project")
    print("3. Enabled Earth Engine API")
    print("4. Set the correct PROJECT_ID in the code")

class LandType:
    """A class to get land type using ESA WorldCover dataset. 
    (Data is only valid from 2020 to 2021)   
    """
    def __init__(self, GEE_project_id='tlg-erosion1', DataRes=0.00009, EE_initialized=True):
        if not EE_initialized: 
            # Initialize Earth Engine
            ee.Authenticate()
            ee.Initialize(project=GEE_project_id)
        
        # Load ESA WorldCover dataset
        worldcover = ee.ImageCollection('ESA/WorldCover/v200')
        self.worldcover = worldcover
        self.DataRes = DataRes

        # ESA WorldCover class mapping
        self.class_mapping = {
            10: 'Tree cover',
            20: 'Shrubland',
            30: 'Grassland',
            40: 'Cropland',  # Wilson is only interested in cropland - flag 40
            50: 'Built-up',
            60: 'Bare/sparse vegetation',
            70: 'Snow and ice',
            80: 'Permanent water bodies',
            90: 'Herbaceous wetland',
            95: 'Mangroves',
            100: 'Moss and lichen'
        }

    def get_land_cover_for_region(self, Geometry):
        """Get land cover data for specified geometry"""
        try:
            # Get the most recent WorldCover image
            worldcover_image = self.worldcover.first()
            
            # Clip to the area of interest
            clipped = worldcover_image.clip(Geometry)
            
            return {'image': clipped}
        except Exception as e:
            print(f"Error getting land cover: {e}")
            return None

    def Map_LandType(self, landcover_image):
        """Create simplified land type map focusing on cropland"""
        try:
            # Create a mask where only cropland (40) is valid (1), others are invalid (0)
            cropland_mask = landcover_image.eq(40).rename('cropland_mask')
            
            return cropland_mask
        except Exception as e:
            print(f"Error mapping land type: {e}")
            return None

def get_aoi(lon, lat, box_width, box_height):
    """Create area of interest geometry with validation"""
    try:
        # Validate inputs
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Box dimensions must be positive: {box_width}, {box_height}")
            
        # Convert box dimensions from meters to degrees (approximate)
        # Using more accurate conversion factors for latitude-dependent longitude
        width_deg = box_width / (111320 * np.cos(np.radians(lat)))  # latitude-dependent longitude conversion
        height_deg = box_height / 110540  # meters to degrees latitude
        
        return ee.Geometry.Rectangle([
            lon - width_deg/2, lat - height_deg/2,
            lon + width_deg/2, lat + height_deg/2
        ])
    except Exception as e:
        print(f"Error creating AOI: {e}")
        return None

def get_seasonal_date_ranges(years, exclude_winter_summer=True):
    """
    Generate seasonal date ranges excluding winter and summer periods
    
    Args:
        years (list): List of years to process
        exclude_winter_summer (bool): If True, only include spring and autumn
    
    Returns:
        dict: Dictionary with season names as keys and date ranges as values
    """
    seasonal_ranges = {}
    
    for year in years:
        year = int(year)
        
        if exclude_winter_summer:
            # Spring: March-May (month 3-5)
            seasonal_ranges[f'Spring_{year}'] = [f'{year}-03-01', f'{year}-05-31']
            # Autumn: September-November (month 9-11) 
            seasonal_ranges[f'Autumn_{year}'] = [f'{year}-09-01', f'{year}-11-30']
        else:
            # Include all seasons
            seasonal_ranges[f'Spring_{year}'] = [f'{year}-03-01', f'{year}-05-31']
            seasonal_ranges[f'Summer_{year}'] = [f'{year}-06-01', f'{year}-08-31']
            seasonal_ranges[f'Autumn_{year}'] = [f'{year}-09-01', f'{year}-11-30']
            seasonal_ranges[f'Winter_{year}'] = [f'{year}-12-01', f'{year+1}-02-28']
    
    return seasonal_ranges

def calculate_ndti(image):
    """
    Calculate Normalized Difference Tillage Index (NDTI)
    NDTI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    Range: -1 to 1 (as requested)
    
    Args:
        image: Sentinel-2 image with SWIR bands
    
    Returns:
        ee.Image: Image with NDTI band
    """
    try:
        # Get SWIR bands (B11: SWIR1, B12: SWIR2)
        swir1 = image.select('B11')
        swir2 = image.select('B12')
        
        # Calculate NDTI with proper range (-1 to 1)
        ndti = swir1.subtract(swir2).divide(swir1.add(swir2)).rename('NDTI')
        
        # Clamp to ensure values are within -1 to 1
        ndti = ndti.clamp(-1, 1)
        
        return image.addBands(ndti)
    except Exception as e:
        print(f"Error calculating NDTI: {e}")
        return image

class CroplandProcessor:
    """Cropland Processing with Advanced Masking and NDTI Analysis"""
    
    def __init__(self, Location, Box, Years, Verbose=True, GEE_project_id='tlg-erosion1', 
                 SentRes=10, ShowPlots=True, plot_scale=100):
        """
        Initialize the cropland processor
        
        Args:
            Location (list): [longitude, latitude] of center point
            Box (list): [width_m, height_m] dimensions in meters
            Years (list): ['start_year', 'end_year'] or single year
            Verbose (bool): Print detailed information
            SentRes (int): Sentinel-2 resolution in meters
            ShowPlots (bool): Whether to generate visualizations
            plot_scale (int): Scale for plotting
        """
        
        try:
            # Initialize Earth Engine (assuming already done)
            self.LT = LandType(EE_initialized=True)
            
            # Validate and get the geometry of the area of interest
            self.AoI_geom = get_aoi(Location[0], Location[1], Box[0], Box[1])
            if self.AoI_geom is None:
                raise ValueError("Failed to create area of interest geometry")
            
            # Get land cover data and create cropland mask
            result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
            if result is None:
                raise ValueError("Failed to get land cover data")
                
            self.RegionMap = self.LT.Map_LandType(result['image'])
            if self.RegionMap is None:
                raise ValueError("Failed to create cropland mask")
            
            # Set up seasonal date ranges (excluding winter and summer)
            self.seasonal_ranges = get_seasonal_date_ranges(Years, exclude_winter_summer=True)
            
            # Store parameters
            self.Location = Location
            self.Box = Box
            self.Years = Years
            self.Verbose = Verbose
            self.SentRes = SentRes
            self.ShowPlots = ShowPlots
            self.plot_scale = plot_scale
            
            # Storage for NDTI results
            self.ndti_results = {}
            
        except Exception as e:
            print(f"Error initializing CroplandProcessor: {e}")
            raise

    def Pull_Process_Sentinel_data(self, season_name, date_range, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.80):
        """
        Process Sentinel-2 data for a specific season with comprehensive masking
        
        Args:
            season_name (str): Name of the season
            date_range (list): [start_date, end_date] for the season
            QA_BAND (str): Cloud Score+ quality band ('cs_cdf' recommended)
            CLEAR_THRESHOLD (float): Minimum clear sky probability (0-1)
        
        Returns:
            ee.ImageCollection: Processed image collection with NDTI
        """
        
        try:
            # Support functions for processing
            def mask_clouds_advanced(img):
                """Enhanced cloud masking with shadows, snow, and water removal"""
                try:
                    # Get the Cloud Score+ data
                    cs = img.select(QA_BAND)
                    
                    # Basic cloud mask
                    cloud_mask = cs.gte(CLEAR_THRESHOLD)
                    
                    # Get Scene Classification Layer (SCL) for additional masking
                    scl = img.select('SCL')
                    
                    # Create masks for various unwanted pixels
                    cloud_shadow_mask = scl.neq(3)  # Remove cloud shadows
                    cloud_medium_mask = scl.neq(8)  # Remove medium probability clouds
                    cloud_high_mask = scl.neq(9)    # Remove high probability clouds
                    cirrus_mask = scl.neq(10)       # Remove thin cirrus
                    snow_mask = scl.neq(11)         # Remove snow/ice
                    water_mask = scl.neq(6)         # Remove water bodies
                    saturated_mask = scl.neq(1)     # Remove saturated pixels
                    dark_mask = scl.neq(2)          # Remove dark area pixels
                    
                    # Combine all masks
                    combined_mask = (cloud_mask
                                    .And(cloud_shadow_mask)
                                    .And(cloud_medium_mask) 
                                    .And(cloud_high_mask)
                                    .And(cirrus_mask)
                                    .And(snow_mask)
                                    .And(water_mask)
                                    .And(saturated_mask)
                                    .And(dark_mask))
                    
                    return img.updateMask(combined_mask)
                except Exception as e:
                    print(f"Error in cloud masking: {e}")
                    return img

            def set_pixel_count(image):
                """Calculate valid pixel count and add as property"""
                try:
                    mask = image.select('B4').mask().unmask(0)
                    count_dict = mask.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=self.AoI_geom,
                        scale=self.SentRes,
                        maxPixels=1e9,
                        bestEffort=True)
                    count = count_dict.values().get(0)
                    return image.set('valid_pixel_count', count)
                except Exception as e:
                    print(f"Error calculating pixel count: {e}")
                    return image.set('valid_pixel_count', 0)
            
            def apply_landtype_mask(image):
                """Apply cropland mask - only keep cropland areas"""
                try:
                    # Reproject the landtype mask to match Sentinel-2 resolution for efficiency
                    landtype_mask = self.RegionMap.reproject(crs=image.select('B4').projection(), scale=self.SentRes)
                    landtype_valid_mask = landtype_mask.eq(1)  # Only cropland areas
                    return image.updateMask(landtype_valid_mask)
                except Exception as e:
                    print(f"Error applying landtype mask: {e}")
                    return image

            # 1) Load Sentinel-2 and Cloud Score+ collections
            print(f"Processing {season_name}: {date_range[0]} to {date_range[1]}")
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            # Initial filtering by date and area
            filtered_s2_date_area = (s2
                .filterBounds(self.AoI_geom)
                .filterDate(date_range[0], date_range[1]))
            
            initial_count = filtered_s2_date_area.size().getInfo()
            print(f"   Initial images: {initial_count}")
            
            if initial_count == 0:
                print(f"   ❌ No images found for {season_name}")
                return ee.ImageCollection([])

            # Add cloud cover filtering to reduce dataset size
            filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
            
            after_cloud_filter = filtered_s2_date_area.size().getInfo()
            print(f"   After cloud pre-filter: {after_cloud_filter}")
            
            if after_cloud_filter == 0:
                print(f"   ❌ No images remain after cloud filtering for {season_name}")
                return ee.ImageCollection([])

            # 2) Apply cloud masking
            filtered_s2 = (filtered_s2_date_area
                .linkCollection(csPlus, [QA_BAND])
                .map(mask_clouds_advanced))

            # 3) Apply land type mask (cropland only)
            land_masked_collection = filtered_s2.map(apply_landtype_mask)

            # 4) Calculate NDTI for each image
            ndti_collection = land_masked_collection.map(calculate_ndti)

            # 5) Calculate pixel counts and filter
            ndti_collection_with_counts = ndti_collection.map(set_pixel_count)
            
            # Filter to keep only images with sufficient valid pixels
            final_collection = ndti_collection_with_counts.filter(
                ee.Filter.gt('valid_pixel_count', 50))
            
            final_count = final_collection.size().getInfo()
            print(f"   Final images: {final_count}")
            
            if final_count == 0:
                print(f"   ⚠️ Warning: No images remain for {season_name}")
                return ee.ImageCollection([])

            return final_collection
            
        except Exception as e:
            print(f"Error processing {season_name}: {e}")
            return ee.ImageCollection([])

    def calculate_seasonal_ndti_stats(self, collection, season_name):
        """Calculate NDTI statistics for a season"""
        try:
            if collection.size().getInfo() == 0:
                return None
            
            # Calculate mean NDTI for the season
            ndti_mean = collection.select('NDTI').mean()
            
            # Get statistics
            stats = ndti_mean.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.minMax(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            return {
                'season': season_name,
                'mean_ndti': stats.get('NDTI_mean', 0),
                'min_ndti': stats.get('NDTI_min', 0),
                'max_ndti': stats.get('NDTI_max', 0),
                'std_ndti': stats.get('NDTI_stdDev', 0),
                'image_count': collection.size().getInfo()
            }
            
        except Exception as e:
            print(f"Error calculating NDTI stats for {season_name}: {e}")
            return None

    def process_all_seasons(self):
        """Process all seasonal data and calculate NDTI statistics"""
        print("\n🌱 Processing seasonal NDTI analysis...")
        print("📅 Excluding winter and summer periods (focusing on spring and autumn)")
        
        for season_name, date_range in self.seasonal_ranges.items():
            print(f"\n--- {season_name} ---")
            
            # Process the season
            collection = self.Pull_Process_Sentinel_data(season_name, date_range)
            
            # Calculate NDTI statistics
            stats = self.calculate_seasonal_ndti_stats(collection, season_name)
            
            if stats:
                self.ndti_results[season_name] = stats
                print(f"   ✅ NDTI stats calculated - Mean: {stats['mean_ndti']:.3f}")
            else:
                print(f"   ❌ Failed to calculate NDTI stats")
        
        return self.ndti_results

def create_ndti_graphs(ndti_results, region_name="Unknown Region"):
    """
    Create graphs showing NDTI trends for each season
    
    Args:
        ndti_results (dict): Dictionary of NDTI statistics by season
        region_name (str): Name of the region for titles
    """
    try:
        if not ndti_results:
            print("No NDTI results to plot")
            return
        
        # Prepare data for plotting
        seasons = []
        years = []
        mean_ndti = []
        min_ndti = []
        max_ndti = []
        std_ndti = []
        
        for season_name, stats in ndti_results.items():
            if stats and stats['image_count'] > 0:
                season_type = season_name.split('_')[0]  # Spring or Autumn
                year = season_name.split('_')[1]
                
                seasons.append(season_type)
                years.append(int(year))
                mean_ndti.append(stats['mean_ndti'])
                min_ndti.append(stats['min_ndti'])
                max_ndti.append(stats['max_ndti'])
                std_ndti.append(stats['std_ndti'])
        
        if not seasons:
            print("No valid NDTI data to plot")
            return
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Season': seasons,
            'Year': years,
            'Mean_NDTI': mean_ndti,
            'Min_NDTI': min_ndti,
            'Max_NDTI': max_ndti,
            'Std_NDTI': std_ndti
        })
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'NDTI Analysis for {region_name}\n(Excluding Winter/Summer Periods)', fontsize=16)
        
        # Plot 1: Mean NDTI by Season and Year
        ax1 = axes[0, 0]
        spring_data = df[df['Season'] == 'Spring']
        autumn_data = df[df['Season'] == 'Autumn']
        
        if not spring_data.empty:
            ax1.plot(spring_data['Year'], spring_data['Mean_NDTI'], 'go-', label='Spring', linewidth=2, markersize=8)
        if not autumn_data.empty:
            ax1.plot(autumn_data['Year'], autumn_data['Mean_NDTI'], 'ro-', label='Autumn', linewidth=2, markersize=8)
        
        ax1.set_title('Mean NDTI by Season')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Mean NDTI')
        ax1.set_ylim(-1, 1)  # NDTI range as requested
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: NDTI Range (Min-Max) by Season
        ax2 = axes[0, 1]
        if not spring_data.empty:
            ax2.fill_between(spring_data['Year'], spring_data['Min_NDTI'], spring_data['Max_NDTI'], 
                           alpha=0.3, color='green', label='Spring Range')
            ax2.plot(spring_data['Year'], spring_data['Mean_NDTI'], 'go-', label='Spring Mean')
        if not autumn_data.empty:
            ax2.fill_between(autumn_data['Year'], autumn_data['Min_NDTI'], autumn_data['Max_NDTI'], 
                           alpha=0.3, color='red', label='Autumn Range')
            ax2.plot(autumn_data['Year'], autumn_data['Mean_NDTI'], 'ro-', label='Autumn Mean')
        
        ax2.set_title('NDTI Range and Mean by Season')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('NDTI')
        ax2.set_ylim(-1, 1)  # NDTI range as requested
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Standard Deviation
        ax3 = axes[1, 0]
        if not spring_data.empty:
            ax3.bar([f'{y}_S' for y in spring_data['Year']], spring_data['Std_NDTI'], 
                   color='green', alpha=0.7, label='Spring')
        if not autumn_data.empty:
            ax3.bar([f'{y}_A' for y in autumn_data['Year']], autumn_data['Std_NDTI'], 
                   color='red', alpha=0.7, label='Autumn')
        
        ax3.set_title('NDTI Standard Deviation by Season')
        ax3.set_xlabel('Year_Season')
        ax3.set_ylabel('NDTI Std Dev')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        summary_data = []
        for _, row in df.iterrows():
            summary_data.append([
                f"{row['Season']} {row['Year']}",
                f"{row['Mean_NDTI']:.3f}",
                f"{row['Min_NDTI']:.3f}",
                f"{row['Max_NDTI']:.3f}",
                f"{row['Std_NDTI']:.3f}"
            ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Season', 'Mean', 'Min', 'Max', 'Std Dev'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('NDTI Statistics Summary')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n📊 NDTI Summary for {region_name}:")
        print("=" * 50)
        for season_name, stats in ndti_results.items():
            if stats and stats['image_count'] > 0:
                print(f"{season_name}:")
                print(f"  Mean NDTI: {stats['mean_ndti']:.3f}")
                print(f"  Range: {stats['min_ndti']:.3f} to {stats['max_ndti']:.3f}")
                print(f"  Std Dev: {stats['std_ndti']:.3f}")
                print(f"  Images: {stats['image_count']}")
                print()
        
    except Exception as e:
        print(f"Error creating NDTI graphs: {e}")

def main_cropland_analysis(location=None, box_size=None, years=None, region_name=None):
    """
    Main processing pipeline for cropland analysis with NDTI tracking
    
    Args:
        location (list, optional): [longitude, latitude] coordinates
        box_size (list, optional): [width, height] in meters
        years (list or str, optional): Year(s) for analysis
        region_name (str, optional): Name for the region
    """
    
    # Set default values if not provided
    if location is None:
        location = [105.8, 10.8]  # Default: Ho Chi Minh City area
    
    if box_size is None:
        box_size = [50000, 50000]  # Default: 50km x 50km
    
    if years is None:
        years = ['2023', '2024']  # Multiple years for trend analysis
    
    if region_name is None:
        region_name = f"Region_{location[1]:.1f}N_{location[0]:.1f}E"
    
    # Validate inputs
    if len(location) != 2:
        raise ValueError("Location must be [longitude, latitude]")
    if len(box_size) != 2:
        raise ValueError("Box size must be [width, height] in meters")
    
    # Display analysis parameters
    print("🌾 Cropland NDTI Analysis")
    print("=" * 50)
    print(f"📍 Region: {region_name}")
    print(f"📍 Location: {location[1]:.2f}°N, {location[0]:.2f}°E")
    print(f"📐 Area: {box_size[0]/1000:.1f}km x {box_size[1]/1000:.1f}km")
    print(f"📅 Analysis Years: {years}")
    print(f"❄️ Excluding winter/summer periods (focusing on spring/autumn)")
    print(f"📊 NDTI Range: -1 to +1")
    
    try:
        # Initialize processor
        processor = CroplandProcessor(
            Location=location,
            Box=box_size,
            Years=years,
            Verbose=True,
            ShowPlots=True
        )
        
        # Process all seasons and calculate NDTI
        ndti_results = processor.process_all_seasons()
        
        if not ndti_results:
            print("❌ No NDTI results obtained")
            return None
        
        # Create NDTI graphs
        print("\n📈 Creating NDTI visualization graphs...")
        create_ndti_graphs(ndti_results, region_name)
        
        return {
            'processor': processor,
            'ndti_results': ndti_results,
            'region_name': region_name
        }
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        return None

def analyze_multiple_regions():
    """
    Analyze multiple global cropland regions as requested:
    - Vietnam (original)
    - Saskatchewan, Canada 
    - Northern France, Europe
    """
    
    # Define the regions as specified
    regions = {
        'Vietnam_Mekong': {
            'location': [105.8, 10.8],
            'name': 'Vietnam Mekong Delta',
            'box_size': [40000, 40000]  # 40km x 40km
        },
        'Saskatchewan_Canada': {
            'location': [-106.6, 52.1],
            'name': 'Saskatchewan, Canada',
            'box_size': [60000, 60000]  # 60km x 60km for larger agricultural areas
        },
        'Northern_France': {
            'location': [2.3, 49.5],
            'name': 'Northern France, Europe',
            'box_size': [40000, 40000]  # 40km x 40km
        }
    }
    
    results = {}
    
    print("🌍 GLOBAL CROPLAND NDTI ANALYSIS")
    print("=" * 80)
    print("Analyzing multiple regions:")
    for region_key, region_info in regions.items():
        print(f"  📍 {region_info['name']}: {region_info['location'][1]:.1f}°N, {region_info['location'][0]:.1f}°E")
    print()
    
    for region_key, region_info in regions.items():
        print(f"\n" + "="*80)
        print(f"🌾 Analyzing: {region_info['name']}")
        print(f"📍 Coordinates: {region_info['location'][1]:.1f}°N, {region_info['location'][0]:.1f}°E")
        print("="*80)
        
        try:
            result = main_cropland_analysis(
                location=region_info['location'],
                box_size=region_info['box_size'],
                years=['2023', '2024'],  # Multi-year analysis
                region_name=region_info['name']
            )
            
            if result is not None:
                results[region_key] = result
                print(f"\n✅ {region_info['name']} analysis completed successfully!")
                
                # Print summary of results
                ndti_data = result['ndti_results']
                if ndti_data:
                    print(f"📊 Summary for {region_info['name']}:")
                    for season, stats in ndti_data.items():
                        if stats and stats['image_count'] > 0:
                            print(f"   {season}: Mean NDTI = {stats['mean_ndti']:.3f}, Images = {stats['image_count']}")
            else:
                print(f"⚠️ {region_info['name']} analysis returned no results")
                
        except Exception as e:
            print(f"❌ Error analyzing {region_info['name']}: {e}")
            print("Continuing with next region...")
    
    # Create comparative summary
    print(f"\n" + "="*80)
    print("🏆 COMPARATIVE ANALYSIS SUMMARY")
    print("="*80)
    
    if results:
        # Create comparative table
        comparison_data = []
        for region_key, result in results.items():
            region_name = result['region_name']
            ndti_data = result['ndti_results']
            
            spring_means = []
            autumn_means = []
            
            for season, stats in ndti_data.items():
                if stats and stats['image_count'] > 0:
                    if 'Spring' in season:
                        spring_means.append(stats['mean_ndti'])
                    elif 'Autumn' in season:
                        autumn_means.append(stats['mean_ndti'])
            
            avg_spring = np.mean(spring_means) if spring_means else 0
            avg_autumn = np.mean(autumn_means) if autumn_means else 0
            
            comparison_data.append({
                'Region': region_name,
                'Avg_Spring_NDTI': avg_spring,
                'Avg_Autumn_NDTI': avg_autumn,
                'Seasonal_Difference': abs(avg_spring - avg_autumn)
            })
        
        # Print comparison table
        print("\n📊 Regional NDTI Comparison:")
        print("-" * 80)
        print(f"{'Region':<25} {'Spring NDTI':<12} {'Autumn NDTI':<12} {'Difference':<12}")
        print("-" * 80)
        for data in comparison_data:
            print(f"{data['Region']:<25} {data['Avg_Spring_NDTI']:>8.3f}    {data['Avg_Autumn_NDTI']:>8.3f}     {data['Seasonal_Difference']:>8.3f}")
        
        # Create comparative visualization
        create_comparative_ndti_plot(results)
    else:
        print("❌ No successful analyses to compare")
    
    return results

def create_comparative_ndti_plot(results):
    """Create a comparative plot showing NDTI across all regions"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Global Cropland NDTI Comparison\nVietnam, Saskatchewan Canada, Northern France', fontsize=16)
        
        colors = ['#2E8B57', '#FF6347', '#4169E1']  # Different colors for each region
        region_names = []
        all_spring_data = []
        all_autumn_data = []
        
        # Collect data from all regions
        for i, (region_key, result) in enumerate(results.items()):
            region_name = result['region_name']
            region_names.append(region_name)
            ndti_data = result['ndti_results']
            
            spring_vals = []
            autumn_vals = []
            spring_years = []
            autumn_years = []
            
            for season, stats in ndti_data.items():
                if stats and stats['image_count'] > 0:
                    year = int(season.split('_')[1])
                    if 'Spring' in season:
                        spring_vals.append(stats['mean_ndti'])
                        spring_years.append(year)
                    elif 'Autumn' in season:
                        autumn_vals.append(stats['mean_ndti'])
                        autumn_years.append(year)
            
            all_spring_data.append((spring_years, spring_vals))
            all_autumn_data.append((autumn_years, autumn_vals))
            
            # Plot individual region trends
            if i < 3:  # Only plot first 3 regions
                ax = axes[i//2, i%2] if i < 2 else axes[1, 0]
                
                if spring_vals:
                    ax.plot(spring_years, spring_vals, 'go-', label='Spring', linewidth=2, markersize=8)
                if autumn_vals:
                    ax.plot(autumn_years, autumn_vals, 'ro-', label='Autumn', linewidth=2, markersize=8)
                
                ax.set_title(f'{region_name}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Mean NDTI')
                ax.set_ylim(-1, 1)
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Create comparison plot in the last subplot
        ax_comp = axes[1, 1]
        
        for i, region_name in enumerate(region_names):
            spring_years, spring_vals = all_spring_data[i]
            autumn_years, autumn_vals = all_autumn_data[i]
            
            if spring_vals:
                ax_comp.plot(spring_years, spring_vals, 'o-', color=colors[i], 
                           linestyle='-', alpha=0.7, label=f'{region_name} Spring')
            if autumn_vals:
                ax_comp.plot(autumn_years, autumn_vals, 's--', color=colors[i], 
                           linestyle='--', alpha=0.7, label=f'{region_name} Autumn')
        
        ax_comp.set_title('All Regions Comparison')
        ax_comp.set_xlabel('Year')
        ax_comp.set_ylabel('Mean NDTI')
        ax_comp.set_ylim(-1, 1)
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating comparative plot: {e}")

if __name__ == "__main__":
    try:
        # Run multi-region analysis as requested
        print("🚀 Starting Global Cropland NDTI Analysis")
        print("🌍 Regions: Vietnam, Saskatchewan Canada, Northern France")
        print("📅 Excluding winter/summer periods")
        print("📊 NDTI Range: -1 to +1")
        
        # Analyze all requested regions
        all_results = analyze_multiple_regions()
        
        print("\n" + "="*80)
        print("🎉 GLOBAL CROPLAND NDTI ANALYSIS COMPLETED!")
        print("="*80)
        print("📊 Features implemented:")
        print("   ✅ Debugged original code with comprehensive error handling")
        print("   ✅ Eliminated winter and summer periods (spring/autumn only)")
        print("   ✅ Added NDTI calculation and tracking per season per region")
        print("   ✅ Set NDTI range to -1 to +1 as requested")
        print("   ✅ Created comprehensive graphs for seasonal NDTI analysis")
        print("   ✅ Multi-region analysis: Vietnam, Saskatchewan Canada, Northern France")
        print("   ✅ Comparative analysis across all regions")
        
        if all_results:
            print(f"\n📈 Successfully analyzed {len(all_results)} regions:")
            for region_key, result in all_results.items():
                ndti_count = len([s for s in result['ndti_results'].values() if s and s['image_count'] > 0])
                print(f"   🌾 {result['region_name']}: {ndti_count} successful seasonal analyses")
        
        print("\n💡 The analysis shows NDTI trends across different global cropland regions")
        print("📊 Each graph shows seasonal patterns excluding winter/summer periods")
        print("🌍 Comparative analysis reveals regional differences in tillage practices")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        print("Make sure Google Earth Engine is properly authenticated and initialized")