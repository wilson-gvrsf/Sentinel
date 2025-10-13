"""
Cropland Analysis with NDTI Tracking for Maryland/Delaware Region
Modified to analyze specific bounding box for May 14, 2016
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
            40: 'Cropland',
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
            worldcover_image = self.worldcover.first()
            clipped = worldcover_image.clip(Geometry)
            return {'image': clipped}
        except Exception as e:
            print(f"Error getting land cover: {e}")
            return None

    def Map_LandType(self, landcover_image):
        """Create simplified land type map focusing on cropland"""
        try:
            cropland_mask = landcover_image.eq(40).rename('cropland_mask')
            return cropland_mask
        except Exception as e:
            print(f"Error mapping land type: {e}")
            return None

def get_aoi_from_bounds(north, south, west, east):
    """Create area of interest geometry from bounding box coordinates"""
    try:
        # Validate inputs
        if not (-180 <= west <= 180) or not (-180 <= east <= 180):
            raise ValueError(f"Invalid longitude: west={west}, east={east}")
        if not (-90 <= south <= 90) or not (-90 <= north <= 90):
            raise ValueError(f"Invalid latitude: north={north}, south={south}")
        if south >= north:
            raise ValueError(f"South latitude must be less than north latitude")
        if west >= east:
            raise ValueError(f"West longitude must be less than east longitude")
            
        return ee.Geometry.Rectangle([west, south, east, north])
    except Exception as e:
        print(f"Error creating AOI: {e}")
        return None

def calculate_ndti(image):
    """
    Calculate Normalized Difference Tillage Index (NDTI)
    NDTI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    Range: -1 to 1
    """
    try:
        swir1 = image.select('B11')
        swir2 = image.select('B12')
        ndti = swir1.subtract(swir2).divide(swir1.add(swir2)).rename('NDTI')
        ndti = ndti.clamp(-1, 1)
        return image.addBands(ndti)
    except Exception as e:
        print(f"Error calculating NDTI: {e}")
        return image

class CroplandProcessor:
    """Cropland Processing with Advanced Masking and NDTI Analysis"""
    
    def __init__(self, bounds, start_date, end_date, Verbose=True, 
                 SentRes=10, ShowPlots=True, plot_scale=100):
        """
        Initialize the cropland processor
        
        Args:
            bounds (dict): Dictionary with 'north', 'south', 'west', 'east' keys
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            Verbose (bool): Print detailed information
            SentRes (int): Sentinel-2 resolution in meters
            ShowPlots (bool): Whether to generate visualizations
            plot_scale (int): Scale for plotting
        """
        
        try:
            self.LT = LandType(EE_initialized=True)
            
            # Create geometry from bounds
            self.AoI_geom = get_aoi_from_bounds(
                bounds['north'], bounds['south'], 
                bounds['west'], bounds['east']
            )
            if self.AoI_geom is None:
                raise ValueError("Failed to create area of interest geometry")
            
            # Get land cover data and create cropland mask
            result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
            if result is None:
                raise ValueError("Failed to get land cover data")
                
            self.RegionMap = self.LT.Map_LandType(result['image'])
            if self.RegionMap is None:
                raise ValueError("Failed to create cropland mask")
            
            # Store parameters
            self.bounds = bounds
            self.start_date = start_date
            self.end_date = end_date
            self.Verbose = Verbose
            self.SentRes = SentRes
            self.ShowPlots = ShowPlots
            self.plot_scale = plot_scale
            
            # Storage for NDTI results
            self.ndti_results = {}
            
        except Exception as e:
            print(f"Error initializing CroplandProcessor: {e}")
            raise

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.80):
        """
        Process Sentinel-2 data with comprehensive masking
        """
        
        try:
            def mask_clouds_advanced(img):
                """Enhanced cloud masking with shadows, snow, and water removal"""
                try:
                    cs = img.select(QA_BAND)
                    cloud_mask = cs.gte(CLEAR_THRESHOLD)
                    scl = img.select('SCL')
                    
                    cloud_shadow_mask = scl.neq(3)
                    cloud_medium_mask = scl.neq(8)
                    cloud_high_mask = scl.neq(9)
                    cirrus_mask = scl.neq(10)
                    snow_mask = scl.neq(11)
                    water_mask = scl.neq(6)
                    saturated_mask = scl.neq(1)
                    dark_mask = scl.neq(2)
                    
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
                    landtype_mask = self.RegionMap.reproject(crs=image.select('B4').projection(), scale=self.SentRes)
                    landtype_valid_mask = landtype_mask.eq(1)
                    return image.updateMask(landtype_valid_mask)
                except Exception as e:
                    print(f"Error applying landtype mask: {e}")
                    return image

            print(f"Processing: {self.start_date} to {self.end_date}")
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            filtered_s2_date_area = (s2
                .filterBounds(self.AoI_geom)
                .filterDate(self.start_date, self.end_date))
            
            initial_count = filtered_s2_date_area.size().getInfo()
            print(f"   Initial images: {initial_count}")
            
            if initial_count == 0:
                print(f"   ❌ No images found")
                return ee.ImageCollection([])

            filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
            
            after_cloud_filter = filtered_s2_date_area.size().getInfo()
            print(f"   After cloud pre-filter: {after_cloud_filter}")
            
            if after_cloud_filter == 0:
                print(f"   ❌ No images remain after cloud filtering")
                return ee.ImageCollection([])

            filtered_s2 = (filtered_s2_date_area
                .linkCollection(csPlus, [QA_BAND])
                .map(mask_clouds_advanced))

            land_masked_collection = filtered_s2.map(apply_landtype_mask)
            ndti_collection = land_masked_collection.map(calculate_ndti)
            ndti_collection_with_counts = ndti_collection.map(set_pixel_count)
            
            final_collection = ndti_collection_with_counts.filter(
                ee.Filter.gt('valid_pixel_count', 50))
            
            final_count = final_collection.size().getInfo()
            print(f"   Final images: {final_count}")
            
            if final_count == 0:
                print(f"   ⚠️ Warning: No images remain")
                return ee.ImageCollection([])

            return final_collection
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return ee.ImageCollection([])

    def calculate_ndti_stats(self, collection):
        """Calculate NDTI statistics"""
        try:
            if collection.size().getInfo() == 0:
                return None
            
            ndti_mean = collection.select('NDTI').mean()
            
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
                'mean_ndti': stats.get('NDTI_mean', 0),
                'min_ndti': stats.get('NDTI_min', 0),
                'max_ndti': stats.get('NDTI_max', 0),
                'std_ndti': stats.get('NDTI_stdDev', 0),
                'image_count': collection.size().getInfo()
            }
            
        except Exception as e:
            print(f"Error calculating NDTI stats: {e}")
            return None

    def process(self):
        """Process data and calculate NDTI statistics"""
        print("\n🌱 Processing NDTI analysis...")
        
        collection = self.Pull_Process_Sentinel_data()
        stats = self.calculate_ndti_stats(collection)
        
        if stats:
            self.ndti_results = stats
            print(f"   ✅ NDTI stats calculated - Mean: {stats['mean_ndti']:.3f}")
        else:
            print(f"   ❌ Failed to calculate NDTI stats")
        
        return self.ndti_results

def main_cropland_analysis():
    """
    Main processing pipeline for Maryland/Delaware region - May 14, 2016
    """
    
    # Define the bounding box for Maryland/Delaware region
    # NW: 39.05° N, -76.20° W
    # NE: 39.05° N, -75.45° W
    # SE: 38.55° N, -75.45° W
    # SW: 38.55° N, -76.20° W
    bounds = {
        'north': 39.05,
        'south': 38.55,
        'west': -76.20,
        'east': -75.45
    }
    
    # Set time period to May 14, 2016
    start_date = '2016-05-14'
    end_date = '2016-06-14'
    
    region_name = "Maryland/Delaware Region"
    
    print("🌾 Cropland NDTI Analysis")
    print("=" * 50)
    print(f"📍 Region: {region_name}")
    print(f"📐 Bounds:")
    print(f"   North: {bounds['north']}° N")
    print(f"   South: {bounds['south']}° N")
    print(f"   West: {bounds['west']}° W")
    print(f"   East: {bounds['east']}° W")
    print(f"📅 Date: {start_date}")
    print(f"📊 NDTI Range: -1 to +1")
    print("\n⚠️  Note: Sentinel-2 data availability:")
    print("   Sentinel-2A launched: June 23, 2015")
    print("   May 14, 2015 is BEFORE Sentinel-2 launch!")
    print("   No Sentinel-2 data available for this date.")
    print("   Consider using dates after June 2015.\n")
    
    try:
        processor = CroplandProcessor(
            bounds=bounds,
            start_date=start_date,
            end_date=end_date,
            Verbose=True,
            ShowPlots=True
        )
        
        ndti_results = processor.process()
        
        if not ndti_results:
            print("❌ No NDTI results obtained")
            print("\n💡 Suggestion: Try a date after June 23, 2015")
            print("   Example: '2015-07-14' or '2015-08-14'")
            return None
        
        # Print detailed results
        print("\n📊 NDTI Results for May 14, 2015:")
        print("=" * 50)
        print(f"Mean NDTI: {ndti_results['mean_ndti']:.3f}")
        print(f"Min NDTI: {ndti_results['min_ndti']:.3f}")
        print(f"Max NDTI: {ndti_results['max_ndti']:.3f}")
        print(f"Std Dev: {ndti_results['std_ndti']:.3f}")
        print(f"Images Used: {ndti_results['image_count']}")
        
        # Create visualization
        create_ndti_visualization(ndti_results, region_name, start_date)
        
        return {
            'processor': processor,
            'ndti_results': ndti_results,
            'region_name': region_name
        }
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        return None

def create_ndti_visualization(ndti_results, region_name, date_str):
    """Create visualization of NDTI results"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'NDTI Analysis - {region_name}\n{date_str}', fontsize=14, fontweight='bold')
        
        # Plot 1: Bar chart of statistics
        ax1 = axes[0]
        stats = ['Mean', 'Min', 'Max', 'Std Dev']
        values = [
            ndti_results['mean_ndti'],
            ndti_results['min_ndti'],
            ndti_results['max_ndti'],
            ndti_results['std_ndti']
        ]
        colors = ['#2E8B57', '#FF6347', '#4169E1', '#FFD700']
        
        bars = ax1.bar(stats, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('NDTI Value')
        ax1.set_title('NDTI Statistics')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_ylim(-1, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold')
        
        # Plot 2: Summary information
        ax2 = axes[1]
        ax2.axis('off')
        
        summary_text = f"""
        NDTI Analysis Summary
        {'='*40}
        
        Region: {region_name}
        Date: {date_str}
        
        NDTI Statistics:
        • Mean NDTI: {ndti_results['mean_ndti']:.4f}
        • Min NDTI:  {ndti_results['min_ndti']:.4f}
        • Max NDTI:  {ndti_results['max_ndti']:.4f}
        • Std Dev:   {ndti_results['std_ndti']:.4f}
        
        Data Quality:
        • Images Used: {ndti_results['image_count']}
        • Analysis Type: Cropland Only
        • Cloud Filtering: Applied
        
        NDTI Interpretation:
        Higher values → More bare soil/tilled
        Lower values → More vegetation/residue
        """
        
        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    try:
        print("🚀 Starting Cropland NDTI Analysis")
        print("📍 Region: Maryland/Delaware Area")
        print("📅 Date: May 14, 2015")
        print("📊 NDTI Range: -1 to +1")
        print()
        
        result = main_cropland_analysis()
        
        if result:
            print("\n" + "="*50)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*50)
            print("✅ Features implemented:")
            print("   • Analyzed Maryland/Delaware bounding box")
            print("   • Date set to May 14, 2015")
            print("   • NDTI calculation for cropland areas")
            print("   • Comprehensive cloud and quality masking")
            print("   • Statistical analysis and visualization")
        else:
            print("\n❌ Analysis failed - check error messages above")
            print("\n⚠️  IMPORTANT: Sentinel-2 was not operational on May 14, 2015")
            print("   Sentinel-2A first image: June 27, 2015")
            print("   Please use a date after June 2015 for valid results")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        print("Make sure Google Earth Engine is properly authenticated and initialized")