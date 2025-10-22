"""
Cropland Analysis with NDTI Tracking for Bihar, India
Analyzes entire Bihar region split into 4 quadrants
Period: November 2017 to March 2018
"""

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    """A class to get land type using ESA WorldCover dataset."""
    def __init__(self, GEE_project_id='tlg-erosion1', DataRes=0.00009, EE_initialized=True):
        if not EE_initialized: 
            ee.Authenticate()
            ee.Initialize(project=GEE_project_id)
        
        worldcover = ee.ImageCollection('ESA/WorldCover/v200')
        self.worldcover = worldcover
        self.DataRes = DataRes

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
    NDTI = (B11 - B12) / (B11 + B12)
    where B11 = SWIR1 (1610 nm) and B12 = SWIR2 (2190 nm)
    """
    try:
        b11 = image.select('B11')  # SWIR1
        b12 = image.select('B12')  # SWIR2
        ndti = b11.subtract(b12).divide(b11.add(b12)).rename('NDTI')
        ndti = ndti.clamp(-1, 1)
        return image.addBands(ndti)
    except Exception as e:
        print(f"Error calculating NDTI: {e}")
        return image

class CroplandProcessor:
    """Cropland Processing with Advanced Masking and NDTI Analysis"""
    
    def __init__(self, bounds, start_date, end_date, quadrant_name, Verbose=True, SentRes=10):
        """Initialize the cropland processor"""
        
        try:
            self.LT = LandType(EE_initialized=True)
            
            self.AoI_geom = get_aoi_from_bounds(
                bounds['north'], bounds['south'], 
                bounds['west'], bounds['east']
            )
            if self.AoI_geom is None:
                raise ValueError("Failed to create area of interest geometry")
            
            result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
            if result is None:
                raise ValueError("Failed to get land cover data")
                
            self.RegionMap = self.LT.Map_LandType(result['image'])
            if self.RegionMap is None:
                raise ValueError("Failed to create cropland mask")
            
            self.bounds = bounds
            self.start_date = start_date
            self.end_date = end_date
            self.Verbose = Verbose
            self.SentRes = SentRes
            self.quadrant_name = quadrant_name
            
            self.ndti_results = {}
            
        except Exception as e:
            print(f"Error initializing CroplandProcessor for {quadrant_name}: {e}")
            raise

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.70):
        """Process Sentinel-2 data with comprehensive masking"""
        
        try:
            def mask_clouds_advanced(img):
                """Enhanced cloud masking"""
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

            if self.Verbose:
                print(f"\n  Processing {self.quadrant_name}: {self.start_date} to {self.end_date}")
            
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            filtered_s2_date_area = (s2
                .filterBounds(self.AoI_geom)
                .filterDate(self.start_date, self.end_date))
            
            initial_count = filtered_s2_date_area.size().getInfo()
            if self.Verbose:
                print(f"    Initial images: {initial_count}")
            
            if initial_count == 0:
                print(f"    ❌ No images found")
                return ee.ImageCollection([])

            filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
            
            after_cloud_filter = filtered_s2_date_area.size().getInfo()
            if self.Verbose:
                print(f"    After cloud pre-filter: {after_cloud_filter}")
            
            if after_cloud_filter == 0:
                print(f"    ❌ No images remain after cloud filtering")
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
            if self.Verbose:
                print(f"    Final images: {final_count}")
            
            if final_count == 0:
                print(f"    ⚠️ Warning: No images remain")
                return ee.ImageCollection([])

            return final_collection
            
        except Exception as e:
            print(f"Error processing data for {self.quadrant_name}: {e}")
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
                'image_count': collection.size().getInfo(),
                'quadrant': self.quadrant_name
            }
            
        except Exception as e:
            print(f"Error calculating NDTI stats for {self.quadrant_name}: {e}")
            return None

    def process(self):
        """Process data and calculate NDTI statistics"""
        collection = self.Pull_Process_Sentinel_data()
        stats = self.calculate_ndti_stats(collection)
        
        if stats:
            self.ndti_results = stats
            if self.Verbose:
                print(f"    ✅ Mean NDTI: {stats['mean_ndti']:.4f}")
        else:
            if self.Verbose:
                print(f"    ❌ Failed to calculate NDTI stats")
        
        return self.ndti_results

def split_bihar_into_quadrants():
    """
    Split Bihar region into 4 quadrants
    Based on the map: approximately 24°N to 27°N, 84°E to 88°E
    """
    # Bihar approximate bounds
    bihar_north = 27.5
    bihar_south = 24.0
    bihar_west = 84.0
    bihar_east = 88.0
    
    # Calculate midpoints
    mid_lat = (bihar_north + bihar_south) / 2
    mid_lon = (bihar_west + bihar_east) / 2
    
    quadrants = {
        'Northwest': {
            'north': bihar_north,
            'south': mid_lat,
            'west': bihar_west,
            'east': mid_lon
        },
        'Northeast': {
            'north': bihar_north,
            'south': mid_lat,
            'west': mid_lon,
            'east': bihar_east
        },
        'Southwest': {
            'north': mid_lat,
            'south': bihar_south,
            'west': bihar_west,
            'east': mid_lon
        },
        'Southeast': {
            'north': mid_lat,
            'south': bihar_south,
            'west': mid_lon,
            'east': bihar_east
        }
    }
    
    return quadrants

def calculate_overall_mean(quadrant_results):
    """Calculate mean statistics across all quadrants"""
    valid_results = [r for r in quadrant_results.values() if r is not None]
    
    if not valid_results:
        return None
    
    mean_stats = {
        'mean_ndti': np.mean([r['mean_ndti'] for r in valid_results]),
        'min_ndti': np.min([r['min_ndti'] for r in valid_results]),
        'max_ndti': np.max([r['max_ndti'] for r in valid_results]),
        'std_ndti': np.mean([r['std_ndti'] for r in valid_results]),
        'total_images': sum([r['image_count'] for r in valid_results]),
        'quadrants_analyzed': len(valid_results)
    }
    
    return mean_stats

def visualize_bihar_results(quadrant_results, overall_mean, start_date, end_date):
    """Create visualization of Bihar NDTI analysis"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Bihar Region NDTI Analysis (4 Quadrants)\n{start_date} to {end_date}', 
                     fontsize=16, fontweight='bold')
        
        # Plot individual quadrants
        quadrant_positions = {
            'Northwest': (0, 0),
            'Northeast': (0, 1),
            'Southwest': (1, 0),
            'Southeast': (1, 1)
        }
        
        colors = ['#2E8B57', '#FF6347', '#4169E1', '#FFD700']
        
        for quad_name, pos in quadrant_positions.items():
            ax = axes[pos[0], pos[1]]
            result = quadrant_results.get(quad_name)
            
            if result:
                stats = ['Mean', 'Min', 'Max', 'Std Dev']
                values = [
                    result['mean_ndti'],
                    result['min_ndti'],
                    result['max_ndti'],
                    result['std_ndti']
                ]
                
                bars = ax.bar(stats, values, color=colors, alpha=0.7, edgecolor='black')
                ax.set_ylabel('NDTI Value')
                ax.set_title(f'{quad_name}\n(Images: {result["image_count"]})', fontweight='bold')
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                ax.set_ylim(-1, 1)
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}',
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=8, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'{quad_name}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Overall mean plot
        ax_mean = axes[0, 2]
        if overall_mean:
            stats = ['Mean', 'Min', 'Max', 'Avg Std']
            values = [
                overall_mean['mean_ndti'],
                overall_mean['min_ndti'],
                overall_mean['max_ndti'],
                overall_mean['std_ndti']
            ]
            
            bars = ax_mean.bar(stats, values, color=colors, alpha=0.7, edgecolor='black')
            ax_mean.set_ylabel('NDTI Value')
            ax_mean.set_title('Bihar Overall Statistics', fontweight='bold', fontsize=12)
            ax_mean.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax_mean.set_ylim(-1, 1)
            ax_mean.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_mean.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}',
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=9, fontweight='bold')
        
        # Summary text
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        
        if overall_mean:
            summary_text = f"""
Bihar NDTI Summary
{'='*35}

Period: {start_date} to {end_date}

Overall Statistics:
• Mean NDTI: {overall_mean['mean_ndti']:.4f}
• Min NDTI:  {overall_mean['min_ndti']:.4f}
• Max NDTI:  {overall_mean['max_ndti']:.4f}
• Avg Std:   {overall_mean['std_ndti']:.4f}

Data Coverage:
• Quadrants: {overall_mean['quadrants_analyzed']}/4
• Total Images: {overall_mean['total_images']}

Individual Quadrant Means:
"""
            for quad_name in ['Northwest', 'Northeast', 'Southwest', 'Southeast']:
                result = quadrant_results.get(quad_name)
                if result:
                    summary_text += f"• {quad_name}: {result['mean_ndti']:.4f}\n"
                else:
                    summary_text += f"• {quad_name}: No Data\n"
            
            summary_text += """
NDTI Interpretation:
Higher → Bare soil/tilled
Lower → Vegetation/residue

Formula: (B11-B12)/(B11+B12)
            """
            
            ax_summary.text(0.05, 0.5, summary_text, transform=ax_summary.transAxes,
                          fontsize=10, verticalalignment='center',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main_bihar_analysis():
    """Main processing pipeline for Bihar region split into 4 quadrants"""
    
    # Set time period
    start_date = '2017-11-01'
    end_date = '2018-03-02'
    
    print("🌾 Bihar Region NDTI Analysis")
    print("=" * 60)
    print(f"📍 Region: Bihar, India (4 Quadrants)")
    print(f"📅 Period: {start_date} to {end_date} (Dry Season)")
    print(f"📊 NDTI Formula: (B11-B12)/(B11+B12)")
    print(f"📊 NDTI Range: -1 to +1")
    print()
    
    try:
        # Get quadrant boundaries
        quadrants = split_bihar_into_quadrants()
        
        print("📍 Quadrant Boundaries:")
        for name, bounds in quadrants.items():
            print(f"  {name}: {bounds['south']:.1f}°N to {bounds['north']:.1f}°N, "
                  f"{bounds['west']:.1f}°E to {bounds['east']:.1f}°E")
        print()
        
        # Process each quadrant
        quadrant_results = {}
        
        for quad_name, bounds in quadrants.items():
            print(f"🔄 Processing {quad_name}...")
            try:
                processor = CroplandProcessor(
                    bounds=bounds,
                    start_date=start_date,
                    end_date=end_date,
                    quadrant_name=quad_name,
                    Verbose=True
                )
                
                result = processor.process()
                quadrant_results[quad_name] = result
                
            except Exception as e:
                print(f"  ❌ Error processing {quad_name}: {e}")
                quadrant_results[quad_name] = None
        
        # Calculate overall mean
        print("\n" + "="*60)
        print("📊 Calculating Overall Bihar Statistics...")
        overall_mean = calculate_overall_mean(quadrant_results)
        
        if overall_mean:
            print("\n✅ BIHAR OVERALL RESULTS:")
            print("="*60)
            print(f"Mean NDTI across Bihar: {overall_mean['mean_ndti']:.4f}")
            print(f"Min NDTI: {overall_mean['min_ndti']:.4f}")
            print(f"Max NDTI: {overall_mean['max_ndti']:.4f}")
            print(f"Average Std Dev: {overall_mean['std_ndti']:.4f}")
            print(f"Total Images Used: {overall_mean['total_images']}")
            print(f"Quadrants Successfully Analyzed: {overall_mean['quadrants_analyzed']}/4")
            print()
            
            # Print individual quadrant means
            print("📋 Individual Quadrant Means:")
            for quad_name in ['Northwest', 'Northeast', 'Southwest', 'Southeast']:
                result = quadrant_results.get(quad_name)
                if result:
                    print(f"  {quad_name:12s}: {result['mean_ndti']:.4f} (n={result['image_count']})")
                else:
                    print(f"  {quad_name:12s}: No Data")
            
            # Create visualization
            visualize_bihar_results(quadrant_results, overall_mean, start_date, end_date)
            
            return {
                'quadrant_results': quadrant_results,
                'overall_mean': overall_mean
            }
        else:
            print("❌ Failed to calculate overall statistics")
            return None
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        return None

if __name__ == "__main__":
    try:
        print("🚀 Starting Bihar Cropland NDTI Analysis")
        print("📍 Region: Bihar, India (4 Quadrants)")
        print("📅 Period: November 2017 - March 2018 (Dry Season)")
        print()
        
        result = main_bihar_analysis()
        
        if result:
            print("\n" + "="*60)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("✅ Features:")
            print("   • Analyzed entire Bihar region")
            print("   • Split into 4 quadrants for detailed analysis")
            print("   • Calculated individual quadrant statistics")
            print("   • Computed overall Bihar mean NDTI")
            print("   • Generated comprehensive visualizations")
        else:
            print("\n❌ Analysis incomplete - check error messages above")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        print("Make sure Google Earth Engine is properly authenticated and initialized")