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

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.60):
        """
        Process Sentinel-2 data with comprehensive masking
        Relaxed thresholds to capture more images
        """
        
        try:
            def mask_clouds_advanced(img):
                """
                More relaxed cloud masking to capture more images
                Still removes obvious clouds but keeps more data
                """
                try:
                    cs = img.select(QA_BAND)
                    cloud_mask = cs.gte(CLEAR_THRESHOLD)  # Now 0.60 (relaxed from 0.70)
                    scl = img.select('SCL')
                    
                    # Only mask the most problematic classes
                    cloud_shadow_mask = scl.neq(3)   # Cloud shadows
                    cloud_high_mask = scl.neq(9)     # High probability clouds
                    cirrus_mask = scl.neq(10)        # Cirrus
                    water_mask = scl.neq(6)          # Water
                    saturated_mask = scl.neq(1)      # Saturated/defective
                    
                    # Allow medium clouds (8), snow (11), and dark areas (2) for more data
                    combined_mask = (cloud_mask
                                    .And(cloud_shadow_mask)
                                    .And(cloud_high_mask)
                                    .And(cirrus_mask)
                                    .And(water_mask)
                                    .And(saturated_mask))
                    
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

            # More relaxed cloud filtering to get more images
            filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
            
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
            
            # Lower threshold for valid pixels to include more images
            final_collection = ndti_collection_with_counts.filter(
                ee.Filter.gt('valid_pixel_count', 10))  # Reduced from 50 to 10
            
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
        """
        Calculate NDTI statistics with histogram distribution:
        - Mean of all pixel means across time
        - Mean of pixel-wise minimums
        - Mean of pixel-wise maximums
        - Histogram distribution with defined bins
        """
        try:
            if collection.size().getInfo() == 0:
                return None
            
            # Calculate temporal mean (average NDTI over time period)
            ndti_mean_image = collection.select('NDTI').mean()
            
            # Calculate pixel-wise min and max across time
            ndti_min_image = collection.select('NDTI').min()
            ndti_max_image = collection.select('NDTI').max()
            
            # Get mean NDTI (average of all pixel means)
            mean_stats = ndti_mean_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Get mean of pixel-wise minimums (not the absolute minimum)
            min_stats = ndti_min_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Get mean of pixel-wise maximums (not the absolute maximum)
            max_stats = ndti_max_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Get histogram with fixed bins: -1.0 to 1.0 in 0.2 intervals
            # Bins: [-1.0, -0.8), [-0.8, -0.6), [-0.6, -0.4), [-0.4, -0.2), 
            #       [-0.2, 0.0), [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
            histogram = ndti_mean_image.reduceRegion(
                reducer=ee.Reducer.fixedHistogram(-1.0, 1.0, 10),  # 10 bins of 0.2 width
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            return {
                'mean_ndti': mean_stats.get('NDTI_mean', 0),
                'min_ndti': min_stats.get('NDTI', 0),  # Mean of pixel minimums
                'max_ndti': max_stats.get('NDTI', 0),  # Mean of pixel maximums
                'std_ndti': mean_stats.get('NDTI_stdDev', 0),
                'image_count': collection.size().getInfo(),
                'quadrant': self.quadrant_name,
                'histogram': histogram.get('NDTI', [])
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
    Split Bihar region into 4 quadrants with extended coverage
    Based on the map: approximately 24°N to 27.5°N, 83.5°E to 88.5°E
    Extended boundaries to ensure full coverage
    """
    # Bihar approximate bounds - extended for better coverage
    bihar_north = 27.5
    bihar_south = 24.0
    bihar_west = 83.5   # Extended west
    bihar_east = 88.5   # Extended east
    
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

def parse_histogram(histogram_data):
    """Parse GEE histogram format into bins and counts"""
    if not histogram_data or len(histogram_data) == 0:
        return None, None
    
    bins = []
    counts = []
    
    for entry in histogram_data:
        if len(entry) >= 2:
            bins.append(entry[0])  # Bin start value
            counts.append(entry[1])  # Count in that bin
    
    return bins, counts

def visualize_bihar_results(quadrant_results, overall_mean, start_date, end_date):
    """Create visualization focusing on NDTI distribution histograms"""
    try:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        fig.suptitle(f'Bihar Region NDTI Distribution Analysis\n{start_date} to {end_date}', 
                     fontsize=18, fontweight='bold')
        
        # Plot individual quadrant histograms (top 2 rows)
        quadrant_positions = {
            'Northwest': (0, 0),
            'Northeast': (0, 1),
            'Southwest': (0, 2),
            'Southeast': (0, 3)
        }
        
        # Define bin edges for labeling (10 bins from -1.0 to 1.0)
        bin_edges = np.arange(-1.0, 1.2, 0.2)
        bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]
        
        for quad_name, pos in quadrant_positions.items():
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            result = quadrant_results.get(quad_name)
            
            if result and 'histogram' in result and result['histogram']:
                bins, counts = parse_histogram(result['histogram'])
                
                if bins and counts:
                    # Create bar chart
                    colors = plt.cm.viridis(np.linspace(0, 1, len(counts)))
                    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.8, edgecolor='black')
                    
                    ax.set_xlabel('NDTI Interval', fontweight='bold', fontsize=9)
                    ax.set_ylabel('Pixel Count', fontweight='bold', fontsize=9)
                    ax.set_title(f'{quad_name}\nMean: {result["mean_ndti"]:.4f} | n={result["image_count"]}', 
                               fontweight='bold', fontsize=10)
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=7)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add count labels on bars
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(count)}',
                                   ha='center', va='bottom', fontsize=7)
                else:
                    ax.text(0.5, 0.5, f'{quad_name}\nNo histogram data', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'{quad_name}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Row 2: Quadrant statistics bars
        stats_positions = {
            'Northwest': (1, 0),
            'Northeast': (1, 1),
            'Southwest': (1, 2),
            'Southeast': (1, 3)
        }
        
        for quad_name, pos in stats_positions.items():
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            result = quadrant_results.get(quad_name)
            
            if result:
                stats = ['Mean', 'Min', 'Max']
                values = [result['mean_ndti'], result['min_ndti'], result['max_ndti']]
                colors_stat = ['#2E8B57', '#FF6347', '#4169E1']
                
                bars = ax.bar(stats, values, color=colors_stat, alpha=0.7, edgecolor='black')
                ax.set_ylabel('NDTI Value', fontweight='bold', fontsize=9)
                ax.set_title(f'{quad_name} Stats', fontweight='bold', fontsize=10)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                ax.set_ylim(-0.2, 0.5)
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}',
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=8, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Row 3: Overall Bihar histogram (spans 3 columns)
        ax_overall_hist = fig.add_subplot(gs[2, :3])
        
        # Combine all quadrant histograms
        all_bins = None
        all_counts = np.zeros(10)  # 10 bins
        quadrants_with_data = 0
        
        for quad_name in ['Northwest', 'Northeast', 'Southwest', 'Southeast']:
            result = quadrant_results.get(quad_name)
            if result and 'histogram' in result and result['histogram']:
                bins, counts = parse_histogram(result['histogram'])
                if bins and counts:
                    all_counts += np.array(counts[:10])  # Ensure we only take 10 bins
                    quadrants_with_data += 1
        
        if quadrants_with_data > 0:
            colors = plt.cm.plasma(np.linspace(0, 1, len(all_counts)))
            bars = ax_overall_hist.bar(range(len(all_counts)), all_counts, 
                                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax_overall_hist.set_xlabel('NDTI Interval', fontweight='bold', fontsize=11)
            ax_overall_hist.set_ylabel('Total Pixel Count (All Quadrants)', fontweight='bold', fontsize=11)
            ax_overall_hist.set_title('Bihar Overall NDTI Distribution', fontweight='bold', fontsize=13)
            ax_overall_hist.set_xticks(range(len(all_counts)))
            ax_overall_hist.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
            ax_overall_hist.grid(True, alpha=0.3, axis='y')
            
            # Add count labels
            for bar, count in zip(bars, all_counts):
                height = bar.get_height()
                if height > 0:
                    ax_overall_hist.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{int(count)}',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add mean line
            if overall_mean:
                mean_val = overall_mean['mean_ndti']
                # Convert mean to bin position
                bin_pos = (mean_val + 1.0) / 0.2  # Map -1 to 1 range to 0-10 bins
                ax_overall_hist.axvline(x=bin_pos, color='red', linestyle='--', 
                                       linewidth=3, label=f"Mean: {mean_val:.4f}", alpha=0.8)
                ax_overall_hist.legend(fontsize=10, loc='upper right')
        else:
            ax_overall_hist.text(0.5, 0.5, 'No histogram data available', 
                                ha='center', va='center', transform=ax_overall_hist.transAxes,
                                fontsize=12)
        
        # Row 3: Summary statistics (rightmost column)
        ax_summary = fig.add_subplot(gs[2, 3])
        ax_summary.axis('off')
        
        if overall_mean:
            summary_text = f"""
BIHAR OVERALL STATISTICS
{'='*28}

Mean:  {overall_mean['mean_ndti']:.6f}
Min:   {overall_mean['min_ndti']:.6f}
Max:   {overall_mean['max_ndti']:.6f}
Std:   {overall_mean['std_ndti']:.6f}

Ground Truth:
Mean:  0.142487
Min:   0.064551
Max:   0.292054

Difference:
ΔMean: {abs(overall_mean['mean_ndti'] - 0.142487):.6f}
ΔMin:  {abs(overall_mean['min_ndti'] - 0.064551):.6f}
ΔMax:  {abs(overall_mean['max_ndti'] - 0.292054):.6f}

Data:
Quadrants: {overall_mean['quadrants_analyzed']}/4
Images: {overall_mean['total_images']}

Histogram Bins:
10 intervals of 0.2 width
Range: -1.0 to +1.0
            """
            
            ax_summary.text(0.05, 0.5, summary_text, transform=ax_summary.transAxes,
                          fontsize=9, verticalalignment='center',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def main_bihar_analysis():
    """Main processing pipeline for Bihar region split into 4 quadrants"""
    
    # Set time period - Extended to capture more images
    start_date = '2017-10-15'  # Start earlier (mid-October)
    end_date = '2018-03-15'    # End later (mid-March)
    
    print("🌾 Bihar Region NDTI Analysis (EXTENDED)")
    print("=" * 60)
    print(f"📍 Region: Bihar, India (4 Quadrants - Extended Coverage)")
    print(f"📅 Period: {start_date} to {end_date} (Extended Dry Season)")
    print(f"📊 NDTI Formula: (B11-B12)/(B11+B12)")
    print(f"📊 NDTI Range: -1 to +1")
    print()
    print("🔧 Optimizations for More Images:")
    print("   • Extended time period (Oct 15 - Mar 15)")
    print("   • Relaxed cloud threshold (60% vs 70%)")
    print("   • Relaxed clear sky threshold (0.60 vs 0.70)")
    print("   • Lower pixel count requirement (10 vs 50)")
    print("   • Less restrictive masking (allows medium clouds)")
    print("   • Extended geographic boundaries")
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
            print(f"Mean NDTI across Bihar: {overall_mean['mean_ndti']:.6f}")
            print(f"Mean Min NDTI: {overall_mean['min_ndti']:.6f}")
            print(f"Mean Max NDTI: {overall_mean['max_ndti']:.6f}")
            print(f"Average Std Dev: {overall_mean['std_ndti']:.6f}")
            print(f"Total Images Used: {overall_mean['total_images']}")
            print(f"Quadrants Successfully Analyzed: {overall_mean['quadrants_analyzed']}/4")
            print()
            
            print("📊 COMPARISON WITH GROUND TRUTH:")
            print("="*60)
            print(f"                  Our Results    Ground Truth    Difference")
            print(f"Mean NDTI:        {overall_mean['mean_ndti']:.6f}       0.142487        {abs(overall_mean['mean_ndti'] - 0.142487):.6f}")
            print(f"Mean Min:         {overall_mean['min_ndti']:.6f}       0.064551        {abs(overall_mean['min_ndti'] - 0.064551):.6f}")
            print(f"Mean Max:         {overall_mean['max_ndti']:.6f}       0.292054        {abs(overall_mean['max_ndti'] - 0.292054):.6f}")
            print()
            
            # Print individual quadrant means
            print("📋 Individual Quadrant Means:")
            for quad_name in ['Northwest', 'Northeast', 'Southwest', 'Southeast']:
                result = quadrant_results.get(quad_name)
                if result:
                    print(f"  {quad_name:12s}: {result['mean_ndti']:.6f} (n={result['image_count']})")
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
        print("🚀 Starting Bihar Cropland NDTI Analysis (EXTENDED)")
        print("📍 Region: Bihar, India (4 Quadrants)")
        print("📅 Period: October 2017 - March 2018 (Extended Dry Season)")
        print("🔧 Mode: Maximum Image Capture (Relaxed Filters)")
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