"""
Optimized Cropland Analysis with NDTI Tracking for Bihar, India
Analyzes entire Bihar region using single 13-vertex polygon
Period: Extended for maximum image capture
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

def get_bihar_polygon():
    """
    Create Bihar polygon geometry based on actual state boundaries
    Uses precise polygon coordinates
    """
    try:
        # Bihar polygon coordinates (13 vertices)
        coordinates = [[
            [83.0, 27.5],
            [84.0, 27.6],
            [85.5, 27.4],
            [87.0, 26.9],
            [87.5, 26.5],
            [87.6, 26.0],
            [87.0, 25.2],
            [86.0, 24.6],
            [84.8, 24.3],
            [83.3, 24.4],
            [83.0, 25.0],
            [82.7, 25.8],
            [83.0, 27.5]
        ]]
        
        return ee.Geometry.Polygon(coordinates)
    except Exception as e:
        print(f"Error creating Bihar polygon: {e}")
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
    
    def __init__(self, geometry, start_date, end_date, Verbose=True, SentRes=10):
        """Initialize the cropland processor with polygon geometry"""
        
        try:
            self.LT = LandType(EE_initialized=True)
            
            self.AoI_geom = geometry
            if self.AoI_geom is None:
                raise ValueError("Failed to create area of interest geometry")
            
            result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
            if result is None:
                raise ValueError("Failed to get land cover data")
                
            self.RegionMap = self.LT.Map_LandType(result['image'])
            if self.RegionMap is None:
                raise ValueError("Failed to create cropland mask")
            
            self.start_date = start_date
            self.end_date = end_date
            self.Verbose = Verbose
            self.SentRes = SentRes
            
            self.ndti_results = {}
            
        except Exception as e:
            print(f"Error initializing CroplandProcessor: {e}")
            raise

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.40):
        """
        Process Sentinel-2 data with ultra-permissive masking for maximum image capture
        """
        
        try:
            def mask_clouds_ultra_minimal(img):
                """
                Ultra-minimal cloud masking - capture as many images as possible
                Only remove completely unusable pixels
                """
                try:
                    # Very relaxed cloud score threshold
                    cs = img.select(QA_BAND)
                    cloud_mask = cs.gte(CLEAR_THRESHOLD)  # 0.40 - ultra permissive
                    
                    scl = img.select('SCL')
                    
                    # Only mask absolutely unusable pixels
                    saturated_mask = scl.neq(1)
                    
                    combined_mask = cloud_mask.And(saturated_mask)
                    
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
                        maxPixels=1e10,
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
                print(f"\n  Processing Bihar: {self.start_date} to {self.end_date}")
            
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            filtered_s2_date_area = (s2
                .filterBounds(self.AoI_geom)
                .filterDate(self.start_date, self.end_date))
            
            initial_count = filtered_s2_date_area.size().getInfo()
            if self.Verbose:
                print(f"    Initial images in date range: {initial_count}")
            
            if initial_count == 0:
                print(f"    ❌ No images found")
                return ee.ImageCollection([])

            # Ultra-relaxed cloud filtering - allow up to 95% cloud cover
            filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 95))
            
            after_cloud_filter = filtered_s2_date_area.size().getInfo()
            if self.Verbose:
                print(f"    After cloud pre-filter (<95%): {after_cloud_filter}")
            
            if after_cloud_filter == 0:
                print(f"    ❌ No images remain after cloud filtering")
                return ee.ImageCollection([])

            filtered_s2 = (filtered_s2_date_area
                .linkCollection(csPlus, [QA_BAND])
                .map(mask_clouds_ultra_minimal))

            land_masked_collection = filtered_s2.map(apply_landtype_mask)
            ndti_collection = land_masked_collection.map(calculate_ndti)
            ndti_collection_with_counts = ndti_collection.map(set_pixel_count)
            
            # Very low threshold for valid pixels - accept almost anything
            final_collection = ndti_collection_with_counts.filter(
                ee.Filter.gt('valid_pixel_count', 1))
            
            final_count = final_collection.size().getInfo()
            if self.Verbose:
                print(f"    Final usable images: {final_count}")
            
            if final_count == 0:
                print(f"    ⚠️ Warning: No images remain after processing")
                return ee.ImageCollection([])

            return final_collection
            
        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return ee.ImageCollection([])

    def calculate_ndti_stats(self, collection):
        """
        Calculate NDTI statistics with improved histogram distribution
        """
        try:
            if collection.size().getInfo() == 0:
                return None
            
            # Calculate temporal mean
            ndti_mean_image = collection.select('NDTI').mean()
            
            # Calculate pixel-wise min and max across time
            ndti_min_image = collection.select('NDTI').min()
            ndti_max_image = collection.select('NDTI').max()
            
            # Get mean NDTI
            mean_stats = ndti_mean_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e10,
                bestEffort=True
            ).getInfo()
            
            # Get mean of pixel-wise minimums
            min_stats = ndti_min_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e10,
                bestEffort=True
            ).getInfo()
            
            # Get mean of pixel-wise maximums
            max_stats = ndti_max_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e10,
                bestEffort=True
            ).getInfo()
            
            # Improved histogram with focused binning on data range (0 to 0.4)
            # 60% of bins between 0 and 0.4, remaining 40% for rest
            bin_edges = [-1.0, -0.5, -0.2, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]
            
            histogram = ndti_mean_image.reduceRegion(
                reducer=ee.Reducer.fixedHistogram(-1.0, 1.0, 50),  # Get fine histogram first
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e10,
                bestEffort=True
            ).getInfo()
            
            return {
                'mean_ndti': mean_stats.get('NDTI_mean', 0),
                'min_ndti': min_stats.get('NDTI', 0),
                'max_ndti': max_stats.get('NDTI', 0),
                'std_ndti': mean_stats.get('NDTI_stdDev', 0),
                'image_count': collection.size().getInfo(),
                'histogram': histogram.get('NDTI', [])
            }
            
        except Exception as e:
            print(f"Error calculating NDTI stats: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process(self):
        """Process data and calculate NDTI statistics"""
        collection = self.Pull_Process_Sentinel_data()
        stats = self.calculate_ndti_stats(collection)
        
        if stats:
            self.ndti_results = stats
            if self.Verbose:
                print(f"    ✅ Mean NDTI: {stats['mean_ndti']:.6f}")
                print(f"    ✅ Min NDTI:  {stats['min_ndti']:.6f}")
                print(f"    ✅ Max NDTI:  {stats['max_ndti']:.6f}")
        else:
            if self.Verbose:
                print(f"    ❌ Failed to calculate NDTI stats")
        
        return self.ndti_results

def create_focused_histogram(histogram_data, bin_edges):
    """
    Convert GEE fine histogram to focused custom bins
    60% of bins between 0 and 0.4, remaining for rest of range
    """
    if not histogram_data or len(histogram_data) == 0:
        return None, None
    
    # Extract values and counts from GEE histogram
    values = []
    counts = []
    for entry in histogram_data:
        if len(entry) >= 2:
            values.append(entry[0])
            counts.append(entry[1])
    
    if not values:
        return None, None
    
    # Create custom histogram with focused bins
    custom_counts = []
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Sum counts for values falling in this bin
        bin_count = 0
        for val, cnt in zip(values, counts):
            if bin_start <= val < bin_end:
                bin_count += cnt
        
        custom_counts.append(bin_count)
    
    return bin_edges, custom_counts

def load_ground_truth_histogram(csv_path, bin_edges):
    """
    Load ground truth data from CSV and create histogram with custom bins
    CSV has columns: NDTI_p0_peak (min), NDTI_p50_peak (mean), NDTI_p100_peak (max)
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        print(f"   CSV columns found: {df.columns.tolist()}")
        
        # Check for the specific columns we need
        required_cols = {
            'mean': 'NDTI_p50_peak',
            'min': 'NDTI_p0_peak', 
            'max': 'NDTI_p100_peak'
        }
        
        # Verify all required columns exist
        missing_cols = []
        for stat_name, col_name in required_cols.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"   Available columns: {df.columns.tolist()}")
            return None, None
        
        # Extract the values
        mean_values = df['NDTI_p50_peak'].dropna().values
        min_values = df['NDTI_p0_peak'].dropna().values
        max_values = df['NDTI_p100_peak'].dropna().values
        
        print(f"   ✅ Loaded {len(mean_values)} rows of data")
        print(f"   Mean NDTI range: {np.min(mean_values):.6f} to {np.max(mean_values):.6f}")
        print(f"   Min NDTI range:  {np.min(min_values):.6f} to {np.max(min_values):.6f}")
        print(f"   Max NDTI range:  {np.min(max_values):.6f} to {np.max(max_values):.6f}")
        
        # Create histogram using the mean (p50) values for distribution
        counts, _ = np.histogram(mean_values, bins=bin_edges)
        
        # Calculate statistics from the appropriate columns
        stats = {
            'mean': np.mean(mean_values),  # Overall mean of p50 values
            'min': np.mean(min_values),    # Overall mean of p0 values
            'max': np.mean(max_values),    # Overall mean of p100 values
            'std': np.std(mean_values)     # Std dev of p50 values
        }
        
        print(f"   Ground Truth Statistics:")
        print(f"   • Mean (avg of p50): {stats['mean']:.6f}")
        print(f"   • Min (avg of p0):   {stats['min']:.6f}")
        print(f"   • Max (avg of p100): {stats['max']:.6f}")
        print(f"   • Std Dev:           {stats['std']:.6f}")
        
        return counts, stats
        
    except Exception as e:
        print(f"❌ Error loading ground truth CSV: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_results(result, ground_truth_counts, ground_truth_stats, start_date, end_date, csv_path):
    """
    Create comprehensive visualization with focused histogram binning
    """
    try:
        # Define focused bin edges: 60% between 0-0.4, 40% for rest
        bin_edges = [-1.0, -0.5, -0.2, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]
        bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 1.2])
        
        fig.suptitle(f'Bihar Region NDTI Analysis - Optimized Single Polygon\n{start_date} to {end_date}', 
                     fontsize=18, fontweight='bold')
        
        # Convert our histogram to focused bins
        _, our_counts = create_focused_histogram(result['histogram'], bin_edges)
        
        # Plot 1: Our computed histogram
        ax1 = fig.add_subplot(gs[0, 0])
        if our_counts:
            colors = plt.cm.viridis(np.linspace(0, 1, len(our_counts)))
            bars = ax1.bar(range(len(our_counts)), our_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax1.set_xlabel('NDTI Interval', fontweight='bold', fontsize=11)
            ax1.set_ylabel('Pixel Count', fontweight='bold', fontsize=11)
            ax1.set_title(f'Our Results (n={result["image_count"]} images)\nMean: {result["mean_ndti"]:.6f}', 
                         fontweight='bold', fontsize=12)
            ax1.set_xticks(range(len(our_counts)))
            ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, count in zip(bars, our_counts):
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(count)}',
                           ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Ground truth histogram
        ax2 = fig.add_subplot(gs[0, 1])
        if ground_truth_counts is not None and ground_truth_stats is not None:
            colors_gt = plt.cm.plasma(np.linspace(0, 1, len(ground_truth_counts)))
            bars_gt = ax2.bar(range(len(ground_truth_counts)), ground_truth_counts, 
                             color=colors_gt, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax2.set_xlabel('NDTI Interval', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Pixel Count', fontweight='bold', fontsize=11)
            ax2.set_title(f'Ground Truth Data\nMean: {ground_truth_stats["mean"]:.6f}', 
                         fontweight='bold', fontsize=12)
            ax2.set_xticks(range(len(ground_truth_counts)))
            ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, count in zip(bars_gt, ground_truth_counts):
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(count)}',
                           ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, f'Ground Truth Not Available\nProvide CSV path', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        
        # Plot 3: Comparison bar chart
        ax3 = fig.add_subplot(gs[1, :])
        if ground_truth_stats:
            categories = ['Mean NDTI', 'Min NDTI', 'Max NDTI', 'Std Dev']
            our_values = [result['mean_ndti'], result['min_ndti'], result['max_ndti'], result['std_ndti']]
            gt_values = [ground_truth_stats['mean'], ground_truth_stats['min'], 
                        ground_truth_stats['max'], ground_truth_stats['std']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, our_values, width, label='Our Results', 
                           color='steelblue', alpha=0.8, edgecolor='black')
            bars2 = ax3.bar(x + width/2, gt_values, width, label='Ground Truth', 
                           color='coral', alpha=0.8, edgecolor='black')
            
            ax3.set_xlabel('Metric', fontweight='bold', fontsize=12)
            ax3.set_ylabel('NDTI Value', fontweight='bold', fontsize=12)
            ax3.set_title('Statistical Comparison: Our Results vs Ground Truth', fontweight='bold', fontsize=13)
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories, fontsize=11)
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        if ground_truth_stats:
            summary_text = f"""
BIHAR NDTI ANALYSIS SUMMARY
{'='*80}

Dataset Statistics:                Our Results         Ground Truth        Absolute Diff
{'─'*80}
Mean NDTI:                        {result['mean_ndti']:>10.6f}        {ground_truth_stats['mean']:>10.6f}        {abs(result['mean_ndti'] - ground_truth_stats['mean']):>10.6f}
Min NDTI:                         {result['min_ndti']:>10.6f}        {ground_truth_stats['min']:>10.6f}        {abs(result['min_ndti'] - ground_truth_stats['min']):>10.6f}
Max NDTI:                         {result['max_ndti']:>10.6f}        {ground_truth_stats['max']:>10.6f}        {abs(result['max_ndti'] - ground_truth_stats['max']):>10.6f}
Std Dev:                          {result['std_ndti']:>10.6f}        {ground_truth_stats['std']:>10.6f}        {abs(result['std_ndti'] - ground_truth_stats['std']):>10.6f}

Data Collection:
────────────────────────────────────────────────────────────────────────────────
Images Used:                      {result['image_count']}
Analysis Period:                  {start_date} to {end_date}
Region:                          Bihar State (13-vertex polygon)
Ground Truth Source:              {csv_path if csv_path else 'Not provided'}

Histogram Binning Strategy:
────────────────────────────────────────────────────────────────────────────────
Total Bins:                      {len(bin_edges) - 1}
Focused Range (0 to 0.4):        60% of bins (10 bins with 0.05 width)
Remaining Range:                 40% of bins (5 bins, variable width)
Rationale:                       Most cropland NDTI values fall between 0 and 0.4
            """
            
            ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('bihar_ndti_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'bihar_ndti_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def main_bihar_analysis(csv_path=None):
    """
    Main processing pipeline for Bihar region (single polygon)
    
    Args:
        csv_path: Path to CSV file with ground truth NDTI data
    """
    
    # Extended time period for maximum image capture
    start_date = '2017-09-01'  # Extended earlier
    end_date = '2018-05-01'    # Extended later
    
    print("🌾 Bihar Region NDTI Analysis - OPTIMIZED")
    print("=" * 80)
    print(f"📍 Region: Bihar, India (Single 13-vertex polygon)")
    print(f"📅 Period: {start_date} to {end_date} (8 months - EXTENDED)")
    print(f"📊 NDTI Formula: (B11-B12)/(B11+B12)")
    print(f"📊 NDTI Range: -1 to +1")
    print()
    print("🔧 ULTRA-AGGRESSIVE IMAGE CAPTURE:")
    print("   • Extended period: Sep 2017 - May 2018 (8 months)")
    print("   • Cloud threshold: 95% (was 90%)")
    print("   • Clear sky threshold: 0.40 (was 0.50)")
    print("   • Valid pixel minimum: 1 pixel (was 5)")
    print("   • Minimal masking: Only saturated pixels")
    print("   • No quadrant splitting - process entire region at once")
    print()
    print("📊 IMPROVED HISTOGRAM BINNING:")
    print("   • 60% of bins between 0.0 and 0.4 (where data concentrates)")
    print("   • 40% of bins for remaining range")
    print("   • Total: 15 bins with variable widths")
    print()
    
    try:
        # Get Bihar polygon
        bihar_polygon = get_bihar_polygon()
        
        if bihar_polygon is None:
            raise ValueError("Failed to create Bihar polygon")
        
        print("📍 Bihar Polygon Created (13 vertices)")
        print()
        
        # Load ground truth data if provided
        ground_truth_counts = None
        ground_truth_stats = None
        
        if csv_path:
            print(f"📂 Loading ground truth data from: {csv_path}")
            bin_edges = [-1.0, -0.5, -0.2, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]
            ground_truth_counts, ground_truth_stats = load_ground_truth_histogram(csv_path, bin_edges)
            
            if ground_truth_stats:
                print("✅ Ground truth data loaded successfully")
                print(f"   Mean: {ground_truth_stats['mean']:.6f}")
                print(f"   Min:  {ground_truth_stats['min']:.6f}")
                print(f"   Max:  {ground_truth_stats['max']:.6f}")
            else:
                print("⚠️ Failed to load ground truth data")
            print()
        
        # Process entire Bihar region
        print("🔄 Processing entire Bihar region...")
        processor = CroplandProcessor(
            geometry=bihar_polygon,
            start_date=start_date,
            end_date=end_date,
            Verbose=True
        )
        
        result = processor.process()
        
        if result:
            print("\n" + "="*80)
            print("✅ BIHAR ANALYSIS RESULTS:")
            print("="*80)
            print(f"Mean NDTI:        {result['mean_ndti']:.6f}")
            print(f"Min NDTI:         {result['min_ndti']:.6f}")
            print(f"Max NDTI:         {result['max_ndti']:.6f}")
            print(f"Std Dev:          {result['std_ndti']:.6f}")
            print(f"Images Used:      {result['image_count']}")
            print()
            
            if ground_truth_stats:
                print("📊 COMPARISON WITH GROUND TRUTH:")
                print("="*80)
                print(f"Metric          Our Results    Ground Truth    Absolute Diff    % Error")
                print(f"Mean NDTI:      {result['mean_ndti']:.6f}       {ground_truth_stats['mean']:.6f}        {abs(result['mean_ndti'] - ground_truth_stats['mean']):.6f}        {abs((result['mean_ndti'] - ground_truth_stats['mean'])/ground_truth_stats['mean']*100):.2f}%")
                print(f"Min NDTI:       {result['min_ndti']:.6f}       {ground_truth_stats['min']:.6f}        {abs(result['min_ndti'] - ground_truth_stats['min']):.6f}        {abs((result['min_ndti'] - ground_truth_stats['min'])/ground_truth_stats['min']*100):.2f}%")
                print(f"Max NDTI:       {result['max_ndti']:.6f}       {ground_truth_stats['max']:.6f}        {abs(result['max_ndti'] - ground_truth_stats['max']):.6f}        {abs((result['max_ndti'] - ground_truth_stats['max'])/ground_truth_stats['max']*100):.2f}%")
                print()
            
            # Create visualization
            visualize_results(result, ground_truth_counts, ground_truth_stats, 
                            start_date, end_date, csv_path)
            
            return result
        else:
            print("❌ Failed to calculate NDTI statistics")
            return None
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        print("🚀 Starting Optimized Bihar Cropland NDTI Analysis")
        print("📍 Region: Bihar, India (Single 13-vertex polygon)")
        print("📅 Period: September 2017 - May 2018 (8 months)")
        print("🔧 Mode: ULTRA-AGGRESSIVE Image Capture")
        print("🎯 Goal: Maximum images, improved accuracy, focused histogram")
        print()
        
        # IMPORTANT: Provide path to your CSV file with ground truth data
        # The CSV should have a column named 'NDTI' or 'mean_NDTI' with NDTI values
        csv_path = 'NDTI.csv'  # UPDATE THIS PATH
        
        # If you don't have the CSV yet, set to None
        # csv_path = None
        
        result = main_bihar_analysis(csv_path=csv_path)
        
        if result:
            print("\n" + "="*80)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ Improvements:")
            print("   • Removed quadrant splitting for unified analysis")
            print("   • Extended time period (Sep 2017 - May 2018)")
            print("   • Ultra-permissive thresholds for maximum image capture")
            print("   • Focused histogram binning (60% between 0-0.4)")
            print("   • Ground truth comparison visualization")
            print("   • Comprehensive statistical analysis")
            print()
            print("📊 Results Summary:")
            print(f"   • Total images captured: {result['image_count']}")
            print(f"   • Mean NDTI: {result['mean_ndti']:.6f}")
            print(f"   • Analysis complete for entire Bihar region")
        else:
            print("\n❌ Analysis incomplete - check error messages above")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure Google Earth Engine is properly authenticated and initialized")