"""
Cropland Phenology Clustering Analysis for Bihar, India
Uses temporal NDVI patterns to classify land cover types
Period: 2019 (full year for complete seasonal cycle)
Method: Peak timing, min timing, and mean timing clustering
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

def calculate_ndvi(image):
    """
    Calculate Normalized Difference Vegetation Index (NDVI)
    NDVI = (NIR - Red) / (NIR + Red)
    where NIR = B8 (842 nm) and Red = B4 (665 nm)
    """
    try:
        nir = image.select('B8')   # Near Infrared
        red = image.select('B4')   # Red
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        ndvi = ndvi.clamp(-1, 1)
        return image.addBands(ndvi)
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        return image

def add_modified_julian_date(image):
    """
    Add Modified Julian Date (MJD) as an image property
    MJD = Julian Date - 2400000.5
    """
    try:
        # Get image date
        date = ee.Date(image.get('system:time_start'))
        
        # Convert to Modified Julian Date
        # Unix timestamp is milliseconds since 1970-01-01
        # Julian Date for Unix epoch (1970-01-01 00:00:00) is 2440587.5
        unix_millis = date.millis()
        days_since_epoch = unix_millis.divide(86400000)  # ms to days
        julian_date = days_since_epoch.add(2440587.5)
        modified_julian_date = julian_date.subtract(2400000.5)
        
        return image.set('MJD', modified_julian_date)
    except Exception as e:
        print(f"Error adding MJD: {e}")
        return image

class PhenologyProcessor:
    """Phenology Processing with Temporal Clustering"""
    
    def __init__(self, geometry, start_date, end_date, Verbose=True, SentRes=10):
        """Initialize the phenology processor with polygon geometry"""
        
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
            
            self.phenology_results = {}
            
        except Exception as e:
            print(f"Error initializing PhenologyProcessor: {e}")
            raise

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.65):
        """
        Process Sentinel-2 data with STRICT masking to prevent computational overload
        OPTIMIZED: Aggressive filtering for manageable dataset size
        """
        
        try:
            def mask_clouds_strict(img):
                """
                STRICT cloud masking - only keep high quality pixels
                """
                try:
                    # STRICT cloud score threshold (was 0.40, now 0.65)
                    cs = img.select(QA_BAND)
                    cloud_mask = cs.gte(CLEAR_THRESHOLD)
                    
                    scl = img.select('SCL')
                    
                    # Mask out more pixel types for cleaner data
                    # Keep only: 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclassified)
                    # Remove: 1 (saturated), 2 (dark), 3 (cloud shadow), 8 (cloud med), 9 (cloud high), 10 (cirrus), 11 (snow)
                    scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
                    
                    combined_mask = cloud_mask.And(scl_mask)
                    
                    return img.updateMask(combined_mask)
                except Exception as e:
                    print(f"Error in cloud masking: {e}")
                    return img
            
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
                print(f"    Using STRICT masking to reduce computational load")
            
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            filtered_s2_date_area = (s2
                .filterBounds(self.AoI_geom)
                .filterDate(self.start_date, self.end_date))
            
            # Don't call getInfo() here - too expensive with 3000+ images
            if self.Verbose:
                print(f"    Filtering images...")
            
            # STRICT cloud filtering - only allow <60% cloud cover (was 95%)
            filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
            
            if self.Verbose:
                print(f"    Applying STRICT cloud masking and processing...")

            # Link with cloud score and apply masking
            filtered_s2 = (filtered_s2_date_area
                .linkCollection(csPlus, [QA_BAND])
                .map(mask_clouds_strict))

            # Apply land type mask
            land_masked_collection = filtered_s2.map(apply_landtype_mask)
            
            # Calculate NDVI
            ndvi_collection = land_masked_collection.map(calculate_ndvi)
            
            # Add MJD
            final_collection = ndvi_collection.map(add_modified_julian_date)
            
            if self.Verbose:
                print(f"    ✅ Collection ready for temporal compositing")
            
            return final_collection
            
        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return ee.ImageCollection([])

    def create_temporal_composites(self, collection, composite_days=30):
        """
        Average collection based on 30-day ranges (increased from 15 to reduce data volume)
        OPTIMIZED: Uses server-side operations only
        """
        try:
            if self.Verbose:
                print(f"\n  Creating {composite_days}-day temporal composites...")
            
            # Get the date range from the collection
            # Use a simple approach: iterate through the year in fixed intervals
            start_date = ee.Date(self.start_date)
            end_date = ee.Date(self.end_date)
            
            # Calculate number of days
            n_days = end_date.difference(start_date, 'day')
            
            # Number of composites
            n_composites = n_days.divide(composite_days).ceil()
            
            # Create list of composite periods
            composite_indices = ee.List.sequence(0, n_composites.subtract(1))
            
            def create_composite(i):
                """Create a single composite for a time period"""
                i = ee.Number(i)
                composite_start = start_date.advance(i.multiply(composite_days), 'day')
                composite_end = composite_start.advance(composite_days, 'day')
                
                # Filter collection for this time period
                period_collection = collection.filterDate(composite_start, composite_end)
                
                # Calculate mean NDVI
                mean_ndvi = period_collection.select('NDVI').mean()
                
                # Calculate mean MJD for this period
                mean_mjd = period_collection.aggregate_mean('MJD')
                
                # Return the composite with MJD as property
                return mean_ndvi.set({
                    'MJD': mean_mjd,
                    'system:time_start': composite_start.millis(),
                    'composite_start': composite_start.format('YYYY-MM-dd'),
                    'composite_end': composite_end.format('YYYY-MM-dd')
                })
            
            # Create composites
            composites = ee.ImageCollection(composite_indices.map(create_composite))
            
            # Filter out empty composites (where no images existed)
            composites = composites.filter(ee.Filter.notNull(['MJD']))
            
            if self.Verbose:
                # Only check size at the end
                n_comp = composites.size().getInfo()
                print(f"    ✅ Created {n_comp} temporal composites")
            
            return composites
            
        except Exception as e:
            print(f"Error creating temporal composites: {e}")
            import traceback
            traceback.print_exc()
            return ee.ImageCollection([])

    def calculate_phenology_metrics(self, collection):
        """
        Calculate peak_time, min_time, and mean_time for each pixel
        Returns an image with these as bands
        OPTIMIZED: Server-side only operations
        """
        try:
            if self.Verbose:
                print(f"\n  Calculating phenology metrics...")
            
            # For each image, add MJD as a band (from property)
            def add_mjd_band(image):
                mjd = ee.Number(image.get('MJD'))
                # Cast to float to ensure homogeneous collection
                mjd_band = ee.Image.constant(mjd).toFloat().rename('MJD')
                return image.addBands(mjd_band)
            
            collection_with_mjd_band = collection.map(add_mjd_band)
            
            # Find peak NDVI time for each pixel
            # qualityMosaic selects the pixel value from the image with highest NDVI
            peak_time_image = collection_with_mjd_band.qualityMosaic('NDVI').select('MJD').rename('peak_time')
            
            # Find min NDVI time for each pixel
            # Invert NDVI so qualityMosaic finds minimum
            def invert_ndvi(image):
                inverted = image.select('NDVI').multiply(-1).rename('NDVI_inverted')
                return image.addBands(inverted)
            
            collection_inverted = collection_with_mjd_band.map(invert_ndvi)
            min_time_image = collection_inverted.qualityMosaic('NDVI_inverted').select('MJD').rename('min_time')
            
            # Calculate mean time (simple average of all MJD bands)
            mean_time_image = collection_with_mjd_band.select('MJD').mean().rename('mean_time')
            
            # Also get mean NDVI for reference
            mean_ndvi = collection.select('NDVI').mean().rename('mean_NDVI')
            
            # Combine all metrics into single image
            phenology_image = peak_time_image.addBands(min_time_image).addBands(mean_time_image).addBands(mean_ndvi)
            
            if self.Verbose:
                print(f"    ✅ Calculated phenology metrics (peak_time, min_time, mean_time)")
            
            return phenology_image
            
        except Exception as e:
            print(f"Error calculating phenology metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_clusters(self, phenology_image, n_bins=15, cluster_type='peak_time'):
        """
        Assign pixels to bins based on phenology metrics
        
        Args:
            phenology_image: Image with peak_time, min_time, mean_time bands
            n_bins: Number of bins (10-20 recommended)
            cluster_type: Which metric to use ('peak_time', 'min_time', or 'mean_time')
        """
        try:
            if self.Verbose:
                print(f"\n  Creating {n_bins} clusters based on {cluster_type}...")
            
            if phenology_image is None:
                return None
            
            # Select the clustering band
            metric_band = phenology_image.select(cluster_type)
            
            # Get min and max values for the metric
            # Use SMALLER scale and more aggressive settings to prevent timeout
            stats = metric_band.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=self.AoI_geom,
                scale=100,  # Increased from 10 to 100 for faster computation
                maxPixels=1e9,  # Reduced from 1e10
                bestEffort=True,
                tileScale=8  # Increased from 4 to 8
            ).getInfo()
            
            min_val = stats[f'{cluster_type}_min']
            max_val = stats[f'{cluster_type}_max']
            
            if self.Verbose:
                print(f"    {cluster_type} range: {min_val:.2f} to {max_val:.2f} (Modified Julian Date)")
            
            # Create bins
            bin_width = (max_val - min_val) / n_bins
            
            # Assign each pixel to a bin
            # bin_id = floor((value - min) / bin_width)
            cluster_image = metric_band.subtract(min_val).divide(bin_width).floor().rename('cluster')
            
            # Clamp to valid range [0, n_bins-1]
            cluster_image = cluster_image.clamp(0, n_bins - 1)
            
            # Add cluster as band to phenology image
            result_image = phenology_image.addBands(cluster_image)
            
            # Calculate cluster statistics
            if self.Verbose:
                print(f"    Calculating cluster distribution...")
            
            # Use a histogram reducer to get cluster distribution efficiently
            histogram = cluster_image.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=self.AoI_geom,
                scale=100,  # Increased from 10 to 100
                maxPixels=1e9,  # Reduced from 1e10
                bestEffort=True,
                tileScale=8  # Increased from 4 to 8
            ).getInfo()
            
            cluster_stats = histogram.get('cluster', {})
            
            # Convert string keys to integers (handle both '0' and '0.0' formats)
            cluster_stats = {int(float(k)): v for k, v in cluster_stats.items()}
            
            if self.Verbose:
                print(f"    ✅ Created {n_bins} clusters")
                print(f"    Cluster distribution (top 5):")
                sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                for cluster_id, count in sorted_clusters:
                    print(f"      Cluster {cluster_id}: {int(count):,} pixels")
            
            return {
                'image': result_image,
                'n_bins': n_bins,
                'cluster_type': cluster_type,
                'min_val': min_val,
                'max_val': max_val,
                'bin_width': bin_width,
                'cluster_stats': cluster_stats
            }
            
        except Exception as e:
            print(f"Error creating clusters: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process(self, composite_days=30, n_bins=15, cluster_type='peak_time'):
        """
        Full processing pipeline:
        1. Get Sentinel-2 data with cloud masking
        2. Calculate NDVI
        3. Create temporal composites (30-day average for reduced data volume)
        4. Add Modified Julian Date
        5. Calculate peak_time, min_time, mean_time
        6. Assign pixels to bins
        """
        try:
            print("\n" + "="*80)
            print("PHENOLOGY PROCESSING PIPELINE")
            print("="*80)
            
            # Step 1-2: Pull and process data (includes NDVI calculation)
            print("\nSTEP 1-2: Loading and processing Sentinel-2 imagery...")
            collection = self.Pull_Process_Sentinel_data()
            
            # Step 3: Create temporal composites
            print(f"\nSTEP 3: Creating {composite_days}-day temporal composites...")
            composites = self.create_temporal_composites(collection, composite_days)
            
            n_composites = composites.size().getInfo()
            if n_composites == 0:
                print("❌ No composites created")
                return None
            
            print(f"    ✅ {n_composites} composites ready")
            
            # Step 4-5: Calculate phenology metrics (MJD already added in composite step)
            print(f"\nSTEP 4-5: Calculating phenology metrics...")
            phenology_image = self.calculate_phenology_metrics(composites)
            
            if phenology_image is None:
                print("❌ Failed to calculate phenology metrics")
                return None
            
            # Step 6: Create clusters
            print(f"\nSTEP 6: Creating {n_bins} clusters based on {cluster_type}...")
            cluster_result = self.create_clusters(phenology_image, n_bins, cluster_type)
            
            if cluster_result is None:
                print("❌ Failed to create clusters")
                return None
            
            self.phenology_results = cluster_result
            
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            
            return cluster_result
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

def visualize_results(result, start_date, end_date):
    """
    Create visualization of clustering results
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Bihar Phenology Clustering Analysis\n{start_date} to {end_date}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Cluster distribution
        ax1 = axes[0, 0]
        cluster_stats = result['cluster_stats']
        bins = sorted(cluster_stats.keys())
        counts = [cluster_stats[i] for i in bins]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))
        bars = ax1.bar(bins, counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Cluster ID', fontweight='bold')
        ax1.set_ylabel('Pixel Count', fontweight='bold')
        ax1.set_title(f'Cluster Distribution ({result["n_bins"]} bins)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{int(count)}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Cluster statistics
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        info_text = f"""
CLUSTERING PARAMETERS
{'='*40}

Clustering Metric:    {result['cluster_type']}
Number of Bins:       {result['n_bins']}
Value Range:          {result['min_val']:.2f} to {result['max_val']:.2f} MJD
Bin Width:            {result['bin_width']:.2f} days

Total Pixels:         {sum(counts):,}

INTERPRETATION
{'='*40}

Modified Julian Date (MJD):
- Days since Nov 17, 1858
- Single number representing date

Peak Time: When vegetation reaches
           maximum greenness

Min Time:  When vegetation is least
           green (dormant season)

Mean Time: Average timing of greenness
        """
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Plot 3: Temporal range visualization
        ax3 = axes[1, 0]
        
        # Convert MJD back to approximate dates for visualization
        # MJD 0 = Nov 17, 1858
        # Approximate conversion for 2019
        mjd_2019_start = 58484  # Jan 1, 2019
        
        bin_centers = [result['min_val'] + (i + 0.5) * result['bin_width'] for i in bins]
        days_from_jan1 = [(mjd - mjd_2019_start) for mjd in bin_centers]
        
        ax3.plot(bins, days_from_jan1, marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax3.set_xlabel('Cluster ID', fontweight='bold')
        ax3.set_ylabel('Days from Jan 1, 2019', fontweight='bold')
        ax3.set_title(f'{result["cluster_type"].replace("_", " ").title()} Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary of top 5 clusters
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary_text = f"""
TOP 5 CLUSTERS BY SIZE
{'='*40}

"""
        for rank, (cluster_id, count) in enumerate(sorted_clusters, 1):
            mjd = result['min_val'] + (cluster_id + 0.5) * result['bin_width']
            days_from_jan = mjd - mjd_2019_start
            pct = (count / sum(counts)) * 100
            summary_text += f"{rank}. Cluster {cluster_id}:\n"
            summary_text += f"   Pixels: {int(count):,} ({pct:.1f}%)\n"
            summary_text += f"   Day: ~{int(days_from_jan)} (from Jan 1)\n\n"
        
        summary_text += f"""
{'='*40}
LAND COVER INTERPRETATION

Different clusters represent distinct
vegetation phenology patterns:

- Early peaks: Winter/spring crops
- Mid peaks: Summer crops
- Late peaks: Fall crops
- Multiple peaks: Mixed agriculture
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('bihar_phenology_clustering.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'bihar_phenology_clustering.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def main_bihar_phenology():
    """
    Main processing pipeline for Bihar phenology clustering
    """
    
    # Full year 2019 for complete seasonal cycle
    start_date = '2019-01-01'
    end_date = '2019-12-31'
    
    print("🌾 Bihar Phenology Clustering Analysis")
    print("=" * 80)
    print(f"📍 Region: Bihar, India (Single 13-vertex polygon)")
    print(f"📅 Period: {start_date} to {end_date} (Full year)")
    print(f"🔬 Method: Temporal NDVI clustering")
    print()
    print("⚡ STRICT FILTERING TO PREVENT TIMEOUT:")
    print("   • Cloud threshold: <60% (was 95%)")
    print("   • Clear sky threshold: 0.65 (was 0.40)")
    print("   • SCL masking: Only vegetation, bare soil, water")
    print("   • Composite interval: 30 days (was 15)")
    print("   • Processing scale: 100m (was 10m)")
    print()
    print("📊 ANALYSIS STEPS:")
    print("   1. Process Sentinel-2 imagery (STRICT cloud masking, cropland filtering)")
    print("   2. Calculate NDVI for each image")
    print("   3. Create 30-day temporal composites")
    print("   4. Add Modified Julian Date to each composite")
    print("   5. Calculate peak_time, min_time, mean_time for each pixel")
    print("   6. Assign pixels to 10-20 bins based on timing metrics")
    print()
    
    try:
        # Get Bihar polygon
        bihar_polygon = get_bihar_polygon()
        
        if bihar_polygon is None:
            raise ValueError("Failed to create Bihar polygon")
        
        print("🗺️ Bihar Polygon Created (13 vertices)")
        print()
        
        # Process Bihar region
        print("🔄 Starting phenology processing...")
        processor = PhenologyProcessor(
            geometry=bihar_polygon,
            start_date=start_date,
            end_date=end_date,
            Verbose=True
        )
        
        # Run processing with parameters
        result = processor.process(
            composite_days=30,      # 30-day composites (reduced from 15)
            n_bins=15,              # 15 clusters
            cluster_type='peak_time'  # Cluster by peak timing
        )
        
        if result:
            print("\n" + "="*80)
            print("✅ PHENOLOGY CLUSTERING COMPLETED!")
            print("="*80)
            print(f"Clustering Metric:  {result['cluster_type']}")
            print(f"Number of Bins:     {result['n_bins']}")
            print(f"Value Range:        {result['min_val']:.2f} to {result['max_val']:.2f} MJD")
            print(f"Total Pixels:       {sum(result['cluster_stats'].values()):,}")
            print()
            
            # Create visualization
            visualize_results(result, start_date, end_date)
            
            return result
        else:
            print("❌ Failed to complete phenology analysis")
            return None
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        print("🚀 Starting Bihar Phenology Clustering Analysis")
        print("📍 Region: Bihar, India")
        print("📅 Period: 2019 (Full year)")
        print("🎯 Goal: Classify land cover by vegetation timing patterns")
        print()
        
        result = main_bihar_phenology()
        
        if result:
            print("\n" + "="*80)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ Key Features:")
            print("   • Temporal NDVI composites created")
            print("   • Phenology metrics calculated (peak, min, mean timing)")
            print("   • Pixels clustered into distinct phenology groups")
            print("   • Results visualized and saved")
            print()
            print("📊 Output: bihar_phenology_clustering.png")
        else:
            print("\n❌ Analysis incomplete - check error messages above")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()