"""
Bihar NDTI Verification with Spatial Heat Map
Creates detailed spatial visualization of NDTI values across Bihar croplands
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
import os
import sys
import time
import rasterio
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Initialize Google Earth Engine
PROJECT_ID = '590577866979'  # Replace with your Google Cloud Project ID

try:
    ee.Initialize(project=PROJECT_ID)
    print("✅ Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing GEE: {e}")
    sys.exit(1)

# ============================================================================
# LAND TYPE CLASS (from your code)
# ============================================================================
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
            10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland',
            40: 'Cropland', 50: 'Built-up', 60: 'Bare/sparse vegetation',
            70: 'Snow and ice', 80: 'Permanent water bodies',
            90: 'Herbaceous wetland', 95: 'Mangroves', 100: 'Moss and lichen'
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

# ============================================================================
# BIHAR GEOMETRY
# ============================================================================
def get_bihar_polygon():
    """Create Bihar polygon geometry based on actual state boundaries"""
    try:
        coordinates = [[
            [83.0, 27.5], [84.0, 27.6], [85.5, 27.4], [87.0, 26.9],
            [87.5, 26.5], [87.6, 26.0], [87.0, 25.2], [86.0, 24.6],
            [84.8, 24.3], [83.3, 24.4], [83.0, 25.0], [82.7, 25.8],
            [83.0, 27.5]
        ]]
        return ee.Geometry.Polygon(coordinates)
    except Exception as e:
        print(f"Error creating Bihar polygon: {e}")
        return None

# ============================================================================
# NDTI CALCULATION
# ============================================================================
def calculate_ndti(image):
    """Calculate Normalized Difference Tillage Index (NDTI)"""
    try:
        b11 = image.select('B11')  # SWIR1
        b12 = image.select('B12')  # SWIR2
        ndti = b11.subtract(b12).divide(b11.add(b12)).rename('NDTI')
        ndti = ndti.clamp(-1, 1)
        return image.addBands(ndti)
    except Exception as e:
        print(f"Error calculating NDTI: {e}")
        return image

# ============================================================================
# DATA EXTRACTION (Modified from your SampleData function)
# ============================================================================
def extract_ndti_map(bihar_geom, start_date='2017-09-01', end_date='2018-05-01', 
                     scale=100, scale_factor=1000):
    """
    Extract NDTI map for Bihar region using GEE export method
    Returns: data_array, lons, lats, bounds
    """
    
    print("\n🔄 Extracting NDTI data from Google Earth Engine...")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Scale: {scale}m per pixel")
    
    # Initialize Land Type
    LT = LandType(EE_initialized=True)
    
    # Get cropland mask
    result = LT.get_land_cover_for_region(Geometry=bihar_geom)
    if result is None:
        raise ValueError("Failed to get land cover data")
    
    cropland_mask = LT.Map_LandType(result['image'])
    if cropland_mask is None:
        raise ValueError("Failed to create cropland mask")
    
    # Get Sentinel-2 data
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    
    # Filter and mask clouds
    def mask_clouds(img):
        cs = img.select('cs_cdf')
        cloud_mask = cs.gte(0.40)
        scl = img.select('SCL')
        saturated_mask = scl.neq(1)
        combined_mask = cloud_mask.And(saturated_mask)
        return img.updateMask(combined_mask)
    
    def apply_landtype_mask(image):
        landtype_mask = cropland_mask.reproject(
            crs=image.select('B4').projection(), scale=scale)
        landtype_valid_mask = landtype_mask.eq(1)
        return image.updateMask(landtype_valid_mask)
    
    # Process collection
    filtered_s2 = (s2
        .filterBounds(bihar_geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 95))
        .linkCollection(csPlus, ['cs_cdf'])
        .map(mask_clouds)
        .map(apply_landtype_mask)
        .map(calculate_ndti))
    
    image_count = filtered_s2.size().getInfo()
    print(f"   ✅ Found {image_count} usable images")
    
    if image_count == 0:
        raise ValueError("No images found for processing")
    
    # Create mean NDTI image
    ndti_mean = filtered_s2.select('NDTI').mean()
    
    # Get bounds
    bounds_coords = bihar_geom.bounds().coordinates().get(0).getInfo()
    min_lon = min([coord[0] for coord in bounds_coords])
    max_lon = max([coord[0] for coord in bounds_coords])
    min_lat = min([coord[1] for coord in bounds_coords])
    max_lat = max([coord[1] for coord in bounds_coords])
    
    # Get original value range
    print("   📊 Getting NDTI value range...")
    stats = ndti_mean.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=bihar_geom,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True).getInfo()
    
    original_min = stats.get('NDTI_min', -1)
    original_max = stats.get('NDTI_max', 1)
    print(f"   📈 NDTI range: {original_min:.6f} to {original_max:.6f}")
    
    # Scale for export
    range_size = original_max - original_min
    if range_size == 0:
        mid = int(scale_factor/2)
        scaled_image = ee.Image.constant(mid).rename('NDTI').toInt16()
    else:
        scaled_image = (ndti_mean
                      .subtract(original_min)
                      .divide(range_size)
                      .multiply(scale_factor)
                      .toInt16())
    
    # Export to Google Drive
    GoogleDriveFolder = 'GEE_exports'
    TIFF_file_name = f"Bihar_NDTI_mean_{start_date.replace('-','')}_{end_date.replace('-','')}"
    
    print(f"\n📤 EXPORTING TO GOOGLE DRIVE:")
    print(f"   Folder: {GoogleDriveFolder}")
    print(f"   File: {TIFF_file_name}_scaled_at_{scale_factor}")
    
    task = ee.batch.Export.image.toDrive(
        image=scaled_image,
        description=f"Bihar_NDTI_scaled_{scale_factor}",
        folder=GoogleDriveFolder,
        fileNamePrefix=f"{TIFF_file_name}_scaled_at_{scale_factor}",
        region=bihar_geom,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF')
    
    task.start()
    
    # Monitor task
    start_time = time.time()
    while task.active():
        elapsed = time.time() - start_time
        print(f"   ⏳ Exporting... ({int(elapsed)}s) - Status: {task.status()['state']}")
        time.sleep(30)
    
    status = task.status()
    if status['state'] != 'COMPLETED':
        raise Exception(f"Export failed: {status}")
    
    runtime = time.time() - start_time
    print(f"   ✅ Export finished after {int(runtime)}s")
    
    # Download from Google Drive
    print("\n📥 Downloading from Google Drive...")
    credential_file = 'creds.json'
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile('client_secrets.json')
    
    if os.path.exists(credential_file):
        gauth.LoadCredentialsFile(credential_file)
    
    if gauth.access_token_expired:
        print("   🔐 Starting authentication...")
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile(credential_file)
    
    drive = GoogleDrive(gauth)
    
    # Find file
    search_name = f"{TIFF_file_name}_scaled_at_{scale_factor}"
    file_list = drive.ListFile({
        'q': f"title contains '{search_name}' and trashed=false"
    }).GetList()
    
    if not file_list:
        raise Exception("Exported file not found in Drive")
    
    file_drive = file_list[0]
    
    # Download
    TIFF_save_path = os.path.join(os.getcwd(), GoogleDriveFolder)
    os.makedirs(TIFF_save_path, exist_ok=True)
    scaled_filepath = os.path.join(TIFF_save_path, f"{search_name}.tif")
    
    file_drive.GetContentFile(scaled_filepath)
    print(f"   ✅ Downloaded to: {scaled_filepath}")
    
    # Clean up Drive if multiple files
    if len(file_list) > 1:
        file_drive.Delete()
        print("   🗑️ Cleaned up Drive")
    
    # Read and rescale
    print("\n📖 Reading and rescaling data...")
    with rasterio.open(scaled_filepath) as src:
        scaled_data = src.read(1)
        transform = src.transform
        width, height = src.width, src.height
        
        # Compute coordinates
        lons = np.linspace(
            transform[2], transform[2] + transform[0] * width, width)
        lats = np.linspace(
            transform[5], transform[5] + transform[4] * height, height)
    
    # Rescale to original range
    scaled_data = scaled_data.astype(np.float64)
    
    if range_size == 0:
        data_array = np.full_like(scaled_data, original_min)
    else:
        data_array = (scaled_data / scale_factor) * range_size + original_min
    
    print(f"   ✅ Rescaled: {np.nanmin(data_array):.6f} to {np.nanmax(data_array):.6f}")
    
    # Clean up temporary file
    try:
        os.remove(scaled_filepath)
        print("   🧹 Cleaned up temporary files")
    except:
        pass
    
    return data_array, lons, lats, (min_lon, max_lon, min_lat, max_lat), image_count

# ============================================================================
# CUSTOM COLORMAP (from your code)
# ============================================================================
def create_ndti_colormap():
    """
    Create custom colormap optimized for NDTI values (0 to 0.4 range)
    """
    # Define color scheme for NDTI (focused on 0-0.4 where most data lies)
    color_codes = [
        '#8B0000',  # Dark red (negative/very low)
        '#FF0000',  # Red (low)
        '#FF6347',  # Tomato (low-moderate)
        '#FFA500',  # Orange
        '#FFD700',  # Gold
        '#FFFF00',  # Yellow
        '#ADFF2F',  # Yellow-green
        '#90EE90',  # Light green
        '#32CD32',  # Lime green
        '#228B22',  # Forest green
        '#006400'   # Dark green (high tillage)
    ]
    
    # Boundaries optimized for typical NDTI range (0 to 0.4)
    color_bnds = [-1.0, -0.2, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0]
    
    # Create labels
    tick_labels = [f"{color_bnds[i]:.2f}-{color_bnds[i+1]:.2f}" 
                   for i in range(len(color_bnds) - 1)]
    tick_labels[-1] = f"{color_bnds[-2]:.2f}+"
    
    # Calculate tick positions (midpoints)
    tick_locals = [(color_bnds[i] + color_bnds[i+1]) / 2 
                   for i in range(len(color_bnds) - 1)]
    
    # Create colormap and normalization
    cmap = ListedColormap(color_codes)
    norm = BoundaryNorm(color_bnds, len(color_codes))
    
    return cmap, norm, tick_locals, tick_labels, color_bnds

# ============================================================================
# PLOTTING FUNCTION (Modified from your code)
# ============================================================================
def plot_ndti_heat_map(data_array, lons, lats, coord_bnds, image_count,
                       start_date, end_date, save_path='bihar_ndti_verification.png'):
    """
    Create comprehensive heat map visualization with all backgrounds
    """
    
    # Calculate statistics
    data_min = np.nanmin(data_array)
    data_max = np.nanmax(data_array)
    data_mean = np.nanmean(data_array)
    data_std = np.nanstd(data_array)
    valid_pixels = np.sum(~np.isnan(data_array))
    
    print(f"\n📊 NDTI Statistics:")
    print(f"   Min:  {data_min:.6f}")
    print(f"   Max:  {data_max:.6f}")
    print(f"   Mean: {data_mean:.6f}")
    print(f"   Std:  {data_std:.6f}")
    print(f"   Valid pixels: {valid_pixels:,}")
    
    # Create custom colormap
    custom_cmap, custom_norm, ticks, tick_labels, color_bnds = create_ndti_colormap()
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(24, 20))
    backgrounds = ['None', 'Satellite', 'Cartopy', 'Both']
    
    for idx, bg in enumerate(backgrounds, 1):
        ax = plt.subplot(2, 2, idx, projection=ccrs.PlateCarree())
        
        # Add background based on type
        bg_lower = bg.lower()
        if bg_lower == 'satellite' or bg_lower == 'both':
            from cartopy.io.img_tiles import GoogleTiles
            imagery = GoogleTiles(style='satellite')
            ax.add_image(imagery, 10)
            ax.set_extent(list(coord_bnds), crs=ccrs.PlateCarree())
        
        if bg_lower == 'cartopy' or bg_lower == 'both':
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.RIVERS, linewidth=0.3)
            ax.add_feature(cfeature.LAKES, alpha=0.7, facecolor='lightblue')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        # Create meshgrid and plot
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        im = ax.pcolormesh(lon_mesh, lat_mesh, data_array,
                          cmap=custom_cmap, norm=custom_norm,
                          alpha=0.7, transform=ccrs.PlateCarree())
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.05)
        cbar.set_label('NDTI Value', rotation=270, labelpad=20, fontsize=10)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels, fontsize=8)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Title for each subplot
        ax.set_title(f'Background: {bg}', fontsize=12, fontweight='bold', pad=10)
        
        # Add statistics text box
        stats_text = (f'Images: {image_count}\n'
                     f'Min: {data_min:.4f}\n'
                     f'Max: {data_max:.4f}\n'
                     f'Mean: {data_mean:.4f}\n'
                     f'Std: {data_std:.4f}\n'
                     f'Pixels: {valid_pixels:,}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='black', linewidth=1.5))
    
    # Overall title
    fig.suptitle(f'Bihar Region NDTI Verification Map (Cropland Only)\n'
                f'Period: {start_date} to {end_date} | Resolution: 100m',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Map saved as: {save_path}")
    plt.show()
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    
    print("="*80)
    print("🌾 BIHAR NDTI VERIFICATION MAP GENERATOR")
    print("="*80)
    print("📍 Region: Bihar, India (13-vertex polygon)")
    print("📊 Index: NDTI = (B11-B12)/(B11+B12)")
    print("🎯 Purpose: Verify NDTI calculations with spatial visualization")
    print("="*80)
    
    try:
        # Configuration
        start_date = '2017-09-01'
        end_date = '2018-05-01'
        scale = 100  # 100m resolution
        
        # Get Bihar polygon
        print("\n🗺️  Creating Bihar polygon...")
        bihar_geom = get_bihar_polygon()
        if bihar_geom is None:
            raise ValueError("Failed to create Bihar geometry")
        print("   ✅ Bihar polygon created (13 vertices)")
        
        # Extract NDTI data
        data_array, lons, lats, bounds, image_count = extract_ndti_map(
            bihar_geom, start_date, end_date, scale)
        
        # Create visualization
        print("\n🎨 Creating verification maps...")
        plot_ndti_heat_map(data_array, lons, lats, bounds, image_count,
                          start_date, end_date)
        
        print("\n" + "="*80)
        print("🎉 VERIFICATION MAP GENERATION COMPLETE!")
        print("="*80)
        print("✅ Map shows NDTI distribution across Bihar croplands")
        print("✅ Four background options displayed for comprehensive view")
        print("✅ Statistics box shows data quality metrics")
        print("✅ Color scale optimized for typical NDTI range (0-0.4)")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()