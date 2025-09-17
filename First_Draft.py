"""
Southeast Asia Cropland Analysis with Advanced Masking
Combines Sentinel-2 processing with ESA WorldCover land type classification
Focuses on cropland areas with comprehensive cloud and quality masking
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
        # Get the most recent WorldCover image
        worldcover_image = self.worldcover.first()
        
        # Clip to the area of interest
        clipped = worldcover_image.clip(Geometry)
        
        return {'image': clipped}

    def Map_LandType(self, landcover_image):
        """Create simplified land type map focusing on cropland"""
        # Create a mask where only cropland (40) is valid (1), others are invalid (0)
        cropland_mask = landcover_image.eq(40).rename('cropland_mask')
        
        return cropland_mask

def get_aoi(lon, lat, box_width, box_height):
    """Create area of interest geometry"""
    # Convert box dimensions from meters to degrees (approximate)
    # Using more accurate conversion factors for latitude-dependent longitude
    width_deg = box_width / (111320 * np.cos(np.radians(lat)))  # latitude-dependent longitude conversion
    height_deg = box_height / 110540  # meters to degrees latitude
    
    return ee.Geometry.Rectangle([
        lon - width_deg/2, lat - height_deg/2,
        lon + width_deg/2, lat + height_deg/2
    ])

class CroplandProcessor:  # Generalized name instead of SEA_CroplandProcessor
    """Cropland Processing with Advanced Masking - Works for any region"""
    
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
        
        # Initialize Earth Engine (assuming already done)
        self.LT = LandType(EE_initialized=True)
        
        # Get the geometry of the area of interest
        self.AoI_geom = get_aoi(Location[0], Location[1], Box[0], Box[1])
        
        # Get land cover data and create cropland mask
        result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
        self.RegionMap = self.LT.Map_LandType(result['image'])
        
        # Set up date ranges
        if isinstance(Years, list) and len(Years) > 1:
            year0, yearE = int(Years[0]), int(Years[-1])
        else:
            year0 = yearE = int(Years[0]) if isinstance(Years, list) else int(Years)
            
        self.Dates = [f'{year0}-01-01', f'{yearE}-12-31']
        
        # Store parameters
        self.Location = Location
        self.Box = Box
        self.Years = Years
        self.Verbose = Verbose
        self.SentRes = SentRes
        self.ShowPlots = ShowPlots
        self.plot_scale = plot_scale

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.80):
        """
        Process Sentinel-2 data with comprehensive masking
        
        Args:
            QA_BAND (str): Cloud Score+ quality band ('cs_cdf' recommended)
            CLEAR_THRESHOLD (float): Minimum clear sky probability (0-1)
        
        Returns:
            ee.ImageCollection: Processed image collection with valid_pixel_count property
        """
        
        # Support functions for processing
        def mask_clouds_advanced(img):
            """Enhanced cloud masking with shadows, snow, and water removal"""
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

        def set_pixel_count(image):
            """Calculate valid pixel count and add as property"""
            mask = image.select('B4').mask().unmask(0)
            count_dict = mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True)
            count = count_dict.values().get(0)
            return image.set('valid_pixel_count', count)
        
        def apply_landtype_mask(image):
            """Apply cropland mask - only keep cropland areas"""
            # Reproject the landtype mask to match Sentinel-2 resolution for efficiency
            landtype_mask = self.RegionMap.reproject(crs=image.select('B4').projection(), scale=self.SentRes)
            landtype_valid_mask = landtype_mask.eq(1)  # Only cropland areas
            return image.updateMask(landtype_valid_mask)

        # 1) Load Sentinel-2 and Cloud Score+ collections
        print("1.) Loading Sentinel-2, Surface Reflectance, and Cloud Score+ collections...")
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

        # Initial filtering by date and area
        filtered_s2_date_area = (s2
            .filterBounds(self.AoI_geom)
            .filterDate(self.Dates[0], self.Dates[1]))
        
        total_count = filtered_s2_date_area.size().getInfo()
        print(f"Total images in date range: {total_count}")
        
        # Add cloud cover filtering to reduce dataset size
        filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        print(f"Filtering from {self.Dates[0]} to {self.Dates[1]}")
        initial_count = filtered_s2_date_area.size().getInfo()
        print(f"Images after cloud pre-filter (<30% clouds): {initial_count}")
        
        if initial_count == 0:
            print("❌ No images found after initial filtering!")
            return ee.ImageCollection([])

        # 2) Apply cloud masking
        print("2) Filtering pixels blocked by clouds and shadows...")
        filtered_s2 = (filtered_s2_date_area
            .linkCollection(csPlus, [QA_BAND])
            .map(mask_clouds_advanced))

        # 3) Apply land type mask (cropland only)
        print("3) Filtering to cropland areas only...")
        print("   (This step may take a moment for land type classification...)")
        land_masked_collection = filtered_s2.map(apply_landtype_mask)

        # 4) Calculate pixel counts and filter
        print("4) Calculating valid pixel counts and removing low-quality images...")
        land_masked_collection_with_counts = land_masked_collection.map(set_pixel_count)
        
        # Filter to keep only images with sufficient valid pixels
        final_s2 = land_masked_collection_with_counts.filter(
            ee.Filter.gt('valid_pixel_count', 100))
        
        clean_count = final_s2.size().getInfo()
        removed = initial_count - clean_count
        print(f"Removed {removed} images with insufficient valid pixels")
        print(f"Final remaining images: {clean_count}")
        
        if clean_count == 0:
            print("⚠️ Warning: No images remain after all filtering!")
            return ee.ImageCollection([])

        # Return collection WITH counts for best image selection
        return final_s2

def create_rgb_composite(image):
    """Create RGB composite from Sentinel-2 bands (B4, B3, B2)"""
    rgb = image.select(['B4', 'B3', 'B2']).multiply(0.0001)
    return rgb

def create_false_color_composite(image):
    """Create false color composite (NIR, Red, Green) for vegetation analysis"""
    false_color = image.select(['B8', 'B4', 'B3']).multiply(0.0001)
    return false_color

def visualize_image(image, region, vis_params, title="Sentinel-2 Image"):
    """
    Visualize Earth Engine image using matplotlib with direct URL download approach
    
    This function generates a thumbnail URL from Google Earth Engine, downloads the image,
    and displays it using matplotlib. Includes comprehensive error handling for network
    issues and display problems.
    
    Args:
        image: Earth Engine image object to visualize
        region: Earth Engine geometry defining the area to display
        vis_params: Dictionary with visualization parameters (min, max, bands, palette)
        title: String title for the displayed image
    
    Returns:
        bool: True if successful, False if any errors occurred
    """
    try:
        # Start processing with user feedback
        print(f"   ⏳ Processing {title}...")
        
        # Configure thumbnail parameters for Earth Engine image export
        # These parameters control how GEE generates the preview image
        thumbnail_params = {
            'region': region,        # Geographic bounds for the image
            'dimensions': 800,       # Output image width in pixels (height auto-calculated)
            'format': 'png'         # Image format (png for quality, jpg for smaller size)
        }
        # Merge user-provided visualization parameters (bands, min/max values, palette)
        thumbnail_params.update(vis_params)
        
        # Generate the thumbnail URL from Google Earth Engine
        # This creates a temporary URL that GEE uses to serve the processed image
        url = image.getThumbURL(thumbnail_params)
        print(f"   🔗 URL: {url}")
        
        # Attempt to download and display the image
        try:
            print(f"   📥 Downloading image...")
            
            # Make HTTP request to download the image from GEE servers
            # Timeout prevents hanging on slow connections
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raises exception for HTTP error codes (4xx, 5xx)
            
            # Convert the downloaded bytes into a PIL Image object
            # io.BytesIO creates a file-like object from the response content
            img = Image.open(io.BytesIO(response.content))
            print(f"   📏 Image size: {img.size}")
            
            # Create matplotlib figure and display the image
            plt.figure(figsize=(12, 10))    # Set figure size in inches
            plt.imshow(img)                 # Display the PIL image
            plt.title(f"{title}\nImage Size: {img.size}")  # Add title with image dimensions
            plt.axis('off')                 # Remove axis ticks and labels for cleaner look
            plt.tight_layout()              # Adjust spacing to prevent title cutoff
            plt.show()                      # Render the plot
            
            print(f"   ✅ Successfully displayed: {title}")
            return True
            
        except Exception as display_error:
            # Handle download or display errors gracefully
            print(f"   ⚠️ Could not display image: {display_error}")
            print(f"   🌐 Copy this URL to your browser: {url}")
            return False
            
    except Exception as e:
        # Handle any other unexpected errors (URL generation, parameter issues, etc.)
        print(f"   ❌ Error generating {title}: {e}")
        return False

def main_cropland_analysis(location=None, box_size=None, years=None):
    """
    Main processing pipeline for cropland analysis - works for any region
    
    Args:
        location (list, optional): [longitude, latitude] coordinates. 
                                 Defaults to Ho Chi Minh City area [105.8, 10.8]
        box_size (list, optional): [width, height] in meters. 
                                 Defaults to [50000, 50000] (50km x 50km)
        years (list or str, optional): Year(s) for analysis. 
                                     Defaults to ['2024']
    """
    
    # Set default values if not provided
    if location is None:
        location = [105.8, 10.8]  # Default: Ho Chi Minh City area
    
    if box_size is None:
        box_size = [50000, 50000]  # Default: 50km x 50km for optimal processing speed
    
    if years is None:
        years = ['2024']  # Default: Single year for faster processing
    
    # Validate inputs
    if len(location) != 2:
        raise ValueError("Location must be [longitude, latitude]")
    if len(box_size) != 2:
        raise ValueError("Box size must be [width, height] in meters")
    
    # Display analysis parameters
    print("🌾 Cropland Analysis")
    print("=" * 50)
    print(f"📍 Location: {location[1]:.2f}°N, {location[0]:.2f}°E")
    print(f"📐 Area: {box_size[0]/1000:.1f}km x {box_size[1]/1000:.1f}km")
    print(f"📅 Analysis Period: {years}")
    
    # Initialize processor
    processor = CroplandProcessor(
        Location=location,
        Box=box_size,
        Years=years,
        Verbose=True,
        ShowPlots=True
    )
    
    # Process Sentinel-2 data with all masking
    print("\n🛰️ Processing Sentinel-2 data...")
    collection = processor.Pull_Process_Sentinel_data()
    
    if collection.size().getInfo() == 0:
        print("❌ No valid images found for the specified criteria")
        return None
    
    # Get the best image using the valid_pixel_count flag
    print("\n🔍 Selecting best quality image based on valid pixel count...")
    
    # Sort the collection by pixel count in descending order and get the best one
    best_image = collection.sort('valid_pixel_count', False).first()
    
    # Get quality metrics for the selected image
    best_pixel_count = best_image.get('valid_pixel_count').getInfo()
    best_cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    best_scene_id = best_image.get('PRODUCT_ID').getInfo()
    
    print(f"🏆 Selected best image based on highest valid pixel count:")
    print(f"   📊 Valid cropland pixels: {best_pixel_count}")
    print(f"   ☁️ Cloud cover: {best_cloud_cover:.1f}%")
    print(f"   🛰️ Scene ID: {best_scene_id}")
    
    # Create composites
    rgb_image = create_rgb_composite(best_image)
    false_color_image = create_false_color_composite(best_image)
    
    # Visualization parameters
    rgb_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
    false_color_vis = {'min': 0, 'max': 0.3, 'bands': ['B8', 'B4', 'B3']}
    
    # Display images
    print("\n🎨 Generating cropland visualizations...")
    print("💡 If images don't display, copy the URLs to your browser!")
    
    # RGB Composite of cropland areas
    print("\n🌾 RGB Composite (Cropland Areas Only):")
    visualize_image(rgb_image, processor.AoI_geom, rgb_vis, 
                   "Cropland - RGB Composite")
    
    # False Color Composite for vegetation analysis
    print("\n🌱 False Color Composite (Cropland Vegetation):")
    visualize_image(false_color_image, processor.AoI_geom, false_color_vis, 
                   "Cropland - False Color")
    
    # Get image properties
    acquisition_date = best_image.get('system:time_start').getInfo()
    
    print(f"\n📋 Image Information:")
    print(f"🛰️ Scene ID: {best_scene_id}")
    print(f"☁️ Cloud Cover: {best_cloud_cover:.1f}%")
    print(f"📅 Acquisition Date: {datetime.fromtimestamp(acquisition_date/1000)}")
    
    # Show cropland mask
    print("\n🗺️ Cropland Mask:")
    cropland_vis = {'min': 0, 'max': 1, 'palette': ['black', 'yellow']}
    visualize_image(processor.RegionMap, processor.AoI_geom, cropland_vis,
                   "Cropland Areas (Yellow)")
    
    return {
        'processor': processor,
        'collection': collection,
        'rgb_image': rgb_image,
        'false_color_image': false_color_image,
        'best_image': best_image
    }

def test_multiple_regions():
    """Test the analysis with multiple regions to ensure correctness"""
    
    # Test locations
    test_locations = {
        'Vietnam_Mekong': [105.8, 10.8],
        'Thailand_Central': [100.5, 14.0],
        'Indonesia_Java': [110.4, -7.8]
    }
    
    for region_name, coords in test_locations.items():
        print(f"\n" + "="*60)
        print(f"Testing region: {region_name}")
        print("="*60)
        
        try:
            results = main_cropland_analysis(
                location=coords,
                box_size=[30000, 30000],  # Smaller for testing
                years=['2024']
            )
            
            if results is not None:
                print(f"✅ {region_name} analysis completed successfully!")
            else:
                print(f"⚠️ {region_name} analysis returned no results")
                
        except Exception as e:
            print(f"❌ Error in {region_name}: {e}")

if __name__ == "__main__":
    try:
        # Default analysis with Ho Chi Minh City area
        results = main_cropland_analysis()
        
        # Uncomment to test multiple regions
        # test_multiple_regions()
        
        print("\n🎉 Cropland analysis completed successfully!")
        print("📊 The images show only cropland areas with comprehensive masking applied:")
        print("   ✅ Cloud masking (clouds, shadows, cirrus)")
        print("   ✅ Land type masking (cropland areas only)")  
        print("   ✅ Time frame masking (specified date range)")
        print(" First Commit Worked") #Simple testing for github
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        print("Make sure Google Earth Engine is properly authenticated and initialized")