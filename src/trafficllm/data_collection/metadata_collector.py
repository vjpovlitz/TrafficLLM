"""
Metadata Collection System for Traffic Camera Data
"""
import requests
import json
import datetime
import pytz
from typing import Dict, Optional, Tuple
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class WeatherData:
    """Weather information structure"""
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    wind_direction: float
    visibility: float
    conditions: str
    timestamp: str

@dataclass
class TimeData:
    """Time-related metadata"""
    timestamp: str
    utc_timestamp: str
    local_time: str
    timezone: str
    day_of_week: str
    hour: int
    is_daytime: bool
    is_rush_hour: bool
    is_weekend: bool

@dataclass
class CameraMetadata:
    """Complete metadata for a camera capture"""
    camera_info: Dict
    time_data: TimeData
    weather_data: Optional[WeatherData]
    image_info: Dict
    quality_metrics: Dict
    location_data: Dict

class MetadataCollector:
    """Collect comprehensive metadata for traffic camera captures"""
    
    def __init__(self, weather_api_key: str = None):
        self.weather_api_key = weather_api_key
        self.logger = logging.getLogger(__name__)
        
        # Load API key from environment if not provided
        if not self.weather_api_key:
            self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
        # Rush hour definitions (can be customized per location)
        self.rush_hours = {
            'morning': (7, 9),   # 7:00 AM - 9:00 AM
            'evening': (17, 19)  # 5:00 PM - 7:00 PM
        }
    
    def collect_time_metadata(self, timezone_str: str = 'America/New_York') -> TimeData:
        """Collect comprehensive time-related metadata"""
        now_utc = datetime.datetime.now(pytz.UTC)
        timezone = pytz.timezone(timezone_str)
        local_time = now_utc.astimezone(timezone)
        
        # Determine if it's daytime (rough estimate: 6 AM - 8 PM)
        is_daytime = 6 <= local_time.hour <= 20
        
        # Check if it's rush hour
        is_rush_hour = (
            (self.rush_hours['morning'][0] <= local_time.hour < self.rush_hours['morning'][1]) or
            (self.rush_hours['evening'][0] <= local_time.hour < self.rush_hours['evening'][1])
        )
        
        # Check if it's weekend
        is_weekend = local_time.weekday() >= 5  # Saturday = 5, Sunday = 6
        
        return TimeData(
            timestamp=local_time.isoformat(),
            utc_timestamp=now_utc.isoformat(),
            local_time=local_time.strftime('%Y-%m-%d %H:%M:%S'),
            timezone=timezone_str,
            day_of_week=local_time.strftime('%A'),
            hour=local_time.hour,
            is_daytime=is_daytime,
            is_rush_hour=is_rush_hour,
            is_weekend=is_weekend
        )
    
    def collect_weather_metadata(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Collect weather data from OpenWeatherMap API"""
        if not self.weather_api_key:
            self.logger.warning("No weather API key provided, skipping weather data")
            return None
        
        try:
            # OpenWeatherMap API endpoint
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'units': 'imperial'  # Fahrenheit, mph
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract weather information
            main = data.get('main', {})
            weather = data.get('weather', [{}])[0]
            wind = data.get('wind', {})
            
            return WeatherData(
                temperature=main.get('temp', 0.0),
                humidity=main.get('humidity', 0.0),
                precipitation=data.get('rain', {}).get('1h', 0.0),  # Rain in last hour
                wind_speed=wind.get('speed', 0.0),
                wind_direction=wind.get('deg', 0.0),
                visibility=data.get('visibility', 10000) / 1000.0,  # Convert to km
                conditions=weather.get('description', 'unknown'),
                timestamp=datetime.datetime.now(pytz.UTC).isoformat()
            )
            
        except requests.RequestException as e:
            self.logger.error(f"Weather API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing weather data: {e}")
            return None
    
    def collect_image_metadata(self, image_path: str, validation_result: Dict = None) -> Dict:
        """Collect image-specific metadata"""
        image_metadata = {
            "file_path": image_path,
            "file_name": os.path.basename(image_path),
            "file_size": 0,
            "creation_time": None,
            "resolution": None,
            "format": None
        }
        
        try:
            if os.path.exists(image_path):
                # File system metadata
                stat = os.stat(image_path)
                image_metadata["file_size"] = stat.st_size
                image_metadata["creation_time"] = datetime.datetime.fromtimestamp(
                    stat.st_ctime, tz=pytz.UTC
                ).isoformat()
                
                # Image format information
                from PIL import Image
                with Image.open(image_path) as img:
                    image_metadata["resolution"] = img.size
                    image_metadata["format"] = img.format
                    image_metadata["mode"] = img.mode
                
                # Include validation results if provided
                if validation_result:
                    image_metadata["validation"] = validation_result
                    
        except Exception as e:
            self.logger.error(f"Error collecting image metadata for {image_path}: {e}")
        
        return image_metadata
    
    def get_location_coordinates(self, location_name: str) -> Tuple[float, float]:
        """Get coordinates for a location (simplified - could use geocoding API)"""
        # Predefined coordinates for known camera locations
        known_locations = {
            "Sandy Point": (38.9726, -76.3969),
            "Exit 23 MD2": (38.9500, -76.5500),
            "Bay Bridge": (38.9726, -76.3969)
        }
        
        return known_locations.get(location_name, (39.0458, -76.6413))  # Default to Baltimore
    
    def collect_complete_metadata(self, camera_config: Dict, image_path: str, 
                                validation_result: Dict = None, 
                                quality_metrics: Dict = None) -> CameraMetadata:
        """Collect all metadata for a camera capture"""
        
        # Get coordinates for weather data
        lat, lon = self.get_location_coordinates(camera_config.get('location', ''))
        
        # Collect all metadata components
        time_data = self.collect_time_metadata()
        weather_data = self.collect_weather_metadata(lat, lon)
        image_info = self.collect_image_metadata(image_path, validation_result)
        
        # Location data
        location_data = {
            "latitude": lat,
            "longitude": lon,
            "location_name": camera_config.get('location', ''),
            "region": camera_config.get('region', ''),
            "timezone": "America/New_York"  # Could be made configurable
        }
        
        return CameraMetadata(
            camera_info=camera_config,
            time_data=time_data,
            weather_data=weather_data,
            image_info=image_info,
            quality_metrics=quality_metrics or {},
            location_data=location_data
        )
    
    def save_metadata(self, metadata: CameraMetadata, output_path: str):
        """Save metadata to JSON file"""
        try:
            # Convert dataclass to dictionary
            metadata_dict = asdict(metadata)
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            self.logger.info(f"Metadata saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata to {output_path}: {e}")
    
    def create_metadata_filename(self, image_path: str) -> str:
        """Create corresponding metadata filename for an image"""
        image_path = Path(image_path)
        metadata_filename = image_path.stem + "_metadata.json"
        return str(image_path.parent / "metadata" / metadata_filename)

class EnhancedCameraCapture:
    """Enhanced camera capture with metadata collection"""
    
    def __init__(self, camera_manager, metadata_collector, image_validator):
        self.camera_manager = camera_manager
        self.metadata_collector = metadata_collector
        self.image_validator = image_validator
        self.logger = logging.getLogger(__name__)
    
    def capture_with_metadata(self, camera_id: str) -> Dict:
        """Capture image and collect comprehensive metadata"""
        camera = self.camera_manager.cameras.get(camera_id)
        if not camera:
            return {"error": f"Camera {camera_id} not found"}
        
        # Capture screenshot
        success, image_path, capture_metadata = self.camera_manager.capture_screenshot(
            camera_id, camera
        )
        
        if not success:
            return {
                "success": False,
                "error": "Image capture failed",
                "metadata": capture_metadata
            }
        
        try:
            # Validate image
            validation_result = self.image_validator.validate_image(image_path)
            
            # Collect quality metrics
            quality_metrics = {
                "quality_score": validation_result.get("quality_score", 0.0),
                "blur_score": validation_result.get("blur_score", 0.0),
                "brightness_score": validation_result.get("brightness_score", 0.0),
                "contrast_score": validation_result.get("contrast_score", 0.0)
            }
            
            # Create camera configuration for metadata
            camera_config = {
                "camera_id": camera_id,
                "name": camera.name,
                "url": camera.url,
                "location": camera.location,
                "region": camera.region
            }
            
            # Collect complete metadata
            complete_metadata = self.metadata_collector.collect_complete_metadata(
                camera_config, image_path, validation_result, quality_metrics
            )
            
            # Save metadata
            metadata_path = self.metadata_collector.create_metadata_filename(image_path)
            self.metadata_collector.save_metadata(complete_metadata, metadata_path)
            
            return {
                "success": True,
                "image_path": image_path,
                "metadata_path": metadata_path,
                "is_valid": validation_result["is_valid"],
                "quality_score": validation_result["quality_score"],
                "metadata": asdict(complete_metadata)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing metadata for {image_path}: {e}")
            return {
                "success": True,  # Image was captured
                "image_path": image_path,
                "error": f"Metadata processing failed: {str(e)}"
            }

def main():
    """Test the metadata collection system"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create metadata collector
    collector = MetadataCollector()
    
    # Test time metadata
    print("Time Metadata:")
    time_data = collector.collect_time_metadata()
    print(f"  Local time: {time_data.local_time}")
    print(f"  Is daytime: {time_data.is_daytime}")
    print(f"  Is rush hour: {time_data.is_rush_hour}")
    print(f"  Is weekend: {time_data.is_weekend}")
    
    # Test weather metadata (if API key available)
    print("\nWeather Metadata:")
    lat, lon = collector.get_location_coordinates("Sandy Point")
    weather_data = collector.collect_weather_metadata(lat, lon)
    if weather_data:
        print(f"  Temperature: {weather_data.temperature}°F")
        print(f"  Conditions: {weather_data.conditions}")
        print(f"  Visibility: {weather_data.visibility} km")
    else:
        print("  Weather data not available (no API key)")
    
    # Test with sample image if available
    sample_image = "screenshots"
    if os.path.exists(sample_image):
        images = [f for f in os.listdir(sample_image) if f.endswith(('.png', '.jpg'))]
        if images:
            test_image = os.path.join(sample_image, images[0])
            print(f"\nImage Metadata for {test_image}:")
            image_metadata = collector.collect_image_metadata(test_image)
            print(f"  File size: {image_metadata['file_size']} bytes")
            print(f"  Resolution: {image_metadata['resolution']}")

if __name__ == "__main__":
    main()