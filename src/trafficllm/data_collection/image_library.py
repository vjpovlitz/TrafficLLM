"""
Image Library Management System

Organizes traffic camera images with hierarchical date-based structure:
library/YYYY/MM/DD/camera_id/{raw,validated,metadata}/

Features:
- Automatic date-based organization
- Camera-specific subdirectories
- Separation of raw/validated/metadata
- Search and query utilities
- Storage statistics
- Duplicate detection
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """
    Metadata record for a captured image.
    """
    camera_id: str
    camera_name: str
    file_path: str
    capture_time: datetime
    file_size: int
    quality_score: Optional[float] = None
    validated: bool = False
    fullscreen_mode: bool = False
    location: Optional[str] = None
    region: Optional[str] = None
    weather: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['capture_time'] = self.capture_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageRecord':
        """Create from dictionary"""
        data['capture_time'] = datetime.fromisoformat(data['capture_time'])
        return cls(**data)


class ImageLibrary:
    """
    Manages hierarchical storage of traffic camera images.

    Directory structure:
        library_root/
            YYYY/
                MM/
                    DD/
                        camera_id/
                            raw/           # Original captures
                            validated/     # Quality-validated images
                            metadata/      # JSON metadata files
                            thumbnails/    # Optional thumbnails

    Example:
        >>> library = ImageLibrary("data/image_library")
        >>> library.store_image(
        ...     camera_id="us_50_sandy_point",
        ...     image_path="screenshot.png",
        ...     metadata={"quality_score": 0.85}
        ... )
    """

    def __init__(self, library_root: Path):
        """
        Initialize image library.

        Args:
            library_root: Root directory for image storage
        """
        self.root = Path(library_root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Create index directory
        self.index_dir = self.root / "_index"
        self.index_dir.mkdir(exist_ok=True)

        logger.info(f"Image library initialized: {self.root}")

    def get_storage_path(
        self,
        camera_id: str,
        capture_time: Optional[datetime] = None,
        category: str = "raw"
    ) -> Path:
        """
        Get storage path for an image.

        Args:
            camera_id: Camera identifier
            capture_time: Capture timestamp (uses now if None)
            category: Storage category (raw, validated, metadata, thumbnails)

        Returns:
            Path to storage directory
        """
        if capture_time is None:
            capture_time = datetime.now()

        # Create hierarchical path: YYYY/MM/DD/camera_id/category/
        path = (
            self.root
            / f"{capture_time.year:04d}"
            / f"{capture_time.month:02d}"
            / f"{capture_time.day:02d}"
            / camera_id
            / category
        )

        path.mkdir(parents=True, exist_ok=True)
        return path

    def store_image(
        self,
        camera_id: str,
        camera_name: str,
        image_path: Path,
        capture_time: Optional[datetime] = None,
        category: str = "raw",
        metadata: Optional[Dict] = None,
        move: bool = False
    ) -> ImageRecord:
        """
        Store an image in the library.

        Args:
            camera_id: Camera identifier
            camera_name: Human-readable camera name
            image_path: Path to image file
            capture_time: Capture timestamp
            category: Storage category
            metadata: Optional metadata dict
            move: Move file instead of copy

        Returns:
            ImageRecord for the stored image
        """
        if capture_time is None:
            capture_time = datetime.now()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get storage directory
        storage_dir = self.get_storage_path(camera_id, capture_time, category)

        # Generate filename with timestamp
        timestamp_str = capture_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        dest_filename = f"{camera_name}_{timestamp_str}{image_path.suffix}"
        dest_path = storage_dir / dest_filename

        # Copy or move image
        if move:
            shutil.move(str(image_path), str(dest_path))
        else:
            shutil.copy2(str(image_path), str(dest_path))

        logger.debug(f"Stored image: {dest_path}")

        # Create image record
        record = ImageRecord(
            camera_id=camera_id,
            camera_name=camera_name,
            file_path=str(dest_path),
            capture_time=capture_time,
            file_size=dest_path.stat().st_size,
            validated=(category == "validated"),
            location=metadata.get('location') if metadata else None,
            region=metadata.get('region') if metadata else None,
            quality_score=metadata.get('quality_score') if metadata else None,
            fullscreen_mode=metadata.get('fullscreen_mode', False) if metadata else False,
            weather=metadata.get('weather') if metadata else None
        )

        # Save metadata
        self._save_metadata(record, storage_dir)

        # Update index
        self._update_index(record)

        return record

    def _save_metadata(self, record: ImageRecord, storage_dir: Path):
        """Save image metadata as JSON"""
        metadata_dir = storage_dir.parent / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        image_filename = Path(record.file_path).stem
        metadata_path = metadata_dir / f"{image_filename}.json"

        with open(metadata_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)

    def _update_index(self, record: ImageRecord):
        """Update library index"""
        # Daily index
        date_str = record.capture_time.strftime("%Y-%m-%d")
        index_file = self.index_dir / f"{date_str}.jsonl"

        with open(index_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')

    def search_images(
        self,
        camera_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: str = "raw",
        min_quality: Optional[float] = None
    ) -> List[ImageRecord]:
        """
        Search for images matching criteria.

        Args:
            camera_id: Filter by camera
            start_date: Filter by start date
            end_date: Filter by end date
            category: Filter by category
            min_quality: Minimum quality score

        Returns:
            List of matching ImageRecord objects
        """
        results = []

        # Default date range: last 7 days
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()

        # Search index files
        current_date = start_date.date()
        end = end_date.date()

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            index_file = self.index_dir / f"{date_str}.jsonl"

            if index_file.exists():
                with open(index_file, 'r') as f:
                    for line in f:
                        record_dict = json.loads(line)
                        record = ImageRecord.from_dict(record_dict)

                        # Apply filters
                        if camera_id and record.camera_id != camera_id:
                            continue
                        if min_quality and (record.quality_score is None or record.quality_score < min_quality):
                            continue
                        if category not in record.file_path:
                            continue

                        results.append(record)

            current_date += timedelta(days=1)

        return results

    def get_camera_stats(
        self,
        camera_id: Optional[str] = None,
        days: int = 7
    ) -> Dict:
        """
        Get statistics for cameras.

        Args:
            camera_id: Specific camera (all cameras if None)
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        start_date = datetime.now() - timedelta(days=days)
        records = self.search_images(camera_id=camera_id, start_date=start_date)

        # Calculate stats
        stats = {
            'total_images': len(records),
            'total_size_mb': sum(r.file_size for r in records) / (1024 * 1024),
            'cameras': defaultdict(int),
            'by_date': defaultdict(int),
            'validated': sum(1 for r in records if r.validated),
            'fullscreen': sum(1 for r in records if r.fullscreen_mode),
            'avg_quality': None
        }

        # Per-camera counts
        for record in records:
            stats['cameras'][record.camera_id] += 1
            date_str = record.capture_time.strftime("%Y-%m-%d")
            stats['by_date'][date_str] += 1

        # Average quality
        quality_scores = [r.quality_score for r in records if r.quality_score is not None]
        if quality_scores:
            stats['avg_quality'] = sum(quality_scores) / len(quality_scores)

        # Convert defaultdicts to regular dicts
        stats['cameras'] = dict(stats['cameras'])
        stats['by_date'] = dict(sorted(stats['by_date'].items()))

        return stats

    def cleanup_old_images(
        self,
        days_to_keep: int = 30,
        category: str = "raw",
        dry_run: bool = True
    ) -> Tuple[int, int]:
        """
        Clean up old images.

        Args:
            days_to_keep: Keep images from last N days
            category: Category to clean (raw, validated, etc.)
            dry_run: If True, only report what would be deleted

        Returns:
            Tuple of (files_deleted, bytes_freed)
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        files_deleted = 0
        bytes_freed = 0

        # Search for old images
        old_records = self.search_images(
            start_date=datetime.min,
            end_date=cutoff_date,
            category=category
        )

        for record in old_records:
            file_path = Path(record.file_path)
            if file_path.exists():
                size = file_path.stat().st_size

                if not dry_run:
                    file_path.unlink()
                    logger.info(f"Deleted: {file_path}")
                else:
                    logger.info(f"Would delete: {file_path}")

                files_deleted += 1
                bytes_freed += size

        logger.info(
            f"Cleanup {'(dry run)' if dry_run else ''}: "
            f"{files_deleted} files, {bytes_freed / (1024**2):.1f} MB"
        )

        return files_deleted, bytes_freed

    def export_catalog(self, output_path: Path):
        """
        Export complete catalog to JSON.

        Args:
            output_path: Path to save catalog
        """
        catalog = {
            'generated_at': datetime.now().isoformat(),
            'library_root': str(self.root),
            'cameras': {},
            'statistics': self.get_camera_stats(days=365)
        }

        # Get all cameras
        camera_dirs = set()
        for year_dir in self.root.glob('[0-9][0-9][0-9][0-9]'):
            for month_dir in year_dir.glob('[0-9][0-9]'):
                for day_dir in month_dir.glob('[0-9][0-9]'):
                    for cam_dir in day_dir.iterdir():
                        if cam_dir.is_dir() and not cam_dir.name.startswith('_'):
                            camera_dirs.add(cam_dir.name)

        # Get stats for each camera
        for camera_id in sorted(camera_dirs):
            catalog['cameras'][camera_id] = self.get_camera_stats(
                camera_id=camera_id,
                days=365
            )

        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Catalog exported: {output_path}")

    def validate_integrity(self) -> Dict:
        """
        Validate library integrity.

        Returns:
            Dictionary with validation results
        """
        results = {
            'total_records': 0,
            'missing_files': [],
            'missing_metadata': [],
            'corrupt_metadata': [],
            'orphaned_files': []
        }

        # Check index files
        for index_file in self.index_dir.glob('*.jsonl'):
            with open(index_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record_dict = json.loads(line)
                        record = ImageRecord.from_dict(record_dict)
                        results['total_records'] += 1

                        # Check if file exists
                        if not Path(record.file_path).exists():
                            results['missing_files'].append(record.file_path)

                    except json.JSONDecodeError:
                        results['corrupt_metadata'].append(f"{index_file}:{line_num}")
                    except Exception as e:
                        logger.error(f"Error validating {index_file}:{line_num}: {e}")

        logger.info(
            f"Integrity check: {results['total_records']} records, "
            f"{len(results['missing_files'])} missing files"
        )

        return results
