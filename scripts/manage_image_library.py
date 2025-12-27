"""
Image Library Management CLI

Provides utilities for managing the traffic image library.

Usage:
    # View library statistics
    python manage_image_library.py stats

    # Search for images
    python manage_image_library.py search --camera us_50_sandy_point --days 7

    # Export catalog
    python manage_image_library.py export --output library_catalog.json

    # Cleanup old images
    python manage_image_library.py cleanup --days 30 --dry-run

    # Validate library integrity
    python manage_image_library.py validate
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from trafficllm.data_collection.image_library import ImageLibrary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_stats(library, args):
    """Display library statistics"""
    logger.info(f"\n{'='*80}")
    logger.info("Image Library Statistics")
    logger.info(f"{'='*80}")
    logger.info(f"Library: {library.root}")
    logger.info(f"Period: Last {args.days} days")
    logger.info(f"{'='*80}\n")

    # Overall stats
    stats = library.get_camera_stats(days=args.days)

    logger.info("Overall:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")
    logger.info(f"  Validated: {stats['validated']}")
    logger.info(f"  Fullscreen: {stats['fullscreen']}")
    if stats['avg_quality']:
        logger.info(f"  Avg quality: {stats['avg_quality']:.3f}")

    # Per-camera stats
    if stats['cameras']:
        logger.info(f"\nPer-camera breakdown:")
        for camera_id, count in sorted(stats['cameras'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {camera_id}: {count} images")

    # Daily breakdown
    if stats['by_date']:
        logger.info(f"\nDaily breakdown:")
        for date_str, count in sorted(stats['by_date'].items())[-7:]:  # Last 7 days
            logger.info(f"  {date_str}: {count} images")

    # Per-camera detailed stats
    if args.detailed:
        logger.info(f"\n{'='*80}")
        logger.info("Detailed Camera Statistics")
        logger.info(f"{'='*80}\n")

        for camera_id in stats['cameras'].keys():
            cam_stats = library.get_camera_stats(camera_id=camera_id, days=args.days)
            logger.info(f"{camera_id}:")
            logger.info(f"  Images: {cam_stats['total_images']}")
            logger.info(f"  Size: {cam_stats['total_size_mb']:.1f} MB")
            logger.info(f"  Validated: {cam_stats['validated']}")
            if cam_stats['avg_quality']:
                logger.info(f"  Avg quality: {cam_stats['avg_quality']:.3f}")
            logger.info("")


def cmd_search(library, args):
    """Search for images"""
    logger.info(f"\n{'='*80}")
    logger.info("Image Search")
    logger.info(f"{'='*80}\n")

    # Parse dates
    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date)
    else:
        start_date = datetime.now() - timedelta(days=args.days)

    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)
    else:
        end_date = datetime.now()

    logger.info(f"Search criteria:")
    logger.info(f"  Camera: {args.camera or 'All'}")
    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Category: {args.category}")
    if args.min_quality:
        logger.info(f"  Min quality: {args.min_quality}")
    logger.info("")

    # Search
    results = library.search_images(
        camera_id=args.camera,
        start_date=start_date,
        end_date=end_date,
        category=args.category,
        min_quality=args.min_quality
    )

    logger.info(f"Found {len(results)} matching images\n")

    # Display results
    if args.limit:
        results = results[:args.limit]

    for record in results:
        logger.info(f"{record.capture_time.strftime('%Y-%m-%d %H:%M:%S')} - {record.camera_id}")
        logger.info(f"  Path: {record.file_path}")
        logger.info(f"  Size: {record.file_size / 1024:.1f} KB")
        if record.quality_score:
            logger.info(f"  Quality: {record.quality_score:.3f}")
        logger.info("")

    # Save to file if requested
    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


def cmd_export(library, args):
    """Export library catalog"""
    logger.info(f"\n{'='*80}")
    logger.info("Exporting Library Catalog")
    logger.info(f"{'='*80}\n")

    output_path = Path(args.output)
    library.export_catalog(output_path)

    logger.info(f"✅ Catalog exported to: {output_path}")

    # Show summary
    with open(output_path, 'r') as f:
        catalog = json.load(f)

    logger.info(f"\nCatalog summary:")
    logger.info(f"  Total cameras: {len(catalog['cameras'])}")
    logger.info(f"  Total images: {catalog['statistics']['total_images']}")
    logger.info(f"  Total size: {catalog['statistics']['total_size_mb']:.1f} MB")


def cmd_cleanup(library, args):
    """Cleanup old images"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Cleanup {'(DRY RUN)' if args.dry_run else '(LIVE)'}")
    logger.info(f"{'='*80}\n")

    logger.info(f"Parameters:")
    logger.info(f"  Days to keep: {args.days}")
    logger.info(f"  Category: {args.category}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("")

    files_deleted, bytes_freed = library.cleanup_old_images(
        days_to_keep=args.days,
        category=args.category,
        dry_run=args.dry_run
    )

    logger.info(f"\n{'='*80}")
    logger.info("Cleanup Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Files {'would be' if args.dry_run else ''} deleted: {files_deleted}")
    logger.info(f"Space {'would be' if args.dry_run else ''} freed: {bytes_freed / (1024**2):.1f} MB")

    if args.dry_run:
        logger.info(f"\nTo actually delete, run without --dry-run")


def cmd_validate(library, args):
    """Validate library integrity"""
    logger.info(f"\n{'='*80}")
    logger.info("Library Integrity Validation")
    logger.info(f"{'='*80}\n")

    results = library.validate_integrity()

    logger.info("Validation results:")
    logger.info(f"  Total records: {results['total_records']}")
    logger.info(f"  Missing files: {len(results['missing_files'])}")
    logger.info(f"  Missing metadata: {len(results['missing_metadata'])}")
    logger.info(f"  Corrupt metadata: {len(results['corrupt_metadata'])}")

    # Show issues
    if results['missing_files']:
        logger.info(f"\nMissing files (first 10):")
        for path in results['missing_files'][:10]:
            logger.info(f"  - {path}")

    if results['corrupt_metadata']:
        logger.info(f"\nCorrupt metadata:")
        for location in results['corrupt_metadata']:
            logger.info(f"  - {location}")

    # Overall status
    total_issues = (
        len(results['missing_files']) +
        len(results['missing_metadata']) +
        len(results['corrupt_metadata'])
    )

    logger.info(f"\n{'='*80}")
    if total_issues == 0:
        logger.info("✅ Library integrity: PASSED")
    else:
        logger.warning(f"⚠️  Library integrity: {total_issues} issues found")


def main():
    parser = argparse.ArgumentParser(
        description="Image Library Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--library',
        type=Path,
        default=Path('data/image_library'),
        help='Path to image library (default: data/image_library)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show library statistics')
    stats_parser.add_argument('--days', type=int, default=7, help='Days to analyze (default: 7)')
    stats_parser.add_argument('--detailed', action='store_true', help='Show per-camera details')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for images')
    search_parser.add_argument('--camera', help='Filter by camera ID')
    search_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    search_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    search_parser.add_argument('--days', type=int, default=7, help='Days to search (if no start-date)')
    search_parser.add_argument('--category', default='raw', help='Category (default: raw)')
    search_parser.add_argument('--min-quality', type=float, help='Minimum quality score')
    search_parser.add_argument('--limit', type=int, help='Limit results')
    search_parser.add_argument('--output', type=Path, help='Save results to JSON')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export library catalog')
    export_parser.add_argument('--output', type=Path, default=Path('library_catalog.json'),
                               help='Output file (default: library_catalog.json)')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old images')
    cleanup_parser.add_argument('--days', type=int, default=30,
                               help='Keep images from last N days (default: 30)')
    cleanup_parser.add_argument('--category', default='raw',
                               help='Category to clean (default: raw)')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be deleted without deleting')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate library integrity')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize library
    library_path = project_root / args.library
    library = ImageLibrary(library_path)

    # Run command
    if args.command == 'stats':
        cmd_stats(library, args)
    elif args.command == 'search':
        cmd_search(library, args)
    elif args.command == 'export':
        cmd_export(library, args)
    elif args.command == 'cleanup':
        cmd_cleanup(library, args)
    elif args.command == 'validate':
        cmd_validate(library, args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
