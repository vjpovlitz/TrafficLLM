"""
Offline Traffic Monitor using Unified Pipeline
Displays real-time detection, tracking, and counting using the centralized pipeline.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trafficllm.core.pipeline import UnifiedTrafficPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Offline Traffic Monitor via Unified Pipeline")
    parser.add_argument("--source", type=str, default="https://cams.cdn-surfline.com/cdn-wc/wc-sandypoint/playlist.m3u8", help="Video source (URL or file path)")
    parser.add_argument("--model_size", type=str, default="s", help="YOLO model size (n, s, m, l, x)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--classifier_path", type=str, default=None, help="Path to trained classifier checkpoint")
    args = parser.parse_args()

    # Initialize Unified Pipeline
    print(f"🚀 Initializing Unified Pipeline with YOLOv8{args.model_size}...")
    pipeline = UnifiedTrafficPipeline(
        yolo_model_size=args.model_size,
        classifier_path=args.classifier_path
    )

    # Run Monitor
    print(f"🎥 Starting monitor on source: {args.source}")
    pipeline.run_monitor(args.source, conf_threshold=args.conf)

if __name__ == "__main__":
    main()
