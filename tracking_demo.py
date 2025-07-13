from ultralytics import YOLO
from ultralytics.engine.results import Results

from consolidator import Consolidator
from embedding_aggregator import EmbeddingAggregator
from image_saver import ImageSaver
from scheduler import Scheduler
from util import get_logger, get_unique_path

import argparse
from yolo_tracker import YoloTracker

parser = argparse.ArgumentParser(description='YOLO tracking with triangle geofence')
parser.add_argument('--source', type=str, default="store.mp4")
parser.add_argument('--skip-frames', type=int, default=5)
parser.add_argument('--output-path', type=str, default="output/results")
parser.add_argument('--preview', action='store_true', default=True)
parser.add_argument('--save-video', action='store_true')
parser.add_argument('--save-images', action='store_true')
parser.add_argument('--skip-consolidation', action='store_true')
args = parser.parse_args()

tracker = YoloTracker(
    source=args.source,
    skip_frames=args.skip_frames,
    output_path=args.output_path,
    preview=args.preview,
    save_video=args.save_video,
    save_images=args.save_images,
    skip_consolidation=args.skip_consolidation,
)
tracker.run()