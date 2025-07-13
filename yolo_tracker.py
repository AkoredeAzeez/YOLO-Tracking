import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import torch

from consolidator import Consolidator
from embedding_aggregator import EmbeddingAggregator
from image_saver import ImageSaver
from scheduler import Scheduler
from triangle_geofence import TriangleGeofence
from polygon_geofence import PolygonGeofence  # ✅ make sure you have this
from util import get_logger, get_unique_path, plot_from_track_results
from video_writer import VideoWriter


class YoloTracker:
    def __init__(
        self, source,
        skip_frames: int = 5,
        output_path: str = "output/results",
        preview: bool = False,
        save_video: bool = False, save_images: bool = False,
        skip_consolidation: bool = False,
        only_person: bool = False, use_beta: bool = False
    ):
        self.SOURCE = source
        self.SHOULD_PREVIEW = preview
        self.SHOULD_SAVE_VIDEO = save_video
        self.OUTPUT_PATH = get_unique_path(output_path)
        self.SHOULD_SAVE_IMAGES = save_images
        self.ONLY_PERSON = only_person

        self._load_model(use_beta)
        self._load_components(skip_consolidation, skip_frames)

        self.logger = get_logger(self.__class__.__name__, f"{self.scheduler.save_path}/logs")
        self.logger.info(f"Tracking {self.SOURCE} with {self.model.model_name}")
        self.logger.info(f"Embedding model: {self.classification_model.model_name}")
        self.logger.info(f"Skipping {skip_frames} frames")
        self.logger.info(f"\n{self.scheduler}")

        self._start_time = None

    def _load_model(self, use_beta: bool):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if use_beta:
            self.classification_model = YOLO(".yolo/models/yolo11n-cls.pt").to(self.device)
            self.model = YOLO(".yolo/models/yolo11n.pt").to(self.device)
            self.layer_indices = [2, 4, 6, 8, 9]
        else:
            self.classification_model = YOLO(".yolo/models/yolov8n-cls.pt").to(self.device)
            self.model = YOLO(".yolo/models/yolov8n.pt").to(self.device)
            self.layer_indices = [2, 4, 6, 8]

    def _load_components(self, skip_consolidation: bool, skip_frames: int):
        if skip_consolidation:
            self.consolidator = None
        else:
            self.consolidator = Consolidator(self.OUTPUT_PATH, interval=None)
            self.consolidator.start()

        self.scheduler = Scheduler(
            save_path=self.OUTPUT_PATH,
            skip_frames=skip_frames,
            consolidator=self.consolidator,
        ) \
            | (ImageSaver() if self.SHOULD_SAVE_IMAGES else None) \
            | EmbeddingAggregator(self.classification_model, self.layer_indices, batch_size=50)

        if self.SHOULD_SAVE_VIDEO:
            self.video_writer = VideoWriter(self.OUTPUT_PATH + '/output.avi')
        else:
            self.video_writer = None

        choice = input("Choose geofence type: A for Polygon or B for Triangle: ").strip().lower()
        if choice == 'a':
            self.geofence = PolygonGeofence()
            self._geofence_type = "polygon"
            print("✅ Using Polygon Geofence")
        else:
            self.geofence = TriangleGeofence()
            self._geofence_type = "triangle"
            print("✅ Using Triangle Geofence")

        self.geofence_initialized = False

    def trigger_geofence_event(self, detected_object):
        print(f"🚨 GEOFENCE TRIGGER: {detected_object['class']} entered geofence!")
        self.logger.info(f"Geofence trigger: {detected_object['class']} at {detected_object['center']}")

    def process_geofence_detections(self, frame, results):
        if not self.geofence_initialized:
            self.geofence_initialized = True

        self.geofence.draw_geofence(frame)

        objects_inside = 0

        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)

                class_name = self.model.names[cls]

                # ✅ Use the correct geofence method
                if self._geofence_type == "polygon":
                    is_inside = self.geofence.point_in_polygon(center)
                else:
                    is_inside = self.geofence.point_in_triangle(center)

                detected_object = {
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'center': center,
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': cls
                }

                color = (0, 255, 0) if is_inside else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, center, 5, color, -1)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if is_inside:
                    objects_inside += 1
                    self.trigger_geofence_event(detected_object)

        status_text = f"Objects in geofence: {objects_inside}"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return objects_inside

    def run(self):
        results: list[Results] = self.model.track(
            source=self.SOURCE, stream=True, verbose=False,
            persist=True, tracker="trackers/botsort_with_reid.yaml",
            project=self.OUTPUT_PATH,
            classes=[0] if self.ONLY_PERSON else None,
            conf=0.5 if self.ONLY_PERSON else 0.5
        )
        self._start_time = time.perf_counter()
        start_detection_time = time.perf_counter()
        for result in results:
            self.scheduler(result)
            end_detection_time = time.perf_counter()
            self.logger.debug(f"Detection took {end_detection_time - start_detection_time:.4f} seconds")
            start_detection_time = end_detection_time

            if self.SHOULD_PREVIEW or self.SHOULD_SAVE_VIDEO:
                im0 = plot_from_track_results(result, self.consolidator)
                if self.SHOULD_PREVIEW:
                    self.process_geofence_detections(im0, result)
                    cv2.imshow("YOLO Tracking", im0)
                    if cv2.waitKey(50) & 0xFF == 27:
                        break
                if self.video_writer:
                    self.video_writer.write(im0)

    def cleanup(self):
        end_time = time.perf_counter()
        self.scheduler.cleanup()
        cv2.destroyAllWindows()
        if self._start_time is not None:
            self.logger.info(f"Took {end_time - self._start_time:.4f} seconds")
        if self.consolidator: self.consolidator.cleanup()
        if self.video_writer: self.video_writer.cleanup()
