# modified_yolo_tracker.py
import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Your existing imports
from consolidator import Consolidator
from embedding_aggregator import EmbeddingAggregator
from image_saver import ImageSaver
from scheduler import Scheduler
from triangle_geofence import TriangleGeofence
from util import get_logger, get_unique_path


class YoloTracker:
    def __init__(self, source, skip_frames, output_path, preview=False, save_video=False, save_images=False, skip_consolidation=False):
        self.SOURCE = source
        self.SHOULD_PREVIEW = preview
        self.SHOULD_SAVE_VIDEO = save_video
        self.SKIP_FRAMES = skip_frames
        self.OUTPUT_PATH = get_unique_path(output_path)
        self.SHOULD_SAVE_IMAGES = save_images
        self.SHOULD_CONSOLIDATE = not skip_consolidation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.classification_model = YOLO(".yolo/models/yolo11n-cls.pt").to(self.device)
        self.model = YOLO(".yolo/models/yolo11n.pt").to(self.device)
        self.layer_indices = [2, 4, 6, 8, 9]

        self.consolidator = Consolidator(self.OUTPUT_PATH)
        self.scheduler = Scheduler(
            save_path=self.OUTPUT_PATH,
            skip_frames=self.SKIP_FRAMES,
            consolidator=self.consolidator,
        ) | (ImageSaver() if self.SHOULD_SAVE_IMAGES else None) | EmbeddingAggregator(self.classification_model, self.layer_indices, batch_size=50)

        self.logger = get_logger(__name__, f"{self.scheduler.save_path}/logs")
        
        # Initialize geofence (will be updated once we get first frame)
        self.geofence = TriangleGeofence()
        self.geofence_initialized = False

    def trigger_geofence_event(self, detected_object):
        """
        Called when an object enters the geofence
        Add your custom trigger logic here
        """
        print(f"🚨 GEOFENCE TRIGGER: {detected_object['class']} entered geofence!")
        self.logger.info(f"Geofence trigger: {detected_object['class']} at {detected_object['center']}")
        
        # Add your custom logic here:
        # - Send alert
        # - Save image
        # - Log to database
        # - etc.

    def process_geofence_detections(self, frame, results):
        """
        Process detections for geofence monitoring
        """
        if not self.geofence_initialized:
            h, w = frame.shape[:2]
            self.geofence = TriangleGeofence(w, h)
            self.geofence_initialized = True
        
        # Draw geofence on frame
        self.geofence.draw_geofence(frame)
        
        objects_inside = 0
        
        # Process YOLO detections
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)
                
                # Get class name
                class_name = self.model.names[cls]
                
                # Check if object is inside geofence
                is_inside = self.geofence.point_in_triangle(center)
                
                # Create detection object
                detected_object = {
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'center': center,
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': cls
                }
                
                # Draw bounding box with color based on geofence status
                color = (0, 255, 0) if is_inside else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, color, -1)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if is_inside:
                    objects_inside += 1
                    self.trigger_geofence_event(detected_object)
        
        # Display detection status
        status_text = f"Objects in geofence: {objects_inside}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return objects_inside

    def run(self):
        self.consolidator.start()

        self.logger.info(f"Tracking {self.SOURCE} with {self.model.model_name}")
        self.logger.info(f"Embedding model: {self.classification_model.model_name}")
        self.logger.info(f"Skipping {self.SKIP_FRAMES} frames")
        self.logger.info(f"\n{self.scheduler}")

        try:
            results: list[Results] = self.model.track(
                source=self.SOURCE, stream=True, verbose=False,
                persist=True, tracker="trackers/botsort_with_reid.yaml",
                save=self.SHOULD_SAVE_VIDEO, project=self.OUTPUT_PATH,
            )
            start_time = time.perf_counter()

            start_detection_time = time.perf_counter()
            for result in results:
                self.scheduler(result)
                end_detection_time = time.perf_counter()
                self.logger.debug(f"Detection took {end_detection_time-start_detection_time:.4f} seconds")
                start_detection_time = end_detection_time

                if self.SHOULD_PREVIEW:
                    im0 = result.plot()
                    
                    # Process geofence detections
                    self.process_geofence_detections(im0, result)
                    
                    cv2.imshow("YOLO Tracking with Geofence", im0)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break
                    elif key == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            end_time = time.perf_counter()
            self.scheduler.cleanup()
            cv2.destroyAllWindows()
            self.logger.info(f"Took {end_time-start_time:.4f} seconds")
            self.consolidator.stop()
            self.consolidator.join()
