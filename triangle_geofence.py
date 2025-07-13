import numpy as np
import cv2

class TriangleGeofence:
    def __init__(self, frame_width=640, frame_height=480):
        """
        Initialize a triangle geofence in the center of the frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define triangle vertices (can be adjusted)
        center_x = frame_width // 2
        center_y = frame_height // 2
        size = min(frame_width, frame_height) // 4
        
        # Create a triangle pointing upward
        self.triangle_points = [
            (center_x, center_y - size),      # Top vertex
            (center_x - size, center_y + size//2),  # Bottom left
            (center_x + size, center_y + size//2)   # Bottom right
        ]
        
        print(f"Triangle geofence created with vertices: {self.triangle_points}")
    
    def point_in_triangle(self, point):
        """
        Check if a point is inside the triangle using barycentric coordinates
        """
        x, y = point
        x1, y1 = self.triangle_points[0]
        x2, y2 = self.triangle_points[1]
        x3, y3 = self.triangle_points[2]
        
        # Calculate barycentric coordinates
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denominator) < 1e-10:
            return False
            
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b
        
        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1
    
    def draw_geofence(self, frame):
        """
        Draw the triangle geofence on the frame
        """
        # Convert points to numpy array for OpenCV
        points = np.array(self.triangle_points, np.int32)
        
        # Draw filled triangle (semi-transparent)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw triangle outline
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # Draw vertices
        for i, point in enumerate(self.triangle_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

