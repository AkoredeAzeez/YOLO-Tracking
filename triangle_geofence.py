import cv2
import numpy as np
from geofence import Geofence

class TriangleGeofence(Geofence):
    def __init__(self, vertices=None, frame_width=640, frame_height=480):
        """
        Initialize a triangle geofence in the center of the frame
        OR use custom vertices if provided.
        If vertices is None, ask user for custom input or use default.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        if vertices is None:
            print("\nNo vertices provided for TriangleGeofence.")
            choice = input("Enter 'm' to manually enter triangle vertices, or press Enter to use default centered triangle: ").strip().lower()
            
            if choice == 'm':
                self.triangle_points = self._prompt_for_vertices()
            else:
                self.triangle_points = self._create_default_triangle()
        else:
            if len(vertices) != 3:
                raise ValueError("Triangle must have exactly 3 vertices!")
            self.triangle_points = vertices

        print(f"Triangle geofence created with vertices: {self.triangle_points}")

    def _create_default_triangle(self):
        """Create a default triangle pointing upward in the center"""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        size = min(self.frame_width, self.frame_height) // 4

        return [
            (center_x, center_y - size),                 # Top vertex
            (center_x - size, center_y + size // 2),     # Bottom left
            (center_x + size, center_y + size // 2)      # Bottom right
        ]

    def _prompt_for_vertices(self):
        """Prompt user to manually enter 3 vertices for the triangle"""
        vertices = []
        try:
            print("Enter coordinates for the 3 triangle vertices:")
            for i in range(3):
                x = int(input(f"Enter x for vertex {i+1}: ").strip())
                y = int(input(f"Enter y for vertex {i+1}: ").strip())
                vertices.append((x, y))
        except Exception as e:
            print(f"Error during manual input: {e}")
            print("Falling back to default triangle.")
            return self._create_default_triangle()

        return vertices

    def point_in_triangle(self, point):
        """
        Check if a point is inside the triangle using barycentric coordinates
        """
        x, y = point
        x1, y1 = self.triangle_points[0]
        x2, y2 = self.triangle_points[1]
        x3, y3 = self.triangle_points[2]

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
        points = np.array(self.triangle_points, np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

        for i, point in enumerate(self.triangle_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
