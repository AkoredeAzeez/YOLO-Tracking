import cv2
import numpy as np
from geofence import Geofence

class PolygonGeofence(Geofence):
    def __init__(self, vertices=None, frame_width=640, frame_height=480):
        """
        Initialize a polygon geofence

        If vertices is None, prompts the user to either:
         - Load vertices from file
         - Enter vertices manually
         - Or use default hexagon
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        if vertices is None:
            print("\nNo vertices provided for PolygonGeofence.")
            choice = input("Enter 'm' for manual input, 'f' to load from file, or press Enter to use default hexagon: ").strip().lower()
            
            if choice == 'm':
                self.polygon_points = self._prompt_for_vertices()
            elif choice == 'f':
                file_path = input("Enter file path for vertices: ").strip()
                self.polygon_points = load_vertices_from_file(file_path)
            else:
                self.polygon_points = self._create_default_hexagon()
        else:
            self.polygon_points = vertices

        print(f"Polygon geofence created with {len(self.polygon_points)} vertices: {self.polygon_points}")

    def _create_default_hexagon(self):
        """Create a default hexagon in the center of the frame"""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        radius = min(self.frame_width, self.frame_height) // 4

        vertices = []
        for i in range(6):
            angle = i * np.pi / 3  # 60 degrees in radians
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            vertices.append((x, y))

        return vertices

    def _prompt_for_vertices(self):
        """Prompt user to manually enter vertices"""
        try:
            num_points = int(input("How many vertices? (min 3): ").strip())
            if num_points < 3:
                raise ValueError("A polygon needs at least 3 vertices!")

            vertices = []
            for i in range(num_points):
                x = int(input(f"Enter x for vertex {i+1}: ").strip())
                y = int(input(f"Enter y for vertex {i+1}: ").strip())
                vertices.append((x, y))

            return vertices
        except Exception as e:
            print(f"Error during manual input: {e}")
            print("Falling back to default hexagon.")
            return self._create_default_hexagon()

    def point_in_polygon(self, point):
        """Check if a point is inside the polygon"""
        x, y = point
        n = len(self.polygon_points)
        inside = False

        p1x, p1y = self.polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def draw_geofence(self, frame):
        """Draw the polygon geofence on the frame"""
        points = np.array(self.polygon_points, np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

        for i, point in enumerate(self.polygon_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def parse_vertices_string(vertices_str):
    """
    Parse vertices from string
    """
    try:
        vertices = []
        coord_pairs = vertices_str.strip().split()

        for pair in coord_pairs:
            x_str, y_str = pair.split(',')
            x = int(x_str.strip())
            y = int(y_str.strip())
            vertices.append((x, y))

        if len(vertices) < 3:
            raise ValueError("At least 3 vertices required for a polygon")

        return vertices
    except Exception as e:
        raise ValueError(f"Invalid vertices format: {e}")


def load_vertices_from_file(file_path):
    """
    Load vertices from a text file
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        if ' ' in content and '\n' not in content:
            return parse_vertices_string(content)

        vertices = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                x_str, y_str = line.split(',')
                x = int(x_str.strip())
                y = int(y_str.strip())
                vertices.append((x, y))

        if len(vertices) < 3:
            raise ValueError("At least 3 vertices required for a polygon")

        return vertices
    except Exception as e:
        raise ValueError(f"Error loading vertices from file: {e}")
