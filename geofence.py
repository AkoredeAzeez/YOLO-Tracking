import cv2
import numpy as np

class Geofence:
    def __init__(self, vertices=None, frame_width=640, frame_height=480, num_vertices=3):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_vertices = num_vertices
        if vertices is None:
            print(f"\nNo vertices provided for {self.__class__.__name__}.")
            choice = input(self._input_prompt()).strip().lower()
            if choice == 'm':
                self.points = self._prompt_for_vertices()
            elif choice == 'f' and num_vertices > 3:
                file_path = input("Enter file path for vertices: ").strip()
                self.points = self._load_vertices_from_file(file_path)
            else:
                self.points = self._create_default_shape()
        else:
            if num_vertices == 3 and len(vertices) != 3:
                raise ValueError("Triangle must have exactly 3 vertices!")
            if num_vertices > 3 and len(vertices) < 3:
                raise ValueError("Polygon must have at least 3 vertices!")
            self.points = vertices
        print(f"{self.__class__.__name__} created with vertices: {self.points}")

    def _input_prompt(self):
        if self.num_vertices == 3:
            return "Enter 'm' to manually enter triangle vertices, or press Enter to use default centered triangle: "
        else:
            return "Enter 'm' for manual input, 'f' to load from file, or press Enter to use default hexagon: "

    def _create_default_shape(self):
        if self.num_vertices == 3:
            # Default triangle
            center_x = self.frame_width // 2
            center_y = self.frame_height // 2
            size = min(self.frame_width, self.frame_height) // 4
            return [
                (center_x, center_y - size),
                (center_x - size, center_y + size // 2),
                (center_x + size, center_y + size // 2)
            ]
        else:
            # Default hexagon
            center_x = self.frame_width // 2
            center_y = self.frame_height // 2
            radius = min(self.frame_width, self.frame_height) // 4
            vertices = []
            for i in range(6):
                angle = i * np.pi / 3
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                vertices.append((x, y))
            return vertices

    def _prompt_for_vertices(self):
        vertices = []
        try:
            if self.num_vertices == 3:
                print("Enter coordinates for the 3 triangle vertices:")
                for i in range(3):
                    x = int(input(f"Enter x for vertex {i+1}: ").strip())
                    y = int(input(f"Enter y for vertex {i+1}: ").strip())
                    vertices.append((x, y))
            else:
                num_points = int(input("How many vertices? (min 3): ").strip())
                if num_points < 3:
                    raise ValueError("A polygon needs at least 3 vertices!")
                for i in range(num_points):
                    x = int(input(f"Enter x for vertex {i+1}: ").strip())
                    y = int(input(f"Enter y for vertex {i+1}: ").strip())
                    vertices.append((x, y))
        except Exception as e:
            print(f"Error during manual input: {e}")
            print("Falling back to default shape.")
            return self._create_default_shape()
        return vertices

    def _parse_vertices_string(self, vertices_str):
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

    def _load_vertices_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            if ' ' in content and '\n' not in content:
                return self._parse_vertices_string(content)
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

    def draw_geofence(self, frame):
        points = np.array(self.points, np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        for i, point in enumerate(self.points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            if self.num_vertices > 3:
                cv2.putText(frame, str(i), (point[0] + 8, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)