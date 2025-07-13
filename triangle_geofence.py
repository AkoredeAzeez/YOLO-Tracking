from geofence import Geofence

class TriangleGeofence(Geofence):
    def __init__(self, vertices=None, frame_width=640, frame_height=480):
        super().__init__(vertices, frame_width, frame_height, num_vertices=3)

    def point_in_triangle(self, point):
        x, y = point
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        x3, y3 = self.points[2]
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denominator) < 1e-10:
            return False
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b
        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1
