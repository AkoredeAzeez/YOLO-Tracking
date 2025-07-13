from geofence import Geofence

class PolygonGeofence(Geofence):
    def __init__(self, vertices=None, frame_width=640, frame_height=480):
        super().__init__(vertices, frame_width, frame_height, num_vertices=6)

    def point_in_polygon(self, point):
        x, y = point
        n = len(self.points)
        inside = False
        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
