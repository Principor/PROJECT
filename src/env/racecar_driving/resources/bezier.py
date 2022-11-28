import pybullet as p


def get_length(a, b, c, d):
    return ((a.get_distance(b) + b.get_distance(c) + c.get_distance(d)) + a.get_distance(d)) / 2


def split_curve_recursive(a, b, c, d, max_length=5):
    if get_length(a, b, c, d) < max_length:
        return [d]
    curve1, curve2 = split_at_point(a, b, c, d, 0.5)
    return split_curve_recursive(*curve1, max_length=max_length) + split_curve_recursive(*curve2, max_length=max_length)


def split_at_point(a, b, c, d, t):
    lerp_points = get_lerp_points(a, b, c, d, t)
    return ((lerp_points[0][0], lerp_points[1][0], lerp_points[2][0], lerp_points[3][0]),
            (lerp_points[3][0], lerp_points[2][1], lerp_points[1][2], lerp_points[0][3]))


def get_lerp_points(a, b, c, d, t):
    points = [[0 for _ in range(4-i)] for i in range(4)]
    points[0] = [a, b, c, d]
    for i in range(1, 4):
        for j in range(4-i):
            points[i][j] = points[i-1][j] * t + points[i-1][j+1] * (1-t)
    return points


class Bezier:
    def __init__(self, *args):
        self.control_points = args
        self.num_points = len(self.control_points)
        self.num_segments = self.num_points // 3

    def get_segment_points(self, segment_index):
        return tuple(self.get_segment_point(segment_index, i) for i in range(4))

    def get_segment_point(self, segment_index, point_index):
        return self.get_control_point(segment_index * 3 + point_index)

    def get_control_point(self, point_index):
        return self.control_points[point_index % self.num_points]

    def draw_lines(self, client):
        previous_point = self.get_segment_point(0, 0).make_3d(0.1)
        for segment in range(self.num_segments):
            points = split_curve_recursive(*tuple(self.get_segment_points(segment)))
            for point in points:
                current_point = point.make_3d(0.1)
                p.addUserDebugLine(previous_point.tuple(),
                                   current_point.tuple(),
                                   lineColorRGB=(1, 0, 0),
                                   lineWidth=1,
                                   physicsClientId=client)
                previous_point = current_point
