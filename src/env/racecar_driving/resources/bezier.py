import pybullet as p

from racecar_driving.resources.util import Vector2


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
    points = [[0 for _ in range(4 - i)] for i in range(4)]
    points[0] = [a, b, c, d]
    for i in range(1, 4):
        for j in range(4 - i):
            points[i][j] = points[i - 1][j] * (1 - t) + points[i - 1][j + 1] * t
    return points


def get_curve_point(a, b, c, d, t):
    return a * (1 - t) ** 3 + b * t * (1 - t) ** 2 * 3 + c * t ** 2 * (1 - t) * 3 + d * t ** 3


def get_distance_from_curve(point, a, b, c, d):
    dir0 = (a - b).normalised()
    dir1 = (b - c).normalised()
    dir2 = (c - d).normalised()
    if dir0 == dir1 and dir1 == dir2:
        x, y = dir0.tuple()
        orthogonal = Vector2(-y, x)
        a += orthogonal * 1e-4

    tolerance = 1e-3 / get_length(a, b, c, d)  # Find root to approximately the nearest centimetre

    # q(t) = n*t^3 + r*t^2 + s*t^ + v
    n = -a + b * 3 - c * 3 + d
    r = a * 3 - b * 6 + c * 3
    s = -a * 3 + b * 3
    v = a

    # q'(t) = j*t^2 + k*t + m
    j = n * 3
    k = r * 2
    m = s

    # Coefficients of (p-q(t))*q'(t) = 0
    inverse_lead = -1 / j.dot(n)  # All terms will be divided by the coefficient of x^5
    coefficient0 = 1
    coefficient1 = inverse_lead * -(j.dot(r) + k.dot(n))
    coefficient2 = inverse_lead * -(j.dot(s) + k.dot(r) + m.dot(n))
    coefficient3 = inverse_lead * -(j.dot(v) + k.dot(s) + m.dot(r))
    coefficient4 = inverse_lead * -(k.dot(v) + m.dot(s))
    coefficient5 = inverse_lead * -(m.dot(v))
    coefficient3 += inverse_lead * point.dot(j)
    coefficient4 += inverse_lead * point.dot(k)
    coefficient5 += inverse_lead * point.dot(m)

    sequence = SturmSequence(coefficient0, coefficient1, coefficient2, coefficient3, coefficient4, coefficient5)
    intervals = [(tolerance, 1 - tolerance)]
    num_expected_roots = sequence.get_sign_changes(tolerance) - sequence.get_sign_changes(1 - tolerance)
    roots = []
    while len(intervals) > 0 and len(roots) != num_expected_roots:
        start, end = intervals[-1]
        intervals.pop()
        num_roots = sequence.get_sign_changes(start) - sequence.get_sign_changes(end)
        if num_roots == 0:
            continue
        mid = (start + end) / 2
        if end - start < tolerance:
            # Decrement number of expected roots if multiple are found at one point
            num_expected_roots -= (num_roots - 1)
            roots.append(mid)
        else:
            intervals += [(start, mid), (mid, end)]
    roots += [0, 1]
    min_dist = float('inf')
    closest_root = 0
    for root in roots:
        root_point = get_curve_point(a, b, c, d, root)
        distance = point.get_distance(root_point)
        if distance < min_dist:
            min_dist = distance
            closest_root = root
    return closest_root, min_dist


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

    def get_total_progress(self, segment_index, t):
        total = 0
        for i in range(segment_index):
            total += self.get_segment_length(i)
        total += self.get_segment_length(segment_index) * self.get_segment_progress(segment_index, t)
        return total

    def get_segment_progress(self, segment_index, t):
        curve1, curve2 = split_at_point(*self.get_segment_points(segment_index), t)
        done = get_length(*curve1)
        remaining = get_length(*curve2)
        return done / (done + remaining)

    def get_total_length(self):
        total = 0
        for i in range(self.num_segments):
            total += self.get_segment_length(i)
        return total

    def get_segment_length(self, segment_index):
        return get_length(*self.get_segment_points(segment_index))

    def get_distance_from_curve(self, point, segment_index):
        return get_distance_from_curve(point, *self.get_segment_points(segment_index))


class SturmSequence:
    def __init__(self, *args):

        self.degree = len(args) - 1
        self.sequence = [[0 for _ in range(self.degree + 1)] for _ in range(self.degree + 1)]

        # sequence[0] = p
        self.sequence[0] = list(args)

        # sequence[1] = p'
        for i in range(self.degree):
            self.sequence[1][i + 1] = self.sequence[0][i] * (self.degree - i)

        # sequence[i] = negative remainder of sequence[i-2] / sequence[i-1] for i > 1
        for i in range(2, self.degree + 1):
            self.sequence[i] = [self.sequence[i - 2][j] for j in range(0, self.degree + 1)]
            for j in range(2):
                numerator = self.sequence[i][i - 2 + j]
                denominator = self.sequence[i - 1][i - 1]
                factor = numerator / denominator
                for k in range(i - 1, self.degree + 1):
                    self.sequence[i][j - 1 + k] -= self.sequence[i - 1][k] * factor
            self.sequence[i] = [-self.sequence[i][j] for j in range(self.degree + 1)]

    def get_sign_changes(self, t):
        changes = 0
        previous = self._solve_polynomial(t, 0) >= 0
        for i in range(1, self.degree + 1):
            current = self._solve_polynomial(t, i) >= 0
            changes += current != previous
            previous = current
        return changes

    def _solve_polynomial(self, t, index):
        total = 0
        for i in range(self.degree + 1):
            total += self.sequence[index][i] * t ** (self.degree - i)
        return total
