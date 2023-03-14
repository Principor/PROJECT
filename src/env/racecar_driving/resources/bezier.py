import glob
import math
import os
import pickle

import pybullet as p

from racecar_driving.resources.util import Vector2


SAVE_PATH = "..\\tracks\\"
EXTENSION = ".track"


def get_length(a, b, c, d):
    """
    Approximate the length of the curve

    :param a: 1st control point
    :param b: 2nd control point
    :param c: 3rd control point
    :param d: 4th Control point
    :return: The length of the curve
    """
    return ((a.get_distance(b) + b.get_distance(c) + c.get_distance(d)) + a.get_distance(d)) / 2


def split_curve_recursive(a, b, c, d, max_length=5):
    """
    Split a curve into sub-curves smaller than max_length

    :param a: 1st control point
    :param b: 2nd control point
    :param c: 3rd control point
    :param d: 4th Control point
    :param max_length: Maximum length of the sub curves
    :return:
    """
    if get_length(a, b, c, d) < max_length:
        return [d]
    curve1, curve2 = split_at_point(a, b, c, d, 0.5)
    return split_curve_recursive(*curve1, max_length=max_length) + split_curve_recursive(*curve2, max_length=max_length)


def split_at_point(a, b, c, d, t):
    """
    Split a curve at the point q(t)

    :param a: 1st control point
    :param b: 2nd control point
    :param c: 3rd control point
    :param d: 4th Control point
    :param t: The point to split at
    :return: Control points of two resulting curves
    """
    lerp_points = get_lerp_points(a, b, c, d, t)
    return ((lerp_points[0][0], lerp_points[1][0], lerp_points[2][0], lerp_points[3][0]),
            (lerp_points[3][0], lerp_points[2][1], lerp_points[1][2], lerp_points[0][3]))


def get_lerp_points(a, b, c, d, t):
    """
    Get all resulting lerp points the curve for point t

    :param a: 1st control point
    :param b: 2nd control point
    :param c: 3rd control point
    :param d: 4th Control point
    :param t: The proportion along the curve to lerp to
    :return: 2D-array of resulting points
    """
    points = [[Vector2() for _ in range(4 - i)] for i in range(4)]
    points[0] = [a, b, c, d]
    for i in range(1, 4):
        for j in range(4 - i):
            points[i][j] = points[i - 1][j] * (1 - t) + points[i - 1][j + 1] * t
    return points


def get_curve_point(a, b, c, d, t):
    """
    Get the point q(t) of the curve

    :param a: 1st control point
    :param b: 2nd control point
    :param c: 3rd control point
    :param d: 4th Control point
    :param t: The proportion along the curve
    :return: The resulting point on the curve
    """
    return a * (1 - t) ** 3 + b * t * (1 - t) ** 2 * 3 + c * t ** 2 * (1 - t) * 3 + d * t ** 3


def get_distance_from_curve(point, a, b, c, d):
    """
    Estimate distance from the curve, and t of closest point

    :param point: Point to find distance from
    :param a: 1st control point
    :param b: 2nd control point
    :param c: 3rd control point
    :param d: 4th Control point
    :return: t, distance
    """

    # Find roots of (p-q(t))q'(t) = 0 and calculate which of these is closest to the point

    dir0 = (a - b).normalised()
    dir1 = (b - c).normalised()
    dir2 = (c - d).normalised()
    # Stop issues from all points lying on a straight line
    if dir0 == dir1 and dir1 == dir2:
        orthogonal = dir0.rotate_90()
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

    # Create sturm sequence to find how many roots are in a given interval
    sequence = SturmSequence(coefficient0, coefficient1, coefficient2, coefficient3, coefficient4, coefficient5)

    # Reduce intervals until the range of the root is less than the tolerance
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

    # Calculate which root is closest
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
    """
    Contains looped Bézier spline

    :param args: Points on the spline
    """
    def __init__(self, *args):
        self.control_points = list(args)
        self.num_points = len(self.control_points)
        self.num_segments = self.num_points // 3

    def get_segment_points(self, segment_index):
        """
        Get control points of a segment

        :param segment_index: The index of the segment
        :return: The control points
        """
        return tuple(self.get_segment_point(segment_index, i) for i in range(4))

    def get_segment_point(self, segment_index, point_index):
        """
        Get one of the segment's control points

        :param segment_index: The index of the segment
        :param point_index: The index of the control points
        :return: The control point
        """
        return self.get_control_point(segment_index * 3 + point_index)

    def get_control_point(self, point_index):
        """
        Get a control point on the spline

        :param point_index: The index of the control point
        :return: The control point
        """
        return self.control_points[point_index % self.num_points]

    def draw_lines(self, client, track_width):
        """
        Draw lines around the track limits in the debugger

        :param client: The client to draw in
        :param track_width: The width of the track
        """
        centre_points = [self.get_segment_point(0, 0)]
        for segment in range(self.num_segments):
            centre_points += split_curve_recursive(*tuple(self.get_segment_points(segment)))
        num_points = len(centre_points)
        left_points, right_points = [], []
        for i in range(num_points):
            mid_point = centre_points[i]
            direction = (centre_points[(i+1) % num_points] - centre_points[i-1]).normalised()
            offset = direction.rotate_90() * (track_width / 2)
            left_points.append(mid_point - offset)
            right_points.append(mid_point + offset)

        for points in [left_points, right_points]:
            previous_point = points[-1].make_3d(0.1)
            for point in points:
                current_point = point.make_3d(0.1)
                p.addUserDebugLine(previous_point.tuple(),
                                   current_point.tuple(),
                                   lineColorRGB=(1, 0, 0),
                                   lineWidth=2,
                                   physicsClientId=client)
                previous_point = current_point

    def get_curve_point(self, segment_index, t):
        """
        Get point along the curve

        :param segment_index: The segment to get the point on
        :param t: The proportion along the curve
        :return: The point on the curve
        """
        return get_curve_point(*self.get_segment_points(segment_index), t)

    def get_total_progress(self, segment_index, t):
        """
        Get how far along the curve (in metres) a point is

        :param segment_index: The index of the segment of the current point
        :param t: The proportion along the current segment of the point
        :return: The total distance from the start of the spline
        """
        total = 0
        for i in range(segment_index):
            total += self.get_segment_length(i)
        total += self.get_segment_length(segment_index) * self.get_segment_progress(segment_index, t)
        return total

    def get_segment_progress(self, segment_index, t):
        """
        Get how far along a segment (in metres) a point is

        :param segment_index: The index of the segment
        :param t: The proportion along the segment
        :return: The distance from the start of the segment
        """
        curve1, curve2 = split_at_point(*self.get_segment_points(segment_index), t)
        done = get_length(*curve1)
        remaining = get_length(*curve2)
        return done / (done + remaining)

    def get_total_length(self):
        """
        Get the total length of the whole spline

        :return: The total length of the spline
        """
        total = 0
        for i in range(self.num_segments):
            total += self.get_segment_length(i)
        return total

    def get_segment_length(self, segment_index):
        """
        Get the length of a segment

        :param segment_index: The index of the segment
        :return: The length of the segment
        """
        return get_length(*self.get_segment_points(segment_index))

    def get_distance_from_curve(self, point, segment_index):

        """
        Estimate distance from a segment, and t of the closest point

        :param point: Point to find distance from
        :param segment_index: The index of the segment
        :return: t, distance
        """
        return get_distance_from_curve(point, *self.get_segment_points(segment_index))

    def get_direction(self, segment_index, t):
        lerp_points = get_lerp_points(*self.get_segment_points(segment_index), t)
        return (lerp_points[2][1] - lerp_points[2][0]).normalised()

    def move_point(self, point_index, x_offset, y_offset):
        offset = Vector2(x_offset, y_offset)
        self.control_points[point_index] += offset
        # The start and end of each segment should have control points that form a straight line, so that direction of
        # curve is continuous
        if point_index % 3 == 0:
            self.control_points[(point_index - 1) % self.num_points] += offset
            self.control_points[(point_index + 1) % self.num_points] += offset
        else:
            if point_index % 3 == 1:
                mid_index = (point_index - 1) % self.num_points
                opp_index = (point_index - 2) % self.num_points
            else:
                mid_index = (point_index + 1) % self.num_points
                opp_index = (point_index + 2) % self.num_points

            cur_point = self.control_points[point_index]
            mid_point = self.control_points[mid_index]
            opp_point = self.control_points[opp_index]

            opp_dir = (mid_point - cur_point).normalised()
            opp_dist = (opp_point - mid_point).magnitude()
            self.control_points[opp_index] = mid_point + opp_dir * opp_dist

    def delete_point(self, point_index):
        if self.num_segments == 2:
            return
        if point_index % 3 == 1:
            point_index -= 1
        elif point_index % 3 == 2:
            point_index += 1
        if point_index > 0:
            self.control_points = self.control_points[:point_index-1] + self.control_points[point_index+2:]
        else:
            print("Tricky one")
            self.control_points = self.control_points[-3:-1] + self.control_points[2:-3]

        self.num_points -= 3
        self.num_segments -= 1

    def split_segment(self, point):
        closest_segment = 0
        min_dist = float('inf')
        closest_t = 0
        for segment_index in range(self.num_segments):
            t, dist = self.get_distance_from_curve(point, segment_index)
            if dist < min_dist:
                closest_segment = segment_index
                min_dist = dist
                closest_t = t
        new_curves = split_at_point(*self.get_segment_points(closest_segment), closest_t)
        self.control_points[closest_segment * 3 + 1] = new_curves[0][1]
        self.control_points[closest_segment * 3 + 2] = new_curves[0][2]
        self.control_points.insert(closest_segment * 3 + 3, new_curves[1][2])
        self.control_points.insert(closest_segment * 3 + 3, new_curves[1][1])
        self.control_points.insert(closest_segment * 3 + 3, new_curves[1][0])

        self.num_points += 3
        self.num_segments += 1

    def save(self, name):
        with open(Bezier.get_path(name), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(name):
        with open(Bezier.get_path(name), "rb") as file:
            return pickle.load(file)

    @staticmethod
    def list_saves():
        path = os.getcwd() + "\\" + SAVE_PATH
        files = os.listdir(path)
        return [file[:-len(EXTENSION)] for file in files if file[-len(EXTENSION):] == EXTENSION]

    @staticmethod
    def get_path(name):
        return SAVE_PATH + name + EXTENSION

    def mirror(self):
        for point_index in range(self.num_points):
            point = self.control_points[point_index]
            x, y = point.tuple()
            self.control_points[point_index] = Vector2(-x, y)

    def reverse(self):
        self.control_points = self.control_points[0:1] + self.control_points[-1:0:-1]


class SturmSequence:
    """
    Generate sequences required for solving a polynomial using Sturm's Theory

    :param args: Coefficients of the polynomial to solve
    """
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
        """
        Calculate sign changes in the sequence for given value

        :param t: Input to the polynomials
        :return: Number of sign changes
        """
        changes = 0
        previous = self._solve_polynomial(t, 0) >= 0
        for i in range(1, self.degree + 1):
            current = self._solve_polynomial(t, i) >= 0
            changes += current != previous
            previous = current
        return changes

    def _solve_polynomial(self, t, index):
        """
        Solve a polynomial occupying on column of the sequence

        :param t: Input of the polynomial
        :param index: Index of the sequence to solve
        :return: Result of the polynomial
        """
        total = 0
        for i in range(self.degree + 1):
            total += self.sequence[index][i] * t ** (self.degree - i)
        return total
