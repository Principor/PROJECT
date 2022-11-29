import numpy as np
import pybullet as p


class Vector2:
    def __init__(self, x=0, y=0):
        self.values = np.array([x, y])

    def __add__(self, other):
        return Vector2(*(self.values + other.values))

    def __sub__(self, other):
        return Vector2(*(self.values - other.values))

    def __mul__(self, other):
        if other is Vector2:
            return self.values * other.values
        return Vector2(*(self.values * other))

    def __truediv__(self, other):
        if other is Vector2:
            return self.values / other.values
        return Vector2(*(self.values / other))

    def __neg__(self):
        return Vector2(*-self.values)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(" + str(round(self.values[0], 2))\
               + ", " + str(round(self.values[1], 2)) + ")"

    def dot(self, other):
        return np.dot(self.values, other.values)

    def get_distance(self, other):
        return (self - other).magnitude()

    def magnitude(self):
        return np.linalg.norm(self.values)

    def normalised(self):
        magnitude = self.magnitude()
        if magnitude == 0:
            return np.zeros_like(self.values)
        return self / magnitude

    def rotate_90(self):
        x, y = self.values
        return Vector2(y, -x)

    def make_3d(self, z=0):
        return Vector3(*self.values, z)

    def tuple(self):
        return tuple(self.values)


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.values = np.array([x, y, z])

    def __add__(self, other):
        return Vector3(*(self.values + other.values))

    def __sub__(self, other):
        return Vector3(*(self.values - other.values))

    def __mul__(self, other):
        if other is Vector3:
            return self.values * other.values
        return Vector3(*(self.values * other))

    def __truediv__(self, other):
        if other is Vector3:
            return self.values / other.values
        return Vector3(*(self.values / other))

    def __neg__(self):
        return Vector3(*-self.values)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(" + str(round(self.values[0], 2))\
               + ", " + str(round(self.values[1], 2))\
               + ", " + str(round(self.values[2], 2)) + ")"

    def dot(self, other):
        return np.dot(self.values, other.values)

    def cross(self, other):
        return Vector3(*np.cross(self.values, other.values))

    def project(self, other):
        dot12 = np.dot(self.values, other.values)
        dot22 = np.dot(other.values, other.values)
        if dot22 == 0:
            return Vector3()
        return Vector3(*(other.values * dot12 / dot22))

    def normalised(self):
        norm = np.linalg.norm(self.values)
        if norm == 0:
            return np.zeros_like(self.values)
        return self / norm

    def project_to_plane(self, normal):
        return self - self.project(normal)

    def get_xy(self):
        return Vector2(self.values[0], self.values[1])

    def tuple(self):
        return tuple(self.values)
    

class Quaternion:
    def __init__(self, x=0, y=0, z=0, w=1):
        self.values = np.array([x, y, z, w])

    def tuple(self):
        return tuple(self.values)


class Transform:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

    def __mul__(self, other):
        position, orientation = p.multiplyTransforms(*self.tuples(), *other.tuples())
        return Transform(Vector3(*position), Quaternion(*orientation))

    def invert(self):
        position, orientation = p.invertTransform(*self.tuples())
        return Transform(Vector3(*position), Quaternion(*orientation))

    def transform_point(self, point):
        other = Transform(point, Quaternion())
        return (self * other).position

    def transform_direction(self, direction):
        this = Transform(Vector3(), self.orientation)
        other = Transform(direction, Quaternion())
        return (this * other).position

    def tuples(self):
        return self.position.tuple(), self.orientation.tuple()


def get_transform(body):
    position, orientation = p.getBasePositionAndOrientation(body)
    return Transform(Vector3(*position), Quaternion(*orientation))


def get_quaternion_from_euler(x, y, z):
    return Quaternion(*p.getQuaternionFromEuler((x, y, z)))
