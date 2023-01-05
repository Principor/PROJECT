import numpy as np
import pybullet as p


class Vector2:
    """
    Representation of 2D vectors

    :param x: x-value of vector
    :param y: y-value of vector
    """
    def __init__(self, x=0, y=0):
        self.values = np.array([x, y])

    def __add__(self, other):
        """
        Add two vectors

        :param other: Vector to be added
        :return: Sum of both vectors
        """
        return Vector2(*(self.values + other.values))

    def __sub__(self, other):
        """
        Subtract one vector from another

        :param other: Vector to be subtracted
        :return: The difference
        """
        return Vector2(*(self.values - other.values))

    def __mul__(self, other):
        """
        Multiply a vector

        :param other:  Either number of Vector2 to multiply by
        :return: The product
        """
        if other is Vector2:
            return self.values * other.values
        return Vector2(*(self.values * other))

    def __truediv__(self, other):
        """
        Divide a vector

        :param other: Either number of Vector2 to divided by
        :return: The division result
        """
        if other is Vector2:
            return self.values / other.values
        return Vector2(*(self.values / other))

    def __neg__(self):
        """
        Negate the vector

        :return: Negated vector
        """
        return Vector2(*-self.values)

    def __repr__(self):
        """
        Get printable representation

        :return: String of self
        """
        return str(self)

    def __str__(self):
        """
        Convert to string of the form "(x, y)"

        :return: String of self
        """
        return "(" + str(round(self.values[0], 2))\
               + ", " + str(round(self.values[1], 2)) + ")"

    def dot(self, other):
        """
        Perform dot product with another vector

        :param other: The other vector to perform dot product with
        :return: The dot product
        """
        return np.dot(self.values, other.values)

    def get_distance(self, other):
        """
        Get distance between two points

        :param other: The point to find the distance to
        :return: The distance between the points
        """
        return (self - other).magnitude()

    def magnitude(self):
        """
        Find the magnitude of a vector

        :return: The magnitude of the vector
        """
        return np.linalg.norm(self.values)

    def normalised(self):
        """
        Find normalised vector

        :return: Normalised vector if magnitude > 0, else (0, 0)
        """
        magnitude = self.magnitude()
        if magnitude == 0:
            return np.zeros_like(self.values)
        return self / magnitude

    def rotate_90(self):
        """
        Rotate vector by 90 degrees

        :return: The rotated vector
        """

        x, y = self.values
        return Vector2(y, -x)

    def make_3d(self, z=0):
        """
        Convert a Vector2 to a Vector3

        :param z: z-coordinate of Vector3
        :return: The Vector3
        """
        return Vector3(*self.values, z)

    def tuple(self):
        """
        Convert to tuple

        :return: Tuple of self
        """
        return tuple(self.values)


class Vector3:
    """
    Representation of 3D vectors

    :param x: x-value of vector
    :param y: y-value of vector
    :param z: z-value of vector
    """
    def __init__(self, x=0, y=0, z=0):
        self.values = np.array([x, y, z])

    def __add__(self, other):
        """
        Add two vectors

        :param other: Vector to be added
        :return: Sum of both vectors
        """
        return Vector3(*(self.values + other.values))

    def __sub__(self, other):
        """
        Subtract one vector from another

        :param other: Vector to be subtracted
        :return: The difference
        """
        return Vector3(*(self.values - other.values))

    def __mul__(self, other):
        """
        Multiply a vector

        :param other:  Either number of Vector3 to multiply by
        :return: The product
        """
        if other is Vector3:
            return self.values * other.values
        return Vector3(*(self.values * other))

    def __truediv__(self, other):
        """
        Divide a vector

        :param other: Either number of Vector3 to divided by
        :return: The division result
        """
        if other is Vector3:
            return self.values / other.values
        return Vector3(*(self.values / other))

    def __neg__(self):
        """
        Negate the vector

        :return: Negated vector
        """
        return Vector3(*-self.values)

    def __repr__(self):
        """
        Get printable representation

        :return: String of self
        """
        return str(self)

    def __str__(self):
        """
        Convert to string of the form "(x, y, z)"

        :return: String of self
        """
        return "(" + str(round(self.values[0], 2))\
               + ", " + str(round(self.values[1], 2))\
               + ", " + str(round(self.values[2], 2)) + ")"

    def dot(self, other):
        """
        Perform dot product with another vector

        :param other: The other vector to perform dot product with
        :return: The dot product
        """
        return np.dot(self.values, other.values)

    def cross(self, other):
        """
        Perform cross product with another vector

        :param other: The other vector to perform cross product with
        :return: The cross product
        """
        return Vector3(*np.cross(self.values, other.values))

    def project(self, other):
        """
        Project a vector onto another

        :param other: Vector to be projected onto
        :return: The projected component
        """

        dot12 = np.dot(self.values, other.values)
        dot22 = np.dot(other.values, other.values)
        if dot22 == 0:
            return Vector3()
        return Vector3(*(other.values * dot12 / dot22))

    def magnitude(self):
        """
        Find the magnitude of a vector

        :return: The magnitude of the vector
        """
        return np.linalg.norm(self.values)

    def normalised(self):
        """
        Find normalised vector

        :return: Normalised vector if magnitude > 0, else (0, 0)
        """
        magnitude = self.magnitude()
        if magnitude == 0:
            return np.zeros_like(self.values)
        return self / magnitude

    def project_to_plane(self, normal):
        """
        Project a vector a plane

        :param normal: The normal of the plane
        :return: The projected component
        """
        return self - self.project(normal)

    def get_xy(self):
        """
        Convert a Vector3 into a Vector2, discarding the z-component
        
        :return: The Vector2
        """
        return Vector2(self.values[0], self.values[1])

    def tuple(self):
        """
        Convert to tuple

        :return: Tuple of self
        """
        return tuple(self.values)
    

class Quaternion:
    """
    Quaternion, representing rotation
    
    :param x: x-value of vector
    :param y: y-value of vector
    :param z: z-value of vector
    :param w: w-value of vector
    """
    def __init__(self, x=0, y=0, z=0, w=1):
        self.values = np.array([x, y, z, w])

    def tuple(self):
        """
        Convert to tuple

        :return: Tuple of self
        """
        return tuple(self.values)


class Transform:
    """
    Transform of an object, consisting of the position and orientation
    
    :param position: Relative position of the transform
    :param orientation: Relative orientation of the transform
    """
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

    def __mul__(self, other):
        """
        Multiply two transforms
        
        :param other: Other transform to multiply by
        :return: The multiplied transform
        """
        position, orientation = p.multiplyTransforms(*self.tuples(), *other.tuples())
        return Transform(Vector3(*position), Quaternion(*orientation))

    def invert(self):
        """
        Invert a transform
        
        :return: The inverted transform 
        """
        position, orientation = p.invertTransform(*self.tuples())
        return Transform(Vector3(*position), Quaternion(*orientation))

    def transform_point(self, point):
        """
        Transform a point from local space in the transform to world space
        
        :param point: The point in local space
        :return: The point in world space
        """
        other = Transform(point, Quaternion())
        return (self * other).position

    def transform_direction(self, direction):
        """
        Transform a direction from local space in the transform to world space
        
        :param direction: The direction in local space
        :return: The direction in world space
        """
        this = Transform(Vector3(), self.orientation)
        other = Transform(direction, Quaternion())
        return (this * other).position

    def tuples(self):
        """
        Convert to tuple

        :return: Tuple of self
        """
        return self.position.tuple(), self.orientation.tuple()


def get_transform(body):
    """
    Get the transform of a body

    :param body: The body to find the transform of
    :return: The transform of the body
    """
    position, orientation = p.getBasePositionAndOrientation(body)
    return Transform(Vector3(*position), Quaternion(*orientation))


def get_quaternion_from_euler(x, y, z):
    """
    Convert an euler angle to a quaternion

    :param x: Rotation around x-axis
    :param y: Rotation around y-axis
    :param z: Rotation around z-axis
    :return: Quaternion of the rotation
    """
    return Quaternion(*p.getQuaternionFromEuler((x, y, z)))
