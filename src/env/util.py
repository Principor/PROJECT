import numpy as np
import pybullet as p


def make_vector(x, y, z):
    return np.array([x, y, z])


def make_quaternion(x, y, z, w=None):
    if w is None:
        return np.array(p.getQuaternionFromEuler([x, y, z]))
    else:
        return np.array([x, y, z, w])


def convert_to_numpy(transform):
    position, orientation = transform
    return np.array(position), np.array(orientation)


def convert_from_numpy(transform):
    position, orientation = transform
    return position.tolist(), orientation.tolist()


def get_transform(body):
    return convert_to_numpy(p.getBasePositionAndOrientation(body))


def get_velocity(body):
    return convert_to_numpy(p.getBaseVelocity(body))


def multiply_transforms(transform1, transform2):
    return convert_to_numpy(p.multiplyTransforms(*convert_from_numpy(transform1), *convert_from_numpy(transform2)))


def invert_transform(transform):
    return convert_to_numpy(p.invertTransform(*convert_from_numpy(transform)))


def transform_position(transform, position):
    new_position, _ = multiply_transforms(transform, (position, make_quaternion(0, 0, 0, 1)))
    return np.array(new_position)


def transform_direction(transform, direction):
    _, rotation = transform
    new_direction, _ = multiply_transforms((make_vector(0, 0, 0), rotation), (direction, make_quaternion(0, 0, 0, 1)))
    return np.array(new_direction)


def cross_vector(vector1, vector2):
    return np.cross(vector1, vector2)
