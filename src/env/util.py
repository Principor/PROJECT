import pybullet as p


def multiply_transforms(transform1, transform2):
    return p.multiplyTransforms(*transform1, *transform2)


def transform_position(transform, position):
    new_position, _ = multiply_transforms(transform, (position, [0, 0, 0, 1]))
    return new_position


def transform_direction(transform, direction):
    _, rotation = transform
    new_direction, _ = multiply_transforms(([0, 0, 0], rotation), (direction, [0, 0, 0, 1]))
    return new_direction


def scale_vector(vector, scale):
    return [x * scale for x in vector]


def add_vectors(vector1, vector2):
    return [x+y for x, y in zip(vector1, vector2)]