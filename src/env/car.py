import pybullet as p

from env import util


class Car:
    def __init__(self):
        assert p.getConnectionInfo()['isConnected'], "No pybullet connection found"

        size = [1, 2, 0.5]

        body_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        body_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size)

        self.body = p.createMultiBody(baseMass=100,
                                      baseCollisionShapeIndex=body_collision_shape,
                                      baseVisualShapeIndex=body_visual_shape,
                                      basePosition=[0, 0, 5])

        self.front_axle = Axle(self, 1.5, 2.2, 1, 1000, 100)
        self.rear_axle = Axle(self, -1.5, 2.2, 1, 1000, 100)

    def update(self, dt):
        self.front_axle.update(dt)
        self.rear_axle.update(dt)

    def get_transform(self):
        return p.getBasePositionAndOrientation(self.body)

    def transform_direction(self, direction):
        _, rotation = p.getBasePositionAndOrientation(self.body)
        return util.rotate_vector(rotation, direction)

    def apply_force(self, position, force):
        p.applyExternalForce(self.body, linkIndex=-1, posObj=position, forceObj=force, flags=p.WORLD_FRAME)


class Axle:
    def __init__(self, car, axle_position, axle_width, spring_length, spring_stiffness, damper_stiffness):
        self.car = car

        self.right_wheel = Wheel(car, [axle_width / 2, axle_position, 0], spring_length, spring_stiffness,
                                 damper_stiffness)
        self.left_wheel = Wheel(car, [-axle_width / 2, axle_position, 0], spring_length, spring_stiffness,
                                damper_stiffness)

    def update(self, dt):
        for wheel in self.left_wheel, self.right_wheel:
            wheel.apply_suspension_force(dt)


class Wheel:
    def __init__(self, car, start_position, spring_length, spring_stiffness, damper_stiffness):
        self.car = car
        self.start_position = start_position
        self.spring_length = spring_length
        self.spring_stiffness = spring_stiffness
        self.damper_stiffness = damper_stiffness

        self.previous_length = 0

    def apply_suspension_force(self, dt):
        car_transform = self.car.get_transform()
        ray_start = util.transform_position(car_transform, self.start_position)
        ray_dir = util.transform_direction(car_transform, [0, 0, -1])
        ray_end = util.add_vectors(ray_start, util.scale_vector(ray_dir, self.spring_length))
        index, link, fraction, position, normal = p.rayTest(ray_start, ray_end)[0]

        current_length = fraction
        spring_force = (1 - current_length) * self.spring_stiffness
        damper_force = -(current_length - self.previous_length) * self.damper_stiffness / dt
        suspension_force = spring_force + damper_force

        self.car.apply_force(ray_start, util.scale_vector(ray_dir, -suspension_force))
        self.previous_length = current_length
