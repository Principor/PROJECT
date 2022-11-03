import math

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

        self.front_axle = Axle(self, 1.5, 2.2, 1, 1000, 100, 0.2, 10)
        self.rear_axle = Axle(self, -1.5, 2.2, 1, 1000, 100, 0.2, 10)

    def update(self, dt):
        self.front_axle.update(dt)
        self.rear_axle.update(dt)

    def get_transform(self):
        return util.get_transform(self.body)

    def get_velocity(self):
        return util.get_velocity(self.body)

    def get_velocity_at_point(self, point):
        velocity, ang_velocity = self.get_velocity()
        return velocity + util.cross_vector(ang_velocity, point)

    def apply_force(self, position, force):
        p.applyExternalForce(self.body, linkIndex=-1, posObj=position, forceObj=force, flags=p.WORLD_FRAME)

    def set_wheel_speed(self, wheel_speed):
        for wheel in self.rear_axle.wheels + self.front_axle.wheels:
            wheel.angular_velocity = wheel_speed * 20

    def set_wheel_angle(self, wheel_angle):
        for wheel in self.front_axle.wheels:
            wheel.steering_angle = wheel_angle * 0.5


class Axle:
    def __init__(self, car, axle_position, axle_width, spring_length, spring_stiffness, damper_stiffness, wheel_radius,
                 wheel_mass):
        self.car = car

        self.right_wheel = Wheel(car, util.make_vector(axle_width / 2, axle_position, 0), spring_length,
                                 spring_stiffness, damper_stiffness, wheel_radius, wheel_mass)
        self.left_wheel = Wheel(car, util.make_vector(-axle_width / 2, axle_position, 0), spring_length,
                                spring_stiffness, damper_stiffness, wheel_radius, wheel_mass)
        self.wheels = [self.left_wheel, self.right_wheel]

    def update(self, dt):
        for wheel in self.wheels:
            wheel.apply_suspension_force(dt)

        for wheel in self.wheels:
            wheel.apply_tire_force()


class Wheel:
    def __init__(self, car, start_position, spring_length, spring_stiffness, damper_stiffness, radius, mass):
        self.car = car
        self.start_position = start_position
        self.spring_length = spring_length
        self.spring_stiffness = spring_stiffness
        self.damper_stiffness = damper_stiffness

        self.radius = radius
        self.mass = mass

        self.angular_velocity = 0
        self.steering_angle = 0

        self.previous_length = 0
        self.reaction_force = 0
        self.contact_position = util.make_vector(0, 0, 0)

    def apply_suspension_force(self, dt):
        car_transform = self.car.get_transform()
        ray_start = util.transform_position(car_transform, self.start_position)
        ray_dir = util.transform_direction(car_transform, util.make_vector(0, 0, -1))
        ray_end = ray_start + ray_dir * self.spring_length
        index, link, fraction, position, normal = p.rayTest(ray_start, ray_end)[0]
        self.contact_position = position

        current_length = fraction
        spring_force = (1 - current_length) * self.spring_stiffness
        damper_force = -(current_length - self.previous_length) * self.damper_stiffness / dt
        suspension_force = spring_force + damper_force

        self.car.apply_force(ray_start, ray_dir * -suspension_force)
        self.previous_length = current_length

        self.reaction_force = suspension_force if fraction < 1 else 0

    def apply_tire_force(self):
        world_velocity = self.car.get_velocity_at_point(self.contact_position)
        wheel_rotation = util.make_quaternion(0, 0, self.steering_angle)
        wheel_transform = util.multiply_transforms(self.car.get_transform(),
                                                   (util.make_vector(0, 0, 0), wheel_rotation))
        local_velocity = util.transform_direction(util.invert_transform(wheel_transform), world_velocity)
        lat_velocity = local_velocity[0]
        long_velocity = local_velocity[1]
        rolling_velocity = self.angular_velocity * self.radius
        lat_slip = atan(-lat_velocity / abs(long_velocity + 1e-8))
        if rolling_velocity != 0 or long_velocity != 0:
            long_slip = (rolling_velocity - long_velocity) / max(abs(rolling_velocity), abs(long_velocity)) * 100
        else:
            long_slip = 0
        lat_force = self.force_from_slip(lat_slip)
        long_force = self.force_from_slip(long_slip)
        final_force = util.transform_direction(wheel_transform, util.make_vector(lat_force, long_force, 0))
        self.car.apply_force(self.contact_position, final_force)

        print(local_velocity, lat_slip, lat_force)

    def force_from_slip(self, slip):
        B = 10
        C = 1.9
        D = 1
        E = 0.97
        return self.reaction_force * D * sin(C * atan(B * slip - E * (B * slip - atan(B * slip))))


def sin(x):
    return math.sin(math.radians(x))


def atan(x):
    return math.degrees(math.atan(x))
