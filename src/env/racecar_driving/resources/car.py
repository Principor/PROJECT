import math

import numpy as np
import pybullet as p

from env.racecar_driving.resources import util


class Car:
    def __init__(self, client, position, orientation):
        self.client = client

        size = [1, 2, 0.5]

        body_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        body_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size)

        self.body = p.createMultiBody(baseMass=100,
                                      baseCollisionShapeIndex=body_collision_shape,
                                      baseVisualShapeIndex=body_visual_shape,
                                      basePosition=position,
                                      baseOrientation=orientation,
                                      physicsClientId=client)

        self.front_axle = Axle(self, 1.5, 2.2, -0.25, 1, 1000, 200, 0.2, 10, self.client)
        self.rear_axle = Axle(self, -1.5, 2.2, -0.25, 1, 1000, 200, 0.2, 10, self.client)

        self.horsepower = 50
        self.max_brake_torque = 300

    def update(self, throttle, steering, dt):
        max_motor_torque = self.horsepower * 7127 / max(self.rear_axle.get_rpm(), 1000)

        steering_angle = np.clip(steering, -1, 1) * 0.5
        motor_torque = np.clip(throttle, 0, 1) * max_motor_torque
        brake_torque = np.clip(-throttle, 0, 1) * self.max_brake_torque

        self.front_axle.set_steering_angle(steering_angle)
        self.front_axle.set_brake_torque(brake_torque)
        self.front_axle.set_motor_torque(motor_torque)

        self.front_axle.update(dt)
        self.rear_axle.update(dt)

    def get_transform(self):
        return util.get_transform(self.body)

    def apply_force(self, position, force):
        p.applyExternalForce(self.body, linkIndex=-1, posObj=position, forceObj=force, flags=p.WORLD_FRAME,
                             physicsClientId=self.client)

    def remove(self):
        p.removeBody(self.body)


class Axle:
    def __init__(self, car, axle_position, axle_width, axle_height, spring_length, spring_stiffness, damper_stiffness,
                 wheel_radius, wheel_mass, client):
        self.car = car

        self.right_wheel = Wheel(car, util.make_vector(axle_width / 2, axle_position, axle_height), spring_length,
                                 spring_stiffness, damper_stiffness, wheel_radius, wheel_mass, client)
        self.left_wheel = Wheel(car, util.make_vector(-axle_width / 2, axle_position, axle_height), spring_length,
                                spring_stiffness, damper_stiffness, wheel_radius, wheel_mass, client)
        self.wheels = [self.left_wheel, self.right_wheel]

    def set_steering_angle(self, steering_angle):
        for wheel in self.wheels:
            wheel.steering_angle = steering_angle

    def set_motor_torque(self, motor_torque):
        for wheel in self.wheels:
            wheel.motor_torque = motor_torque / 2

    def set_brake_torque(self, brake_torque):
        for wheel in self.wheels:
            wheel.brake_torque = brake_torque

    def update(self, dt):
        for wheel in self.wheels:
            wheel.apply_suspension_force(dt)

        for wheel in self.wheels:
            wheel.apply_tire_force(dt)

        for wheel in self.wheels:
            wheel.apply_torque(dt)

    def get_rpm(self):
        rads_per_second = self.left_wheel.angular_velocity + self.right_wheel.angular_velocity
        rpm = rads_per_second * 30 / math.pi
        return rpm


class Wheel:
    def __init__(self, car, start_position, spring_length, spring_stiffness, damper_stiffness, radius, mass, client):
        self.car = car
        self.start_position = start_position
        self.spring_length = spring_length
        self.spring_stiffness = spring_stiffness
        self.damper_stiffness = damper_stiffness

        self.radius = radius
        self.mass = mass

        self.angular_velocity_prev = 0
        self.angular_velocity = 0

        self.target_angular_velocity = 0

        self.traction_torque = 0
        self.motor_torque = 0
        self.steering_angle = 0
        self.brake_torque = 0

        self.previous_length = 0
        self.reaction_force = 0
        self.previous_position = self.contact_position = util.make_vector(0, 0, 0)
        self.contact_normal = util.make_vector(0, 0, 0)

        self.client = client

    def apply_suspension_force(self, dt):
        self.previous_position = self.contact_position

        car_transform = self.car.get_transform()
        ray_start = util.transform_position(car_transform, self.start_position)
        ray_dir = util.transform_direction(car_transform, util.make_vector(0, 0, -1))
        ray_end = ray_start + ray_dir * self.spring_length
        index, link, fraction, position, normal = p.rayTest(ray_start, ray_end, physicsClientId=self.client)[0]
        self.contact_position = util.make_vector(*position)
        self.contact_normal = util.make_vector(*normal)

        current_length = fraction
        spring_force = (1 - current_length) * self.spring_stiffness
        damper_force = -(current_length - self.previous_length) * self.damper_stiffness / dt
        suspension_force = spring_force + damper_force

        self.car.apply_force(ray_start, self.contact_normal * suspension_force)
        self.previous_length = current_length

        self.reaction_force = suspension_force if fraction < 1 else 0

    def apply_tire_force(self, dt):
        velocity = (self.contact_position - self.previous_position) / dt

        wheel_rotation = util.make_quaternion(0, 0, self.steering_angle)
        wheel_transform = util.multiply_transforms(self.car.get_transform(),
                                                   (util.make_vector(0, 0, 0), wheel_rotation))

        long_dir = util.transform_direction(wheel_transform, util.make_vector(0, 1, 0))
        long_dir = util.project_to_plane(long_dir, self.contact_normal)
        lat_dir = util.transform_direction(wheel_transform, util.make_vector(1, 0, 0))
        lat_dir = util.project_to_plane(lat_dir, self.contact_normal)

        long_speed = np.dot(velocity, long_dir)
        lat_speed = np.dot(velocity, lat_dir)

        self.target_angular_velocity = long_speed / self.radius

        rolling_velocity = (self.angular_velocity + self.angular_velocity_prev) / 2 * self.radius

        if long_speed > 0.5:
            long_slip = (rolling_velocity - long_speed) / abs(long_speed)
            long_force = self._force_from_slip(10, 1.6, 2, 0, long_slip)

            lat_slip = math.atan2(-lat_speed, long_speed)
            lat_force = self._force_from_slip(10, 1.3, 2, 0, lat_slip)
        else:
            long_force = (rolling_velocity - long_speed) * (self.reaction_force / 9.8)
            lat_force = -np.clip(lat_speed, -0.5, 0.5) * (self.reaction_force / 9.8)

        final_force = long_force * long_dir + lat_force * lat_dir

        self.car.apply_force(self.contact_position, final_force)

        self.traction_torque = -long_force * self.radius

    def apply_torque(self, dt):
        self.angular_velocity_prev = self.angular_velocity
        inertia = self.mass * self.radius ** 2 / 2

        if self.target_angular_velocity > self.angular_velocity:
            self.angular_velocity = min(
                self.angular_velocity + self.traction_torque / inertia * dt,
                self.target_angular_velocity)
        else:
            self.angular_velocity = max(
                self.angular_velocity + self.traction_torque / inertia * dt,
                self.target_angular_velocity)

        self.angular_velocity += self.motor_torque / inertia * dt

        brake_deceleration = self.brake_torque / inertia * dt
        self.angular_velocity -= np.clip(self.angular_velocity, -brake_deceleration, brake_deceleration)

    def _force_from_slip(self, B, C, D, E, slip):
        return self.reaction_force * D * math.sin(C * math.atan(B * slip - E * (B * slip - math.atan(B * slip))))
