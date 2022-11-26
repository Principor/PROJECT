import math

import numpy as np
import pybullet as p

from env.racecar_driving.resources import util


def get_long_slip(rolling_velocity, long_speed):
    return (rolling_velocity - long_speed) / abs(long_speed)


def get_lat_slip(lat_speed, long_speed):
    return math.atan2(-lat_speed, long_speed)


def force_from_slip(F_z, B, C, D, E, slip):
    return F_z * D * math.sin(C * math.atan(B * slip - E * (B * slip - math.atan(B * slip))))


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

        self.front_axle = Axle(self, 1.5, 2.2, -0.25, 1, 500, 50, 0.2, 10, self.client)
        self.rear_axle = Axle(self, -1.5, 2.2, -0.25, 1, 500, 50, 0.2, 10, self.client)

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
            wheel.update_ground_contact(dt)

        difference = self.left_wheel.spring_length - self.right_wheel.spring_length
        rollbar_force = difference * 500
        self.left_wheel.rollbar_force = -rollbar_force
        self.right_wheel.rollbar_force = rollbar_force

        for wheel in self.wheels:
            wheel.apply_forces()

        for wheel in self.wheels:
            wheel.apply_torque(dt)

    def get_rpm(self):
        rads_per_second = self.left_wheel.angular_velocity + self.right_wheel.angular_velocity
        rpm = rads_per_second * 30 / math.pi
        return rpm


class Wheel:
    def __init__(self, car, start_position, max_spring_length, spring_stiffness, damper_stiffness, radius, mass, client):
        self.car = car
        self.start_position = start_position
        self.max_spring_length = max_spring_length
        self.spring_stiffness = spring_stiffness
        self.damper_stiffness = damper_stiffness

        self.radius = radius
        self.mass = mass

        self.angular_velocity_prev = 0
        self.angular_velocity = 0

        self.target_angular_velocity = 0

        self.traction_torque = 0
        self.motor_torque = 0
        self.brake_torque = 0
        self.steering_angle = 0

        self.rollbar_force = 0
        self.spring_length = 0
        self.spring_speed = 0
        self.velocity = self.contact_position = util.make_vector(0, 0, 0)
        self.contact_normal = util.make_vector(0, 0, 0)
        self.grounded = False

        self.client = client

    def update_ground_contact(self, dt):
        previous_position = self.contact_position
        previous_length = self.spring_length

        car_transform = self.car.get_transform()
        ray_start = util.transform_position(car_transform, self.start_position)
        ray_dir = util.transform_direction(car_transform, util.make_vector(0, 0, -1))
        ray_end = ray_start + ray_dir * self.max_spring_length
        _, _, fraction, position, normal = p.rayTest(ray_start, ray_end, physicsClientId=self.client)[0]

        self.contact_position = util.make_vector(*position)
        self.contact_normal = util.make_vector(*normal)
        self.spring_length = fraction * self.max_spring_length

        if self.grounded:
            self.velocity = (self.contact_position - previous_position) / dt
            self.spring_speed = (self.spring_length - previous_length) / dt
        self.grounded = fraction < 1

    def apply_forces(self):
        if not self.grounded:
            return

        spring_force = (self.max_spring_length - self.spring_length) * self.spring_stiffness
        damper_force = -self.spring_speed * self.damper_stiffness

        reaction_force = spring_force + damper_force + self.rollbar_force

        wheel_rotation = util.make_quaternion(0, 0, self.steering_angle)
        wheel_transform = util.multiply_transforms(self.car.get_transform(),
                                                   (util.make_vector(0, 0, 0), wheel_rotation))

        long_dir = util.transform_direction(wheel_transform, util.make_vector(0, 1, 0))
        long_dir = util.project_to_plane(long_dir, self.contact_normal)
        lat_dir = util.transform_direction(wheel_transform, util.make_vector(1, 0, 0))
        lat_dir = util.project_to_plane(lat_dir, self.contact_normal)

        long_force, lat_force = self._get_tyre_forces(long_dir, lat_dir, reaction_force)

        final_force = long_force * long_dir + lat_force * lat_dir + reaction_force * self.contact_normal

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

    def _get_tyre_forces(self, long_dir, lat_dir, reaction_force):
        long_speed = np.dot(self.velocity, long_dir)
        lat_speed = np.dot(self.velocity, lat_dir)

        self.target_angular_velocity = long_speed / self.radius

        rolling_velocity = (self.angular_velocity + self.angular_velocity_prev) / 2 * self.radius

        if long_speed > 0.5:
            long_slip = get_long_slip(rolling_velocity, long_speed)
            long_force = force_from_slip(reaction_force, 10, 1.6, 2, 0, long_slip)

            lat_slip = math.atan2(-lat_speed, long_speed)
            lat_force = force_from_slip(reaction_force, 10, 1.3, 2, 0, lat_slip)
        else:
            long_force = (rolling_velocity-long_speed) * (reaction_force / 9.8)
            lat_force = np.clip(-lat_speed, -0.5, 0.5) * (reaction_force / 9.8)

        return long_force, lat_force
