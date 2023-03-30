import math

import numpy as np
import pybullet as p

from env.racecar_driving.resources import util


def get_long_slip(rolling_velocity, long_speed):
    """
    Calculate the longitudinal slip of a wheel

    :param rolling_velocity: The rolling velocity of the wheel (angular velocity x wheel radius)
    :param long_speed: The speed of the wheel in its longitudinal direction
    :return: The longitudinal slip ratio
    """
    return (rolling_velocity - long_speed) / abs(long_speed)


def get_lat_slip(lat_speed, long_speed):
    """
    Calculate the lateral slip of a wheel

    :param lat_speed: The speed of the wheel in its lateral direction
    :param long_speed: The speed of the wheel in its longitudinal direction
    :return: The lateral slip angle
    """
    return math.atan2(-lat_speed, long_speed)


def get_force_asymptote(F_z, B, C, D, E):
    """
    Calculate asymptote of Pacejka's formula, amount of force generated when slip tends to infinity

    :param F_z: The reaction force on the wheel
    :param B: Stiffness Factor
    :param C: Shape Factor
    :param D: Peak Factor
    :param E: Curvature Factor
    :return: The force generated for maximum slip
    """
    return F_z * D * math.sin(C)


def force_from_slip(F_z, B, C, D, E, slip):
    """
    Calculate force generated by a tyre for a specific amount of slip

    :param F_z: The reaction force on the wheel
    :param B: Stiffness Factor
    :param C: Shape Factor
    :param D: Peak Factor
    :param E: Curvature Factor
    :param slip: The slip of the tyre
    :return: The force generated by the tyre
    """
    return F_z * D * math.sin(C * math.atan(B * slip - E * (B * slip - math.atan(B * slip))))


class Car:
    """
    Car model using RayCast suspension and Pacejka tyre model

    :param client: The ID of the physics client to simulate the car in
    :param position: The position to create the car at
    :param orientation: The orientation to create  the car with
    :param mass: The mass of the car in kilograms
    :param wheelbase: The distance between the front and rear tyres in metres
    :param track_width: The distance between the left and right tyres in metres
    :param com_y: The offset of the center of mass along the y-axis in metres
    :param com_z: The offset of the center of mass along the z-axis in metres
    :param grip_factor: Multiplier for the amount of tyre grip
    """

    def __init__(self, client, position, orientation, mass=100, wheelbase=3, track_width=2.2, com_y=-0.3, com_z=-0.3,
                 grip_factor=1.0):
        self.client = client

        self.mass = mass
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.com_y = com_y
        self.com_z = com_z
        self.grip_factor = grip_factor

        half_extents = (1, 2, 0.5)
        shape_shift = (0, -com_y, -com_z)
        axle_height = -0.25
        spring_length = 1
        wheel_radius = 0.2
        wheel_mass = 10

        body_collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                                      halfExtents=half_extents,
                                                      collisionFramePosition=shape_shift)
        body_visual_shape = p.createVisualShape(p.GEOM_BOX,
                                                halfExtents=half_extents,
                                                visualFramePosition=shape_shift)

        self.body = p.createMultiBody(baseMass=mass,
                                      baseCollisionShapeIndex=body_collision_shape,
                                      baseVisualShapeIndex=body_visual_shape,
                                      basePosition=position,
                                      baseOrientation=orientation,
                                      physicsClientId=client)
        p.changeDynamics(self.body, -1, linearDamping=0.002)

        # Suspension parameters
        front_weight = mass * (0.5 + com_y / wheelbase)
        rear_weight = mass * (0.5 - com_y / wheelbase)
        spring_scalar = 5
        damper_scalar = 0.5
        rollbar_scalar = 10

        self.front_axle = Axle(self,
                               wheelbase / 2 - com_y,
                               track_width,
                               axle_height - com_z,
                               spring_length,
                               front_weight * spring_scalar,
                               front_weight * damper_scalar,
                               front_weight * rollbar_scalar,
                               wheel_radius,
                               wheel_mass,
                               self.grip_factor,
                               self.client)
        self.rear_axle = Axle(self,
                              -wheelbase / 2 - com_y,
                              track_width,
                              axle_height - com_z,
                              spring_length,
                              rear_weight * spring_scalar,
                              rear_weight * damper_scalar,
                              rear_weight * rollbar_scalar,
                              wheel_radius,
                              wheel_mass,
                               self.grip_factor,
                              self.client)

        self.horsepower = 50
        self.max_brake_torque = 150

    def update(self, throttle, steering, dt):
        """
        Set the throttle and steering input, and apply forces from the suspension and tyres

        :param throttle: The throttle/braking input
        :param steering: The steering input
        :param dt: The length of one time-step of the simulation
        """
        max_motor_torque = self.horsepower * 7127 / max(self.rear_axle.get_rpm(), 1000)

        steering_angle = np.clip(steering, -1, 1) * 0.5
        motor_torque = np.clip(throttle, 0, 1) * max_motor_torque
        brake_torque = np.clip(-throttle, 0, 1) * self.max_brake_torque

        self.front_axle.set_steering_angle(steering_angle)
        self.front_axle.set_motor_torque(motor_torque)

        self.front_axle.set_brake_torque(brake_torque)
        self.rear_axle.set_brake_torque(brake_torque)

        self.front_axle.update(dt)
        self.rear_axle.update(dt)

    def get_transform(self):
        """
        Get the transform of the car
        :return:
        """
        return util.get_transform(self.body)

    def apply_force(self, position, force):
        """
        Apply a force to the object at a position

        :param position: The position to apply the force at
        :param force: The force to be applied
        """
        p.applyExternalForce(self.body, linkIndex=-1, posObj=position.tuple(), forceObj=force.tuple(),
                             flags=p.WORLD_FRAME, physicsClientId=self.client)

    def remove(self):
        """
        Remove this car from the simulation
        """
        p.removeBody(self.body)

    def get_configuration(self):
        """
        Get the physical properties of the car

        :return: The offset of the center of mass along the z-axis in metres
        """
        return [self.com_y]

class Axle:
    """
    Car axle, containing two wheels

    :param car: The car the axle is attached to
    :param axle_position: How far along the y-coordinate that the axle is placed
    :param axle_width: How wide the axle is
    :param axle_height: How high the axle is - defines start of raycast, highest point the bottom of wheel can be
    :param spring_length: The maximum amount the spring can extend
    :param spring_stiffness: How much force the springs will exert to correct its length
    :param damper_stiffness: How much force the springs will exert to slow itself down
    :param rollbar_stiffness: How much both springs will exert to match each other's length
    :param wheel_radius: The radius of the wheels
    :param wheel_mass The mass of the wheels
    :param grip_factor: Multiplier for the amount of tyre grip
    :param client: The ID of the physics client that the car was added to
    """

    def __init__(self, car, axle_position, axle_width, axle_height, spring_length, spring_stiffness, damper_stiffness,
                 rollbar_stiffness, wheel_radius, wheel_mass, grip_factor, client):
        self.car = car

        self.right_wheel = Wheel(car, util.Vector3(axle_width / 2, axle_position, axle_height), spring_length,
                                 spring_stiffness, damper_stiffness, wheel_radius, wheel_mass, grip_factor, client)
        self.left_wheel = Wheel(car, util.Vector3(-axle_width / 2, axle_position, axle_height), spring_length,
                                spring_stiffness, damper_stiffness, wheel_radius, wheel_mass, grip_factor, client)
        self.wheels = [self.left_wheel, self.right_wheel]

        self.rollbar_stiffness = rollbar_stiffness

    def set_steering_angle(self, steering_angle):
        """
        Set the current steering angle to the wheels in the axle

        :param steering_angle: Angle of the wheels
        """
        for wheel in self.wheels:
            wheel.steering_angle = steering_angle

    def set_motor_torque(self, motor_torque):
        """
        Set the torque the axle is currently receiving from the engine

        :param motor_torque: Torque received by the engine
        """
        for wheel in self.wheels:
            wheel.motor_torque = motor_torque / 2

    def set_brake_torque(self, brake_torque):
        """
        Set the torque being generated by the brakes

        :param brake_torque: Torque generated by the brakes
        """
        for wheel in self.wheels:
            wheel.brake_torque = brake_torque

    def update(self, dt):
        """
        Update the suspension and tyres

        :param dt: The length of one time-step of the simulation
        """
        for wheel in self.wheels:
            wheel.update_ground_contact(dt)

        difference = self.left_wheel.spring_length - self.right_wheel.spring_length
        rollbar_force = difference * self.rollbar_stiffness
        self.left_wheel.rollbar_force = -rollbar_force
        self.right_wheel.rollbar_force = rollbar_force

        for wheel in self.wheels:
            wheel.apply_forces()

        for wheel in self.wheels:
            wheel.apply_torque(dt)

    def get_rpm(self):
        """
        Get the Rotations Per Minute of an axle (the average RPM of its two wheels)

        :return: The RPM of the axle
        """
        rads_per_second = self.left_wheel.angular_velocity + self.right_wheel.angular_velocity
        rpm = rads_per_second * 30 / math.pi
        return rpm


class Wheel:
    """
    Wheel of the car

    :param car: The car the wheel is attached to
    :param start_position: The position of the bottom of the wheel when the spring is fully depressed
    :param max_spring_length: The maximum amount the spring can compress
    :param spring_stiffness: How much force the spring will exert to correct its length
    :param damper_stiffness: How much force the spring will exert to slow itself down
    :param radius: The radius of the wheel
    :param mass: The mass of the wheel
    :param grip_factor: Multiplier for the amount of tyre grip
    :param client: The ID of the physics client that the car was added to
    """

    def __init__(self, car, start_position, max_spring_length, spring_stiffness, damper_stiffness, radius, mass,
                 grip_factor, client):
        self.car = car
        self.start_position = start_position
        self.max_spring_length = max_spring_length
        self.spring_stiffness = spring_stiffness
        self.damper_stiffness = damper_stiffness

        self.radius = radius
        self.mass = mass

        self.grip_factor = grip_factor

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
        self.velocity = self.contact_position = util.Vector3(0, 0, 0)
        self.contact_normal = util.Vector3(0, 0, 0)
        self.grounded = False

        self.client = client

    def update_ground_contact(self, dt):
        """
        Find the point where the bottom of the wheel touches the ground

        :param dt: The length of one time-step of the simulation
        """
        previous_position = self.contact_position
        previous_length = self.spring_length

        car_transform = self.car.get_transform()
        ray_start = car_transform.transform_point(self.start_position)
        ray_dir = car_transform.transform_direction(util.Vector3(0, 0, -1))
        ray_end = ray_start + ray_dir * self.max_spring_length
        _, _, fraction, position, normal = p.rayTest(ray_start.tuple(), ray_end.tuple(), physicsClientId=self.client)[0]

        self.contact_position = util.Vector3(*position)
        self.contact_normal = util.Vector3(*normal)
        self.spring_length = fraction * self.max_spring_length

        if self.grounded:
            self.velocity = (self.contact_position - previous_position) / dt
            self.spring_speed = (self.spring_length - previous_length) / dt
        else:
            self.velocity = util.Vector3(0, 0, 0)
            self.spring_speed = 0
        self.grounded = fraction < 1

    def apply_forces(self):
        """
        Apply the forces generated by the suspension and the tyres on the car
        """
        if not self.grounded:
            return

        car_transform = self.car.get_transform()

        spring_force = (self.max_spring_length - self.spring_length) * self.spring_stiffness
        damper_force = -self.spring_speed * self.damper_stiffness
        reaction_force = spring_force + damper_force + self.rollbar_force
        reaction_dir = self.car.get_transform().transform_direction(util.Vector3(0, 0, 1))

        wheel_rotation = util.get_quaternion_from_euler(0, 0, self.steering_angle)
        wheel_local_transform = util.Transform(util.Vector3(), wheel_rotation)
        wheel_transform = car_transform * wheel_local_transform

        long_dir = wheel_transform.transform_direction(util.Vector3(0, 1, 0))
        long_dir = long_dir.project_to_plane(self.contact_normal)
        lat_dir = wheel_transform.transform_direction(util.Vector3(1, 0, 0))
        lat_dir = lat_dir.project_to_plane(self.contact_normal)

        long_force, lat_force = self._get_tyre_forces(long_dir, lat_dir, reaction_force)

        final_force = reaction_dir * reaction_force + long_dir * long_force + lat_dir * lat_force

        self.car.apply_force(self.contact_position, final_force)

        self.traction_torque = -long_force * self.radius

    def apply_torque(self, dt):
        """
        Apply the torque generated by the traction, motor and brakes

        :param dt: The length of one time-step of the simulation
        """
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
        # Calculate the forces from its current orientation and reaction force
        long_speed = self.velocity.dot(long_dir)
        lat_speed = self.velocity.dot(lat_dir)

        self.target_angular_velocity = long_speed / self.radius

        rolling_velocity = (self.angular_velocity + self.angular_velocity_prev) / 2 * self.radius

        if long_speed > 0.5:
            long_slip = get_long_slip(rolling_velocity, long_speed)
            long_force = force_from_slip(reaction_force, 10, 1.6, 2, 0, long_slip)

            lat_slip = math.atan2(-lat_speed, long_speed)
            lat_force = force_from_slip(reaction_force, 10, 1.3, 2, 0, lat_slip)
        else:
            max_long_force = get_force_asymptote(reaction_force, 10, 1.6, 2, 0)
            long_force = (rolling_velocity - long_speed) * (reaction_force / 9.8) * 100
            long_force = np.clip(long_force, -max_long_force, max_long_force)

            max_lat_force = get_force_asymptote(reaction_force, 10, 1.3, 2, 0)
            lat_force = -lat_speed * (reaction_force / 9.8) * 100
            lat_force = np.clip(lat_force, -max_lat_force, max_lat_force)

        return long_force * self.grip_factor, lat_force * self.grip_factor
