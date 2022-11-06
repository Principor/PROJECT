import time

import pybullet as p

from env.car import Car

TIME_STEP = 0.01

client = p.connect(p.GUI)
p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, -10)

car = Car()

plane_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
plane_visual_shape = p.createVisualShape(p.GEOM_PLANE)
ground = p.createMultiBody(baseMass=0,
                           baseCollisionShapeIndex=plane_collision_shape,
                           baseVisualShapeIndex=plane_visual_shape)

p.changeDynamics(ground, -1,
                 restitution=0.9)
throttle_param = p.addUserDebugParameter("Throttle", -1, 1, 0)
steering_param = p.addUserDebugParameter("Steering", -1, 1, 0)

while p.getConnectionInfo()['isConnected']:
    p.stepSimulation()
    throttle = p.readUserDebugParameter(throttle_param)
    steering = p.readUserDebugParameter(steering_param)
    car.update(throttle, steering, TIME_STEP)
    time.sleep(TIME_STEP)
