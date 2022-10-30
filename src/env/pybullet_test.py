import time

import pybullet as p

from src.env.car import Car

TIME_STEP = 0.01

client = p.connect(p.GUI)
p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, -10)

car = Car()

planeCollisionId = p.createCollisionShape(p.GEOM_PLANE)
planeVisualId = p.createVisualShape(p.GEOM_PLANE)
ground = p.createMultiBody(baseMass=0,
                           baseCollisionShapeIndex=planeCollisionId,
                           baseVisualShapeIndex=planeVisualId)

p.changeDynamics(ground, -1, restitution=0.9)

while p.getConnectionInfo()['isConnected']:
    p.stepSimulation()
    time.sleep(TIME_STEP)
