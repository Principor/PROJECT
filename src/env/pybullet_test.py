import time

import pybullet as p

TIME_STEP = 0.01

client = p.connect(p.GUI)
p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, -10)

SPHERE_RADIUS = 0.5
sphereCollisionId = p.createCollisionShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS)
sphereVisualId = p.createVisualShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS)
sphere = p.createMultiBody(baseMass=SPHERE_RADIUS ** 3,
                           baseCollisionShapeIndex=sphereCollisionId,
                           baseVisualShapeIndex=sphereVisualId,
                           basePosition=[0, 0, 2])

planeCollisionId = p.createCollisionShape(p.GEOM_PLANE)
planeVisualId = p.createVisualShape(p.GEOM_PLANE)
plane = p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=planeCollisionId,
                          baseVisualShapeIndex=planeVisualId)

p.changeDynamics(sphere, -1, restitution=0.9)
p.changeDynamics(plane,  -1, restitution=0.9)

while p.getConnectionInfo()['isConnected']:
    p.stepSimulation()
    time.sleep(TIME_STEP)
