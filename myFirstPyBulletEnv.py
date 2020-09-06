import pybullet as p
import time
import pybullet_data

useMaximalCoordinates = 0

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[5, 5, 0.1])
orn = p.getQuaternionFromEuler([0, 0, 0])
p.createMultiBody(0, planeId, baseOrientation=orn)

sphereRadius = 0.5
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

mass = 1
visualShapeId = -1

sphereUid = p.createMultiBody(
            mass,
            colSphereId,
            visualShapeId, [0, 0, 1],
            useMaximalCoordinates=useMaximalCoordinates)

boxUid = p.createMultiBody(
            mass,
            colBoxId,
            visualShapeId,
            [0, 2, 1],
            useMaximalCoordinates=useMaximalCoordinates)

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)
ballPositionZ = 1

# Run while the ball is on the plane
while (ballPositionZ > 0):
    ballPositionZ = p.getBasePositionAndOrientation(sphereUid)[0][2]
    p.applyExternalForce(objectUniqueId=sphereUid, linkIndex=-1, forceObj=(0, 10, 0), posObj=[0, 0, ballPositionZ], flags=p.WORLD_FRAME)
    p.stepSimulation()
    time.sleep(1 / 480)