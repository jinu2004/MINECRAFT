from math import pi, sin, cos
import cv2
import time
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile
from panda3d.core import DirectionalLight, AmbientLight
from panda3d.core import TransparencyAttrib
from panda3d.core import WindowProperties
from panda3d.core import (
    CollisionTraverser,
    CollisionNode,
    CollisionBox,
    CollisionRay,
    CollisionHandlerQueue,
)
from panda3d.core import LVecBase3f
from direct.gui.OnscreenImage import OnscreenImage

from FaceMesh import FaceMesh


loadPrcFile("settings.prc")


def degToRad(degrees):
    return degrees * (pi / 180.0)


class MyGame(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        # check line 83 if the rotation is not good
        self.selectedBlockType = "grass"
        self.cap = cv2.VideoCapture(0)
        self.movment = "Forward"

        self.faceMesh = FaceMesh()
        self.remove_counter = 0
        self.plcae_counter = 0

        self.rotate_speed = 15
        self.rotate_speedP = 15

        self.loadModels()
        self.setupLights()
        self.generateTerrain()
        self.setupCamera()
        self.setupSkybox()
        self.captureMouse()
        self.setupControls()

        self.taskMgr.add(self.update, "update")

    def update(self, task):
        dt = globalClock.getDt()

        playerMoveSpeed = 1

        x_movement = 0
        y_movement = 0
        z_movement = 0

        self.faceMesh.FaceDetector(self.cap)

        if self.keyMap["forward"]:
            x_movement -= dt * playerMoveSpeed * sin(degToRad(self.camera.getH()))
            y_movement += dt * playerMoveSpeed * cos(degToRad(self.camera.getH()))
        if self.keyMap["backward"]:
            x_movement += dt * playerMoveSpeed * sin(degToRad(self.camera.getH()))
            y_movement -= dt * playerMoveSpeed * cos(degToRad(self.camera.getH()))
        if self.keyMap["left"]:
            x_movement -= dt * playerMoveSpeed * cos(degToRad(self.camera.getH()))
            y_movement -= dt * playerMoveSpeed * sin(degToRad(self.camera.getH()))
        if self.keyMap["right"]:
            x_movement += dt * playerMoveSpeed * cos(degToRad(self.camera.getH()))
            y_movement += dt * playerMoveSpeed * sin(degToRad(self.camera.getH()))
        if self.keyMap["up"]:
            z_movement += dt * playerMoveSpeed
        if self.keyMap["down"]:
            z_movement -= dt * playerMoveSpeed

        self.camera.setPos(
            self.camera.getX() + x_movement,
            self.camera.getY() + y_movement,
            self.camera.getZ() + z_movement,
        )

        x, y, z = self.faceMesh.FaceLandmarkesPos(9) 
        text = ""
        # mannully adjust the constants according to our camera quality and screen resolution
        if y < -3:
            self.movment = "Looking Left"
            # self.rotate_cameraH(direction=-1, dt=dt)
            self.updateKeyMap("left" , False)
            self.updateKeyMap("right" , True)
        elif y > 3:
            self.movment = "Looking Right"
            # self.rotate_cameraH(direction=1, dt=dt)
            self.updateKeyMap("right" , False)
            self.updateKeyMap("left" , True)
        elif x < -3:
            self.movment = "Looking Down"
            # self.rotate_cameraP(-1, dt)
            self.updateKeyMap("forward" , True)
            self.updateKeyMap("backward" , False)
        elif x > 3:
            self.movment = "Looking Up"
            # self.rotate_cameraP(1, dt)
            self.updateKeyMap("forward" , False)
            self.updateKeyMap("backward" , True)
        else:
            text = "Forward"
            self.updateKeyMap("right" , False)
            self.updateKeyMap("left" , False)
            self.updateKeyMap("forward" , False)
            self.updateKeyMap("backward" , False)

        face_landmarker_result = self.faceMesh.getFaceBlendShape()
        if face_landmarker_result:
            for category in face_landmarker_result[0]:
                if (
                    category.category_name == "browInnerUp"
                    and float(category.score) >= float(0.7)
                ) and self.remove_counter == 0:
                    self.remove_counter = self.remove_counter + 1

                    self.removeBlock()

                if (
                    category.category_name == "browleftUp"
                    and category.score >= 0.3
                    and self.plcae_counter == 0
                ):
                    self.placeBlock()
                    self.plcae_counter += 1

         
                  
                    
                    
                    

        if self.remove_counter != 0:
            self.remove_counter = self.remove_counter + 1
            print(self.remove_counter)
            if self.remove_counter > 5:
                self.remove_counter = 0

        if self.plcae_counter != 0:
            self.plcae_counter = self.plcae_counter + 1
            print(self.plcae_counter)
            if self.plcae_counter > 5:
                self.plcae_counter = 0
                
                
        self.faceMesh.drawResult()

        return task.cont

    def rotate_cameraH(self, direction, dt):
        angle = direction * self.rotate_speed * dt
        self.camera.setH(self.camera.getH() + angle)

    def rotate_cameraP(self, direction, dt):
        angle = direction * self.rotate_speedP * dt
        self.camera.setP(self.camera.getP() + angle)

    def setupControls(self):
        self.keyMap = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }

        self.accept("escape", self.releaseMouse)
        self.accept("mouse1", self.handleLeftClick)
        self.accept("mouse3", self.placeBlock)

        self.accept("w", self.updateKeyMap, ["forward", True])
        self.accept("w-up", self.updateKeyMap, ["forward", False])
        self.accept("a", self.updateKeyMap, ["left", True])
        self.accept("a-up", self.updateKeyMap, ["left", False])
        self.accept("s", self.updateKeyMap, ["backward", True])
        self.accept("s-up", self.updateKeyMap, ["backward", False])
        self.accept("d", self.updateKeyMap, ["right", True])
        self.accept("d-up", self.updateKeyMap, ["right", False])
        self.accept("space", self.updateKeyMap, ["up", True])
        self.accept("space-up", self.updateKeyMap, ["up", False])
        self.accept("lshift", self.updateKeyMap, ["down", True])
        self.accept("lshift-up", self.updateKeyMap, ["down", False])

        self.accept("1", self.setSelectedBlockType, ["grass"])
        self.accept("2", self.setSelectedBlockType, ["dirt"])
        self.accept("3", self.setSelectedBlockType, ["sand"])
        self.accept("4", self.setSelectedBlockType, ["stone"])

    def setSelectedBlockType(self, type):
        self.selectedBlockType = type

    def handleLeftClick(self):
        self.captureMouse()
        self.removeBlock()

    def removeBlock(self):
        if self.rayQueue.getNumEntries() > 0:
            self.rayQueue.sortEntries()
            rayHit = self.rayQueue.getEntry(0)

            hitNodePath = rayHit.getIntoNodePath()
            hitObject = hitNodePath.getPythonTag("owner")
            distanceFromPlayer = hitObject.getDistance(self.camera)

            if distanceFromPlayer < 12:
                hitNodePath.clearPythonTag("owner")
                hitObject.removeNode()

    def placeBlock(self):
        if self.rayQueue.getNumEntries() > 0:
            self.rayQueue.sortEntries()
            rayHit = self.rayQueue.getEntry(0)
            hitNodePath = rayHit.getIntoNodePath()
            normal = rayHit.getSurfaceNormal(hitNodePath)
            hitObject = hitNodePath.getPythonTag("owner")
            distanceFromPlayer = hitObject.getDistance(self.camera)

            if distanceFromPlayer < 14 and distanceFromPlayer > 3:
                hitBlockPos = hitObject.getPos()
                newBlockPos = hitBlockPos + normal * 2
                self.createNewBlock(
                    newBlockPos.x, newBlockPos.y, newBlockPos.z, self.selectedBlockType
                )

    def updateKeyMap(self, key, value):
        self.keyMap[key] = value

    def captureMouse(self):
        self.cameraSwingActivated = True

        md = self.win.getPointer(0)
        self.lastMouseX = md.getX()
        self.lastMouseY = md.getY()

        properties = WindowProperties()
        properties.setCursorHidden(True)
        properties.setMouseMode(WindowProperties.M_relative)
        self.win.requestProperties(properties)

    def releaseMouse(self):
        self.cameraSwingActivated = False

        properties = WindowProperties()
        properties.setCursorHidden(False)
        properties.setMouseMode(WindowProperties.M_absolute)
        self.win.requestProperties(properties)

    def setupCamera(self):
        self.disableMouse()
        self.camera.setPos(0, 0, 3)
        self.camLens.setFov(80)

        crosshairs = OnscreenImage(
            image="assets/crosshairs.png",
            pos=(0, 0, 0),
            scale=0.05,
        )
        crosshairs.setTransparency(TransparencyAttrib.MAlpha)

        self.cTrav = CollisionTraverser()
        ray = CollisionRay()
        ray.setFromLens(self.camNode, (0, 0))
        rayNode = CollisionNode("line-of-sight")
        rayNode.addSolid(ray)
        rayNodePath = self.camera.attachNewNode(rayNode)
        self.rayQueue = CollisionHandlerQueue()
        self.cTrav.addCollider(rayNodePath, self.rayQueue)

    def setupSkybox(self):
        skybox = self.loader.loadModel("assets/skybox/skybox.egg")
        skybox.setScale(500)
        skybox.setBin("background", 1)
        skybox.setDepthWrite(0)
        skybox.setLightOff()
        skybox.reparentTo(self.render)

    def generateTerrain(self):
        for z in range(20):
            for y in range(30):
                for x in range(30):
                    self.createNewBlock(
                        x * 2 - 30, y * 2 - 30, -z * 2, "grass" if z == 0 else "dirt"
                    )

    def createNewBlock(self, x, y, z, type):
        newBlockNode = self.render.attachNewNode("new-block-placeholder")
        newBlockNode.setPos(x, y, z)

        if type == "grass":
            self.grassBlock.instanceTo(newBlockNode)
        elif type == "dirt":
            self.dirtBlock.instanceTo(newBlockNode)
        elif type == "sand":
            self.sandBlock.instanceTo(newBlockNode)
        elif type == "stone":
            self.stoneBlock.instanceTo(newBlockNode)

        blockSolid = CollisionBox((-1, -1, -1), (1, 1, 1))
        blockNode = CollisionNode("block-collision-node")
        blockNode.addSolid(blockSolid)
        collider = newBlockNode.attachNewNode(blockNode)
        collider.setPythonTag("owner", newBlockNode)

    def loadModels(self):
        self.grassBlock = self.loader.loadModel("assets/grass-block.glb")
        self.dirtBlock = self.loader.loadModel("assets/dirt-block.glb")
        self.stoneBlock = self.loader.loadModel("assets/stone-block.glb")
        self.sandBlock = self.loader.loadModel("assets/sand-block.glb")

    def setupLights(self):
        mainLight = DirectionalLight("main light")
        mainLightNodePath = self.render.attachNewNode(mainLight)
        mainLightNodePath.setHpr(30, -60, 0)
        self.render.setLight(mainLightNodePath)

        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor((0.3, 0.3, 0.3, 1))
        ambientLightNodePath = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNodePath)


game = MyGame()
game.run()
game.cap.release()
