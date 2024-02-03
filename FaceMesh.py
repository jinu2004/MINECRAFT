import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time


class FaceMesh:
    def __init__(self):
        self.base_options = python.BaseOptions(
            model_asset_path="models/face_landmarker.task"
        )
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        self.face_landmark = []

    def FaceDetector(self, cap , landmark_draw = False ):
        x = 0
        y = 0
        z = 0
        success, self.image = cap.read()
        start = time.time()
        self.image = cv2.resize(self.image, (1900, 1080))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.image)
        self.face_landmarker_result = self.detector.detect(mp_image)
        self.face_landmarks_list = self.face_landmarker_result.face_landmarks
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        for idx in range(len(self.face_landmarks_list)):
            face_landmarks = self.face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            if landmark_draw:
                solutions.drawing_utils.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                )
            
            
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(
            self.image,
            f"FPS: {int(fps)}",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
        )

    def FaceLandmarkesPos(self, index):
        img_h, img_w, img_c = self.image.shape
        face_3d = []
        face_2d = []
        x = 0
        y = 0
        z = 0
        for idx in range(len(self.face_landmarks_list)):
            face_landmarks = self.face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            for idx, lm in enumerate(face_landmarks_proto.landmark):
                self.face_landmark.append([idx, lm.x, lm.y, lm.z])
                if (
                    idx == 33
                    or idx == 263
                    or idx == 9
                    or idx == index
                    or idx == 168
                    or idx == 199
                ):
                    if idx == index:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    tx, ty = (lm.x * img_w), (lm.y * img_h)
                    xAvL = []
                    yAvL = []
                    xAvL.append(tx)
                    yAvL.append(ty)

                    if len(xAvL) > 5:
                        xAvL = xAvL.pop(0)
                    if len(yAvL) > 5:
                        yAvL = yAvL.pop(0)
                    if xAvL and yAvL:
                        xAv = sum(xAvL) / len(xAvL)
                        yAv = sum(yAvL) / len(yAvL)

                    # Get the 2D Coordinates
                    face_2d.append([xAv, yAv])

                    # Get the 3D Coordinates
                    face_3d.append([xAv, yAv, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -2:
                text = "Looking Left"
            elif y > 2:
                text = "Looking Right"
            elif x < -2:
                text = "Looking Down"
            elif x > 2:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(self.image, p1, p2, (255, 0, 0), 3)
        return x, y, z

    def getFaceBlendShape(self):
        return self.face_landmarker_result.face_blendshapes


    def drawResult(self):
        self.image = cv2.resize(self.image, (600, 500))
        cv2.imshow("result", self.image)
        return self.image
       
        
        
    
