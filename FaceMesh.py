import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

class FaceMesh():
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                                output_face_blendshapes=True,
                                                output_facial_transformation_matrixes=True,
                                                num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

        
    def FaceDetector(self,cap):
        success, self.image = cap.read()
        start = time.time()
        self.image = cv2.cvtColor(cv2.flip(self.image, 1), cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.image)
        face_landmarker_result = self.detector.detect(mp_image)
        face_landmarks_list = face_landmarker_result.face_landmarks
        self.image.flags.writeable = True
        # self.image = cv2.resize(self.image,(650,400))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = self.image.shape
        face_3d = []
        face_2d = []
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            
            face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])
            
            solutions.drawing_utils.draw_landmarks(
                                    image=self.image,
                                    landmark_list=face_landmarks_proto,
                                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp.solutions.drawing_styles
                                    .get_default_face_mesh_tesselation_style())

          
            
            for idx, lm in enumerate(face_landmarks_proto.landmark):
                if idx == 33 or idx == 263 or idx == 9 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 9:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

        # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            self.x = angles[0] * 360
            self.y = angles[1] * 360
            self.z = angles[2] * 360


            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + self.y * 10) , int(nose_2d[1] - self.x * 10))
            
            cv2.line(self.image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(self.image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(self.image, "x: " + str(np.round(self.x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "y: " + str(np.round(self.y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "z: " + str(np.round(self.z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        self.image = cv2.resize(self.image,(600,500))

        cv2.putText(self.image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow("window",self.image)
        
        return self.x *10, self.y*10, self.z


        








        


        