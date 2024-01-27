import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import math

from cvzone.FaceMeshModule import FaceMeshDetector
from sympy import Point





def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = rgb_image
  facelist = []

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())
    
    for index, landmark in enumerate(face_landmarks_proto.landmark):
        height,width ,_ = rgb_image.shape
        cx = (landmark.x)
        cy =(landmark.y)
        cz = (landmark.z * width)
        facelist.append([index, cx, cy, cz])

    cv2.circle(annotated_image,(int(facelist[468][1]*width),int(facelist[468][2]*height)),4,(255,26,255),5)
    cv2.circle(annotated_image,(int(facelist[473][1]*width),int(facelist[473][2]*height)),4,(255,26,255),5)
    cv2.circle(annotated_image,(int(facelist[152][1]*width),int(facelist[152][2]*height)),4,(255,26,255),5)
    cv2.circle(annotated_image,(int(facelist[10][1]*width),int(facelist[10][2]*height)),4,(255,26,255),5)




  return annotated_image,facelist



base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture("Dance - 32938.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)


while True:
    ret,img = cap.read()
    

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    face_landmarker_result = detector.detect(mp_image)
    annotated_image,facelist = draw_landmarks_on_image(img,face_landmarker_result)

    if (len(facelist)> 473):
        xrotation = (facelist[473][1] - facelist[468][1])*180
        yrotation = (facelist[152][2]-facelist[10][2])*180
        print(xrotation,yrotation)


    cv2.imshow("hay",annotated_image)





    key = cv2.waitKey(1)
    if key == ord("q"):
     break
cap.release()
cv2.destroyAllWindows()