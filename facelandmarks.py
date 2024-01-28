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
  annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
  facelist = []
  face_3d = []
  face_2d = []
  height,width ,_ = rgb_image.shape

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
        
        cx = (landmark.x * width)
        cy =(landmark.y * height)
        cz = (landmark.z)
        facelist.append([index, cx, cy, cz])
        if(len(facelist)>200):
          nose_2d = (facelist[1][1], facelist[1][2])
          nose_3d = (facelist[1][1], facelist[1][2], facelist[1][3]*3000)

          x, y = int(facelist[1][1]),int(facelist[1][2])
          
          face_2d.append([x,y])
          face_3d.append([x,y,facelist[1][3]])
     
     


    face_2d = np.array(face_2d,dtype=np.float32)

    face_3d = np.array(face_3d,dtype=np.float32)

    #camera matrix
    focal_length = 1*width

    cam_matrix = np.array([ [focal_length, 0, width / 2],[0, focal_length, height / 2],[0, 0, 1]],dtype=np.float32)
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float32)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
            # See where the user's head tilting
    if y < -10:
        text = "Looking Left"
        print("Looking Left")
    elif y > 10:
        text = "Looking Right"
        print("Looking Right")
    elif x < -10:
        text = "Looking Down"
        print("Looking Down")
    elif x > 10:
        text = "Looking Up"
        print("Looking Up")
    else:
        text = "Forward"
        print("Forward")
        
        
    # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 150) , int(nose_2d[1] - x * 150))
    cv2.line(annotated_image, p1, p2, (255, 0, 0), 3)
      
    
          

  return annotated_image,facelist




base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture("Dance - 32938.mp4")

while True:
    ret,img = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    face_landmarker_result = detector.detect(mp_image)
    annotated_image,facelist = draw_landmarks_on_image(img,face_landmarker_result)





  
    cv2.imshow("hay",annotated_image)





    key = cv2.waitKey(1)
    if key == ord("q"):
     break
cap.release()
cv2.destroyAllWindows()