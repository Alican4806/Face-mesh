from xml.dom.expatbuilder import parseString
import cv2 as cv
import mediapipe as mp
import time 

# FPS is very low so this is very slow

get = cv.VideoCapture('C:/Users/alica/Desktop/Projects about computer vision/Face detection/Endgame.mp4')
faceMesh = mp.solutions.face_mesh
fMesh = faceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
 
# The points which are occurred on the face, are adjusted
drawSpec = mpDraw.DrawingSpec(thickness = 1,circle_radius = 2,color = (150,255,0))

cTime = 0
pTime = 0
while True:
    
    success , img = get.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = fMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLm,faceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            for id, lm in enumerate(faceLm.landmark):
                h ,w,c =img.shape
                # print(h,w,c)
                # print(lm)
                
                x,y = int(lm.x*w),int(lm.y*h)
            
                print(id,x,y)
    # Fps value is written on the screen      
        cTime = time.time()
        fps = 1/(cTime-pTime)
    
        cv.putText(img,f'FPS: {int(fps)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        pTime = cTime    
    
        cv.imshow('img',img)
    
    
    if cv.waitKey(1) & 0xFF == ord('a'):
        break
        
cv.destroyAllWindows()                                                                                                      