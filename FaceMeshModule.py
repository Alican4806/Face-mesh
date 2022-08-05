from xml.dom.expatbuilder import parseString
import cv2 as cv
import mediapipe as mp
import time 

class FaceMesh:
    # These parameter will be used in FaceMesh
    def __init__(self,
               static_Mod=False,
               max_faces=2,
               refine_landmarks=False,
               min_detectionCon=0.5, # It is rised value to get rid of the fails
               min_trackingCon=0.5):
        
        self.static_Mod= static_Mod
        self.max_faces=max_faces
        self.refine_landmarks=refine_landmarks
        self.min_detectionCon=min_detectionCon
        self.min_trackingCon=min_trackingCon
        
        self.faceMesh = mp.solutions.face_mesh
        self.fMesh = self.faceMesh.FaceMesh(self.static_Mod,self.max_faces,self.refine_landmarks,self.min_detectionCon,self.min_trackingCon) # It is used here
        self.mpDraw = mp.solutions.drawing_utils        

 
        # The points which are occurred on the face, are adjusted
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1,circle_radius = 2,color = (150,255,0))
    
    def FindFaceMesh(self,img,draw = True):
        

        self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.fMesh.process(self.imgRGB)
        faces = []
        
        if self.results.multi_face_landmarks:
            
            for faceLm in self.results.multi_face_landmarks:
                if draw:  
                    
                    self.mpDraw.draw_landmarks(img,faceLm,self.faceMesh.FACEMESH_CONTOURS,self.drawSpec)
                locationCount = []     
                for id, lm in enumerate(faceLm.landmark):
                    h ,w,c =img.shape
                        # print(h,w,c)
                        # print(lm)
                        
                    x,y = int(lm.x*w),int(lm.y*h)
                    cv.putText(img,str(id),(x,y),cv.FONT_HERSHEY_PLAIN,0.5,(150,255,0),1) 
                    locationCount.append([x,y])
                        # print(id,x,y)
                faces.append(locationCount)        
        return img, faces
        
            
    
    
   
        
                  
def main():
    
    
    get = cv.VideoCapture('C:/Users/alica/Desktop/Projects about computer vision/Face detection/I_like_this.mp4')
    
    pTime = 0
    
    FindMesh = FaceMesh()
    while True:
        
        succes, img = get.read()
        cTime = time.time()
        fps = 1/(cTime-pTime)
    
        cv.putText(img,f'FPS: {int(fps)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        pTime = cTime    
        img, facess = FindMesh.FindFaceMesh(img,True)
        if len(facess)!= 0:
            
            print(facess[0])
        cv.imshow('image',img)
        if cv.waitKey(20) & 0xFF == ord('a'):
            break   
        
if __name__== "__main__": # I made it to work module within itself
    main()                                                                                   