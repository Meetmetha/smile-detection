import cv2
import numpy as np
import dlib

predictor = dlib.shape_predictor("shape_predictor.dat")

(mStart, mEnd) = (48,67)

smile_const = 8
counter = 0 #when counter reaches 15 frames a selfie will be captured
selfie_no = 0

cam = cv2.VideoCapture(0)

while(cam.isOpened()):

    
    ret, image = cam.read()
    image = cv2.flip(image,1) #1 represents flip at y axis
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for i in range (0, len(rects)):

        #convert dlib's rectangle to a OpenCV-style bounding box
        #[i.e (x,y,w,h)] then draw the face bounding box
        (x,y,w,h) = rect_to_bb(rects[i])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        #show the face number
        cv2.putText(image,"Face #{}".format(i+1), (x-20,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        #determine the facial landmarks for the face region
        shape = predictor(gray,rects[i])

        #convert the facial landmark (x,y)coordinates into a numpy array
        shape = shape_to_np(shape)

        mouth = shape[mStart:]
        

        #loop over the (x,y) coordinates for the facial landmarks and draw them on the image
        for (x,y) in mouth:
            cv2.circle(image,(x,y),1,(255,255,255),-1)

            #smile parameter from mouth
            smile_param = smile(shape)

            cv2.putText(image,"SP: {:.2f}".format(smile_param),(300,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            if(smile_param > smile_const):
                cv2.putText(image,"Smile Detected",(300,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                counter += 1
                if counter >=15:  #If smile is sustained for 15 frames,take selfie
                    selfie_no +=1
                    ret, frame = cam.read()
                    img_name = "smart_selfie_{}.png".format(selfie_no)
                    cv2.imwrite(img_name,frame)
                    print("{} taken!".format(img_name))
                    counter = 0  #Reset counter once selfie is taken
            else:
                counter = 0  #reset counter once the smile is not detected in a frame
                
            

    cv2.imshow('live face',image)
    key = cv2.waitKey(1)

    if key == 27:
        break   
cam.release()
cv2.destroyAllWindows()

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()-x
    h = rect.bottom()-y

    #return a tuple of (x,y,w,h)
    return (x,y,w,h)

#function to convert the facial coordinates recognized by the predictor
#into a NumPy array, to ease the further usage of the points
def shape_to_np(shape,dtype="int"):
    #initialize the list of (x,y) coordinates
    coords = np.zeros((68,2),dtype=dtype)

    #loop over the 68 facial landmarks and convert them to a 2-tuple of (x,y)-coordinates
    for i in range(0,68):
        coords[i] = (shape.part(i).x,shape.part(i).y)
    return coords

def smile(shape):
    left = shape[48]
    right = shape[54]

    #average of the points in the center of the mouth
    mid = (shape[51] + shape[62] +shape[66] +shape[57])/4

    #perpendicular distance between the mid  and the line joinning left and right
    dist = np.abs(np.cross(right-left,left - mid)/np.linalg.norm(right-left))
    return dist
                  
