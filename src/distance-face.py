import face_recognition
import cv2

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print ("usage:%s (cameraID | filename) Detect faces\
 in the video:%s 0"%(sys.argv[0], sys.argv[0]))
        exit(1)

    try:
    	camID = int(sys.argv[1])
    except:
    	camID = sys.argv[1]   
    

    cap = cv2.VideoCapture(camID)
    #cap.set(3, 1280)
    #cap.set(4, 720)
    windowNotSet = True
    
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        print (h, w)
        image = cv2.flip(image, 1)

        boxes = face_recognition.face_locations(image, model="cnn")
        print(boxes)

        for top, right, bottom, left in boxes:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            # Extract the region of the image that contains the face
            face_image = image[top:bottom, left:right]

            # Blur the face image
            face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

            # Put the blurred face region back into the frame image
            image[top:bottom, left:right] = face_image

        
        cv2.imshow('Video', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()