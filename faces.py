import cv2
### Step 1 : find face + Step 2 : crop around face
##################################################
def auto_crop_image(image):
    if image is not None:
        im = image.copy()
        classifierPath='./classifiers/haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(classifierPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        roi_color=None
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                roi_color = image[y-50:y+h+50, x-50:x+w+50]

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)        
            (x, y, w, h) = faces[0]
            
            center_x = x+w/2
            center_y = y+h/2
            height, width, channels = im.shape
            b_dim = min(max(w,h)*1.2,width, height)
            box = [center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2]
            box = [int(x) for x in box]
            
            # Crop Image
            if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                crpim = im[box[1]:box[3],box[0]:box[2]]
                crpim = cv2.resize(crpim, (224,224), interpolation = cv2.INTER_AREA)
                print("Found {0} faces!".format(len(faces)))
                return crpim, image, (x, y, w, h)
    return roi_color, image, (0,0,0,0)