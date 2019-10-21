import cv2
from faces import auto_crop_image 
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,800)
    cap.set(4,600)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        imgcrop,frame, (x, y, w, h) = auto_crop_image(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()