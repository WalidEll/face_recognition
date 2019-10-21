from __future__ import print_function

import cv2
import requests
import json

from faces import auto_crop_image 
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,800)
    cap.set(4,600)
    addr = 'http://localhost:5000'
    test_url = addr + '/api/test'

    # prepare headers for http request
    content_type = 'image/png'
    headers = {'content-type': content_type}
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        imgcrop,frame, (x, y, w, h) = auto_crop_image(frame)
        img_item = "anchor.png"
        cv2.imwrite(img_item,cv2.resize(imgcrop,(96,96)))
        cv2.imshow('frame',frame)
        _, img_encoded = cv2.imencode('.png', imgcrop)
        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        print(json.loads(response.text))

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()