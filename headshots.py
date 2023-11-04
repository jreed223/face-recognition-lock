import cv2
from picamera2 import Picamera2, Preview

name = 'Jon' #replace with your name
cam = cv2.VideoCapture(0)

cam = Picamera2()
cam.start_preview(Preview.QTGL)
cam.start()

#raqImage = PiRGBArray(cam)


cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL) #creates window to b
cv2.resizeWindow("press space to take a photo", 750, 450)

img_counter = 0

while True:
    frame = cam.capture_array() #binds ret and frame to the tuple items returned: (boolean
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #if not ret: #breaks loop if no video feed or frame captured
    #    print("failed to grab frame")
    #    break
    cv2.imshow("press space to take a photo", rgb)
    
    k = cv2.waitKey(1)
    if k%256 == 27: #breaks loop if escape key pressed
    # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32: #loop continues if space is pressed
    # SPACE pressed
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter) #formats imag
        cv2.imwrite(img_name, frame) #adds current frame to directory specified in the
        print("{} written!".format(img_name))
        img_counter += 1
cam.stop()
cv2.destroyAllWindows()
