import numpy as np
import cv2
import copy
from video import video

cap = cv2.VideoCapture('K1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
num_frames = 500

first_iteration_indicator = 1

for i in range(0,num_frames):
    ret, frame = cap.read()

	#the if part is only for the first image and the else part is for the rest.
    if (first_iteration_indicator == 1):
        first_frame = copy.deepcopy(frame)
        height, width = frame.shape[:2]
        accum_image = np.zeros((height, width), np.uint8)
        first_iteration_indicator = 0
    else:
        fgmask = fgbg.apply(frame)  #remove the background
        thresh = 2
        maxValue = 2
        ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)

        # add to the accumulated image
        accum_image = cv2.add(accum_image, th1)
        #the accumulated image
        #cv2.imwrite('diff-accum.jpg', accum_image)

	#applying a color map.You can also use different color maps like COLORMAP_SUMMER or COLORMAP_BONE or COLORMAP_PINK.
        color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

	#adding two images
        result_overlay = cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)
	
	#storing the files in Image directory by specifying the path of directory
        cv2.imwrite("/root/PycharmProjects/HeatMap/Images/image"+str(i)+".jpg", result_overlay)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

video("/root/PycharmProjects/HeatMap/Images","K1.avi") #This will call the video function of video.py file

cap.release()
cv2.destroyAllWindows()
