import cv2
import numpy as np
import time
from keras.models import load_model



model = load_model('image_recmodel.h5')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width,frame_height))

while (True):
    ret, frame = cap.read()


    if ret == True:


        # Write the frame into the file 'output.avi'
        frame = cv2.resize(frame, (480, 240), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

        # Display the resulting frame

        cv2.imshow('frame', frame)
        frame = frame[np.newaxis, ...]



        pred = model.predict(frame)
        print(pred)
        if (pred[0][0] > pred[0][1]):
            pred_label = "night"
        else:
            pred_label = "noon"

        print(pred_label)
        time.sleep(0.01)




        # Press Q on keyboard to stop recordingf
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

    # When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()