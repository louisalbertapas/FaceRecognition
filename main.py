# Face Recognition

# Import libraries
import cv2 as cv

# Load cascades
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')


# function to detect face
def detect_face(gray, frame):
    faces = face_cascade.detectMultiScale(image=gray,
                                          scaleFactor=1.3,  # tells how much the image will be reduced by a factor
                                          minNeighbors=5)  # minimum number of neighbor that should also be accepted

    # iterate all the detected faces
    for x, y, w, h in faces:
        # draw a rectangle on the detected faces
        cv.rectangle(img=frame,
                     pt1=(x, y),  # upper left point of the rectangle
                     pt2=(x + w, y + h),  # lower right point of the rectangle
                     color=(0, 255, 0),
                     thickness=2)

        # get the region of interest (roi)
        roi_gray = gray[y: y + h, x: x + w]  # get the detected face as grayscale
        roi_color = frame[y: y + h, x: x + w]  # get the detected face as color. NOTE: this is not deep copy

        # detect eyes in the face
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for ex, ey, ew, eh in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    return frame


# driver code to do face recognition
cap = cv.VideoCapture(0)  # 0 - internal web cam, 1 - external web cam

while True:
    # reads the latest frame of the video from the camera
    _, frame = cap.read()

    # convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # call our function to detect faces and eyes
    detected = detect_face(gray, frame)

    cv.imshow('frame', detected)

    if cv.waitKey(1) == '27':  # wait for ESC key
        break

cap.release()
cv.destroyAllWindows()

