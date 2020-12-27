import numpy as np
import cv2

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea =1500
detector = cv2.SimpleBlobDetector_create(detector_params)
faceCascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def blob_process(img, threshold,detector):
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(grayFrame,  threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoint = detector.detect(img)
    print(keypoint)
    return keypoint
 

def detectEye(img, classifier):
    grayFreme = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(grayFreme, 1.3, 5)

    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x+w/2
        if eyecenter < width * 0.5:
            left_eye = img[y:y+h, x:x+w]
        else:
            right_eye = img[y:y+h, x:x+w]
    return left_eye, right_eye

def detectFace(img, classifier):
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(grayFrame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i [3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y+h, x:x+w]
    return frame

def cutEyebrow(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    return img

def nothing(x):
    pass

cv2.namedWindow('eyeTracker')
cv2.createTrackbar('threshold', 'eyeTracker', 0, 255, nothing)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    frame_face = detectFace(img, faceCascade)
    if frame_face is not None:
        eyes = detectEye(frame_face, eyesCascade)
        for eye in eyes:
            if eye is not None:
                threshold = r = cv2.getTrackbarPos('threshold', 'eyeTracker')
                eye = cutEyebrow(eye)
                keypoint = blob_process(eye, threshold, detector)
                eye = cv2.drawKeypoints(eye, keypoint, eye, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('eyeTracker', img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



