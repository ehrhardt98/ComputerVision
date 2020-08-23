import cv2
import matplotlib.pyplot as plt
import numpy as np

dictAruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

board = cv2.aruco.CharucoBoard_create(5, 7, 3.85, 2, dictAruco)

img1 = cv2.imread("charuco.png", cv2.IMREAD_GRAYSCALE)
w1, h1 = img1.shape
mcorners1, mids1, rejectedImgPoints1 = cv2.aruco.detectMarkers(img1, dictAruco)

markers1 = cv2.aruco.drawDetectedMarkers(img1, mcorners1, mids1)

retval1, corners1, charucoIds1 = cv2.aruco.interpolateCornersCharuco(mcorners1, mids1, img1, board)
corn1 = img1.copy()
corn1 = cv2.aruco.drawDetectedCornersCharuco(img1, corners1, charucoIds1)

corners1 = corners1.reshape((24,2)).astype(int)
corners1.shape

path1 = "C:/Users/jorge/Pictures/mems/yee.jpg"
img2 = cv2.imread(path1)
img2 = cv2.resize(img2, (h1, w1))


camera_index = 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
ret, frame = cap.read()

cap.set(3, 1280)
cap.set(4, 1024)

_, frame = cap.read()
while frame is not None:
    
#     warp = frame.copy()
    
    try:
        h, w, _ = frame.shape

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mcorners, mids, rejectedImgPoints = cv2.aruco.detectMarkers(img, dictAruco)

        markers = cv2.aruco.drawDetectedMarkers(frame, mcorners)

        if len(mcorners) > 0:
            retval2, corners2, charucoIds2 = cv2.aruco.interpolateCornersCharuco(mcorners, mids, img, board)
            corn = img.copy()
    #         corn = cv2.aruco.drawDetectedCornersCharuco(frame, corners2, charucoIds2)


            if charucoIds2 is not None:
                if len(charucoIds2) > 3:
                    new_corners = []
                    for i in range(len(corners1)):
                        if i in charucoIds2:
                            new_corners.append(corners1[i])
                    new_corners = np.array(new_corners)

                    homog = cv2.findHomography(new_corners, corners2)[0] #, cv2.RANSAC)

                    warp = cv2.warpPerspective(img2, homog, (w, h), np.zeros_like(frame))

                    nova = (~(warp > 0)*255).astype('uint8')

                    frame = (nova & frame) | warp
    except:
        continue    
    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord(' '):
        break
    elif k == ord('l'):
        path = "C:/Users/jorge/Pictures/mems/long.jpg"
        img2 = cv2.imread(path)
        img2 = cv2.resize(img2, (h1, w1))
    elif k == ord('k'):
        img2 = cv2.imread(path1)
        img2 = cv2.resize(img2, (h1, w1))

    _, frame = cap.read()

cap.release()
cv2.destroyAllWindows()