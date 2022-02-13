import cv2
import numpy as np
import utlis
import time


########################################################################
webCamFeed = True

#Upload video here
cap = cv2.VideoCapture('Test2.mp4')



cap.set(10,160)
heightImg = 480
widthImg  = 640
########################################################################

utlis.initializeTrackbars()
count=0

while True:
    if webCamFeed:success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS


    #Thresolding
    #thresh = cv2.threshold(imgBlur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    rt1,thresh = cv2.threshold(imgBlur, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    kernel = np.ones((3, 3))


    erodeThresh = cv2.erode(thresh,kernel,iterations=0)
    binary = cv2.bitwise_not(erodeThresh)

    (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    cv2.imshow("eroded",img)

    imgCanny = cv2.Canny(thresh,thres[0],thres[1]) # APPLY CANNY BLUR
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2) # APPLY DILATION
    imgErode = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION


    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        noScreen = 0;
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)

        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        print("Points are:",pts2)
        #print(pts2)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        # Image Array for Display
        imageArray = ([img,imgGray,imgBlur,thresh],
                      [imgCanny,imgDial, imgBigContour,imgWarpColored])

    else:
        noScreen = 1;
        imageArray = ([img,imgGray,imgBlur,thresh],
                      [imgCanny, imgDial, imgBigContour, imgBlank])
        imgWarpColored = img


    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Blur","Threshold"],
              ["Canny","Dilate","Points","Warped"]]

    stackedImage = utlis.stackImages(imageArray,0.5,lables)
    imgOutput = imgWarpColored

    if noScreen == 1:
        imgOutput = img
        cv2.putText(imgOutput, "Screen not Detected",
                    (180,200),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

    cv2.imshow("Process",stackedImage)
    cv2.imshow("Output",imgOutput)



    # SAVE IMAGE WHEN 's' key is pressed
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

