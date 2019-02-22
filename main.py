#############################################
# Author:  Fabio Andres Herrera - 
# Mail: fabio.herrera@correounivalle.edu.co
################################################

import os
import cv2
import numpy as np

roiWidth = 36
roiHeight = 36
minContourArea = 36*36
maxContourArea = 60*60
gap=5

def main():
	inputImage = cv2.imread("input_image.jpg")
	imgGray = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
	imgBlurred = cv2.GaussianBlur(imgGray,(9,9),0)
	imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	cv2.imshow("Threshold Image", imgThresh)
	imgThreshCopy = imgThresh.copy()
	imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, 1,2)
	found=0
	for npaContour in npaContours:
		if (cv2.contourArea(npaContour) > minContourArea):
			if ( cv2.contourArea(npaContour) < maxContourArea ):
				[intX, intY, intW, intH] = cv2.boundingRect(npaContour)
				#draw rectangle around each contour
				cv2.rectangle(inputImage,(intX+gap, intY+gap),((intX+intW)-gap,(intY+intH)-gap),(0,0,255),2)
				imgROI = imgThresh[intY+gap:(intY+intH)-gap, intX+gap:(intX+intW)-gap]
				imgROIResized = cv2.resize(imgROI,(roiWidth,roiHeight))
				cv2.imshow("Original File",inputImage)
				found+=1
				cv2.imwrite('output/letter_'+str(found)+'.png',cv2.bitwise_not(imgROIResized))
	print(str(found) + " files written into output folder !")
	cv2.waitKey(0)
	cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
