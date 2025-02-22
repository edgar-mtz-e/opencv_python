import cv2
import numpy as np
#Read input image
image = cv2.imread('images/mark.jpg')

# Draw a line
imageLine = image.copy()
cv2.line(imageLine, (322, 179), (400, 183), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow("imageLine", imageLine)
cv2.imwrite("results/imageLine.jpg", imageLine)

# Draw a circle
imageCircle = image.copy()
cv2.circle(imageCircle, (350, 200), 150, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow("imageCircle", imageCircle)
cv2.imwrite("results/imageCircle.jpg", imageCircle)

# Draw an ellipse
# Note: Ellipse Centers and Axis lengths must be integers
imageEllipse = image.copy()
cv2.ellipse(imageEllipse, (360, 200), (100, 170), 45, 0, 360, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.ellipse(imageEllipse, (360, 200), (100, 170), 135, 0, 360, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow("ellipse", imageEllipse)
cv2.imwrite("results/imageEllipse.jpg", imageEllipse)

# Draw a rectangle (thickness is a positive integer)
imageRectangle = image.copy()
cv2.rectangle(imageRectangle, (208, 55), (450, 355), (0, 255, 0), thickness=2, lineType=cv2.LINE_8)
cv2.imshow("rectangle", imageRectangle)
cv2.imwrite("results/imageRectangle.jpg", imageRectangle)

# Put text into image
imageText = image.copy()
cv2.putText(imageText, "Mark Zuckerberg", (205, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("text", imageText)
cv2.imwrite("results/imageText.jpg", imageText)

cv2.waitKey(0)
cv2.destroyAllWindows()
