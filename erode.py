import cv2

# Read input Image
imageName = "images/opening.png"
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Check for an invalid input
if image is None:
  print("Could not open or find the image")

# Get structuring element/kernel which will be used for dilation
erosionSize = 6
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2*erosionSize+1, 2*erosionSize+1),
                                    (erosionSize, erosionSize))

# Apply erode function on the input image
imageEroded = cv2.erode(image, element)

#  Display original image
cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image", image)

#  Display eroded image
cv2.namedWindow("Erosion", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Erosion", imageEroded)

#  Wait for the user to press any key
cv2.waitKey(0)

cv2.destroyAllWindows()
