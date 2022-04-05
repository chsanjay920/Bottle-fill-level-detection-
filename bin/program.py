import cv2
from matplotlib import pyplot as plt
import imutils

bottle_3_channel = cv2.imread("./TestImages/200ml.jpg")
# bottle_gray = cv2.split(bottle_3_channel)[0]

# cv2.imshow("Bottle Gray", bottle_gray)
# cv2.waitKey(0)


# bottle_gray = cv2.GaussianBlur(bottle_gray, (7, 7), 0)
# # cv2.imshow("Bottle Gray Smoothed 7 x 7", bottle_gray)
# # cv2.waitKey(0)

# plt.hist(bottle_gray.ravel(), 256,[0, 256]); plt.show()


# # manual threshold
# (T, bottle_threshold) = cv2.threshold(bottle_gray, 85.0, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Bottle Gray Threshold 27.5", bottle_threshold)
# cv2.waitKey(0)


img_gray = cv2.cvtColor(bottle_3_channel, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)


# cv2.imshow('Sobel X', sobelx)
# cv2.waitKey(0)
# cv2.imshow('Sobel y', sobely)
# cv2.waitKey(0)
# cv2.imshow('Sobel Xy', sobelxy)
# cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=600) 

# Canny Edge Detection
# Display Canny Edge Detection Image
# cv2.imshow('Canny Edge Detection', edges)

# cv2.waitKey(0)

# plt.hist(edges.ravel(), 256,[0, 256]); plt.show()


(T, bottle_threshold) = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Bottle Gray Threshold 27.5", bottle_threshold)
# cv2.waitKey(0)


contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
bottle_clone = bottle_3_channel.copy()

cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0), 2)
# cv2.imshow("All Contours", bottle_clone)
# cv2.waitKey(0)

areas = [cv2.contourArea(contour) for contour in contours]
(contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))

# print contour with largest area
bottle_clone = bottle_3_channel.copy()
# cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0), 2)
# cv2.imshow("Largest contour", bottle_clone)
# cv2.waitKey(0)

bottle_clone = bottle_3_channel.copy()
(x, y, w, h) = cv2.boundingRect(contours[-1])
aspectRatio = w / float(h)
print(aspectRatio)

if aspectRatio < 0.4:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "Full", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
elif aspectRatio > 0.75:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(bottle_clone, "Low", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
else:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "medium", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    
    
cv2.imshow("Decision", bottle_clone)
cv2.waitKey(0)  