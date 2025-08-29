import cv2  # type: ignore
import pytesseract  # type: ignore


# load image
path = './assets/car_numberpad.png'
image = cv2.imread(path)

# do OCR
text = pytesseract.image_to_string(image, lang='kor+eng')

# show result
print("OCR Result = ", text)
cv2.imshow("original image", image)

cv2.waitKey(0)