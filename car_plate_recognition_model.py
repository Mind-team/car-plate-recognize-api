import cv2
import pytesseract
import re

# Set tesseract path to where the tesseract exe file is located (Edit this path accordingly based on your own settings)
pytesseract.pytesseract.tesseract_cmd = r'D:\Python\tesseract\tesseract.exe'

# Import Haar Cascade XML file for Russian car plate numbers
carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml')


# Function to retrieve only the car plate sub-image itself
def carplate_extract(image):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]

    return carplate_img


# Enlarge image for further image processing later on
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image


def get_tries(img):
    extract_img = carplate_extract(img)
    extract_img = enlarge_img(extract_img, 150)

    extract_img_gray = cv2.cvtColor(extract_img, cv2.COLOR_RGB2GRAY)

    extract_img_gray_blur = cv2.medianBlur(extract_img_gray,3)
    tries = ''

    for i in range(3, 14):
        tries += pytesseract.image_to_string(extract_img_gray_blur,
                                          config = f'--psm {i} --oem 3 '
                                                   f'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    return tries


def get_plate(img):
    carplate_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    carplate = re.search(r'[A-Z]\d\d\d[A-Z][A-Z]\d\d', get_tries(carplate_img_rgb)).group()
    return carplate or 'NaN'
