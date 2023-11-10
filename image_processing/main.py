import cv2

# Function for Grayscale Conversion
def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function for Color Space Conversion (RGB to HSV)
def convert_to_hsv(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

# Function for Image Blurring (Gaussian Blur)
def apply_gaussian_blur(image_path):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image

# Function for Edge Detection (Canny Edge Detector)
def detect_edges(image_path):
    gray_image = convert_to_grayscale(image_path)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

# Function for Image Thresholding
def apply_threshold(image_path):
    gray_image = convert_to_grayscale(image_path)
    _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    return thresholded_image

# Example usage of functions
image_path = 'input_image.jpg'

grayscale_image = convert_to_grayscale(image_path)
hsv_image = convert_to_hsv(image_path)
blurred_image = apply_gaussian_blur(image_path)
edges_image = detect_edges(image_path)
thresholded_image = apply_threshold(image_path)

# Displaying the processed images (you need to have OpenCV window open)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('HSV Image', hsv_image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Edges Image', edges_image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
