import cv2

# Load the pre-trained Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image or video stream
image = cv2.imread('image.jpg')
# Alternatively, you can use a video stream by initializing a VideoCapture object

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()