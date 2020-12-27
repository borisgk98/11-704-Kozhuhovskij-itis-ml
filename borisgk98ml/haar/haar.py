from cv2 import cv2

haarcascade_frontalface_default_xml = "haarcascade_frontalface_default.xml"
image_path = "test.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
model = cv2.CascadeClassifier(haarcascade_frontalface_default_xml)
# ищем лица
faces = model.detectMultiScale(image)

# для каждого лица рисуем квадрат
color = (255, 0, 0)
for face in faces:
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

cv2.imshow('Test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()