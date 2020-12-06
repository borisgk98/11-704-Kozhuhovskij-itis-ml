import pygame
from PIL import Image as PILImage
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('mnist.h5')


def convert_image(image):
    # датасет содержит картинки размером 28х28, поэтому делаем resize
    image = image.resize((28, 28))
    # делаем картинку черно-белой
    image = image.convert('L')
    image = np.array(image)
    # преобразование изображения в формат модели
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    return image


def predict_digit(data):
    result = model.predict([data])[0]
    maxp = 0
    num = 0
    for i in range(len(result)):
        if result[i] > maxp:
            maxp = result[i]
            num = i
    print('Prediction: ', num)


def get_pil_image(screen):
    return PILImage.frombytes("RGBA", sz, bytes(pygame.image.tostring(screen, "RGBA")))


def test_image(screen):
    image = get_pil_image(screen)
    # for test
    # image.save("num.png")
    predict_digit(convert_image(image))


sz = (200, 200)
screen = pygame.display.set_mode(sz)
last_pos = None
drawing = False
background = pygame.color.Color('Black')
pen = pygame.color.Color('White')
screen.fill(background)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            pygame.sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            if (drawing):
                mouse_position = pygame.mouse.get_pos()
                if last_pos is not None:
                    pygame.draw.line(screen, pen, last_pos, mouse_position, 10)
                last_pos = mouse_position
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.KSCAN_L or event.key == pygame.K_e:
                test_image(screen)
            elif event.key == pygame.K_r:
                screen.fill(background)
                drawing = False
                last_pos = None
            elif event.key == pygame.K_q:
                pygame.quit()
                pygame.sys.exit()

    pygame.display.update()