import numpy as np
import pygame
import sklearn.svm as svm

min_x = 0
max_x = 600
min_y = 0
max_y = 400


pygame.init()
display = pygame.display.set_mode((max_x, max_y))
display.fill((255, 255, 255))
pygame.display.update()

clock = pygame.time.Clock()
FPS = 60

svm_model = svm.SVC(kernel='linear', C=1.0)

points = []
clusters = []

def show_line():
    svm_model.fit(points, clusters)

    weights = svm_model.coef_[0]
    w = weights[0]
    b = weights[1]

    a = -w / b
    xx = np.linspace(0, max_x, 2)
    yy = a * xx - (svm_model.intercept_[0]) / b

    pygame.draw.line(display,
                     (0, 0, 0),
                     (xx[0], yy[0]),
                     (xx[len(xx) - 1], yy[len(yy) - 1]))

play = True
while play:
    for i in pygame.event.get():
        print(i)

        if i.type == pygame.QUIT:
            play = False
            pygame.quit()
            break

        # Рисуем точки
        if i.type == pygame.MOUSEBUTTONDOWN:
            def get_color(m):
                if m == 3:
                    return (255, 0, 0)
                else:
                    return (0, 255, 0)

            c = i.button
            if c not in [3, 1]:
                continue
            pygame.draw.circle(display, get_color(c), i.pos, 10)
            clusters.append(c)
            points.append((i.pos[0], i.pos[1]))

        elif i.type == pygame.KEYDOWN:

            if i.key == pygame.KSCAN_L or i.key == pygame.K_e:
                show_line()

            # Перезагрузка поля
            elif i.key == pygame.K_r:
                display.fill((255, 255, 255))
                points = []

            # Выход из программы
            elif i.key == pygame.K_q:
                play = False
                pygame.quit()
                break

    pygame.display.update()
    clock.tick(FPS)
