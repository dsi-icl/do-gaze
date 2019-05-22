import pygame
import random
import sys

#sys.argv[1]

WIDTH = 800
HEIGHT = 600
WHITE = (255,255,255)
RED = (0,0,255)
BLUE = (255,0,0)

game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('FirstGame')
clock = pygame.time.Clock()

class FirstGame:
    def __init__(self, color):
        self.x = random.randrange(0, WIDTH)
        self.y = random.randrange(0, HEIGHT)
        self.size = random.randrange(4,8)
        self.color = color

    def move(self):
        self.move_x = random.randrange(-1, 2)
        self.move_y = random.randrange(-1, 2)
        self.x += self.move_x
        self.y += self.move_y

        if self.x < 0:
            self.x = 0
        elif self.x > WIDTH:
            self.x = WIDTH

        if self.y < 0:
            self.y = 0
        elif self.y > HEIGHT:
            self.y = HEIGHT

def draw_environment():
    game_display.fill(WHITE)
    draw_glob = pygame.draw_cirle()
    pygame.display.update()

'''
def style(srf=self, bg=None, border=(0,0,0), weight=0, padding=0):
    w, h = srf.get_size()

    #Add padding
    padding += weight
    w += 2 * padding
    h += 2 * padding
    img = pygame.Surface((w, h), pygame.SRCALPHA)

    #Add background color and border
    if bg:
        img.fill(rgba(bg))
    if weight:
        drawBorder(img, border, weight)

    return img
'''

def main():
    red_blob = FirstGame(RED)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        draw_environment()
        clock.tick(60)

if __name__ == '__main__':
    main()
