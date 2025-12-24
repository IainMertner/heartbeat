import pygame
import numpy as np

class SpeedSlider:
    def __init__(self, x, y, w, min_val, max_val, value):
        self.rect = pygame.Rect(x, y, w, 6)
        self.knob_r = 8
        self.min = min_val
        self.max = max_val
        self.value = value
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            x = np.clip(event.pos[0], self.rect.left, self.rect.right)
            t = (x - self.rect.left) / self.rect.width
            self.value = self.min + t * (self.max - self.min)

    def draw(self, screen):
        pygame.draw.rect(screen, (180, 180, 180), self.rect)
        t = (self.value - self.min) / (self.max - self.min)
        kx = int(self.rect.left + t * self.rect.width)
        ky = self.rect.centery
        pygame.draw.circle(screen, (60, 60, 60), (kx, ky), self.knob_r)
