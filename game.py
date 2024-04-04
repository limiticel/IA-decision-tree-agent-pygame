from modulo_ia import IA_class
import pygame
from pygame.locals import *
import sys
import os
import random
from time import sleep

# Configurações do ambiente
WIDTH_SCREEN = 600
HEIGTH_SCREEN = 100
BACKGROUND_COLOR = (255, 255, 255)
GRAVITY = 0.5

# Inicialização da janela do Pygame
SCREEN = pygame.display.set_mode((WIDTH_SCREEN, HEIGTH_SCREEN))
fps = pygame.time.Clock()
pygame.init()

# Texto de apresentação do agente
text_brain = 'Olá, me chamo Rick e meu objetivo é alcançar o bloco azul. (Sklearn)'
font = pygame.font.Font(None, 16)

# Pontuação inicial e configuração do tempo
punt = 0
time = 240
time_m = time
sec = 0

# Posição inicial do objeto objetivo
posx_object = WIDTH_SCREEN - 35
act = ''

# Inicialização do agente de IA
Agent_ia = IA_class([0, HEIGTH_SCREEN - 20], 0, 0.1, 0, 0)

while True:
    time_m += 1
    sec = int(time_m / time)
    SCREEN.fill(BACKGROUND_COLOR)
    fps.tick(time)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Desenho do objeto objetivo
    object_ = pygame.draw.rect(SCREEN, (0, 0, 255), (posx_object, HEIGTH_SCREEN - 20, 20, 20))
    
    # Atualização dos sensores do agente
    Agent_ia.sensor_(2, WIDTH_SCREEN, HEIGTH_SCREEN, SCREEN)
    Agent_ia.sensor_2(2, WIDTH_SCREEN, HEIGTH_SCREEN, SCREEN)
    act = Agent_ia.parse()

    try:
        act = Agent_ia.take_d_(Agent_ia.model)
    except:
        pass
    
    # Desenho e movimento do agente
    agent = Agent_ia.indexing(SCREEN, HEIGTH_SCREEN)
    if object_.colliderect(agent):
        posx_object = random.randint(0, WIDTH_SCREEN - 35)
        punt += 1

    if act == 'left':
        Agent_ia.state[0] -= Agent_ia.SPEED
    if act == 'right':
        Agent_ia.state[0] += Agent_ia.SPEED
    
    Agent_ia.limits(WIDTH_SCREEN)
    dis = Agent_ia.cal_dist_()

    # Atualização do texto exibido na tela
    text_punt = f'{punt}-{sec}'
    text_render = font.render(text_brain, True, (0, 0, 0))
    text_render_punt = font.render(text_punt, True, (0, 0, 0))

    # Posicionamento do texto na tela
    text_rect = text_render.get_rect()
    text_rect_punt = text_render_punt.get_rect()
    text_brain = Agent_ia.text_brain_set(text_brain)

    Agent_ia.update_memory(act, dis)

    SCREEN.blit(text_render, text_rect)
    text_rect_punt.center = (480, 10)
    SCREEN.blit(text_render_punt, text_rect_punt)
    Agent_ia.stagnation(sec,punt,act,SCREEN,HEIGTH_SCREEN)
    # Atualização da tela
    pygame.display.flip()
