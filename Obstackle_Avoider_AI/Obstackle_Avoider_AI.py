from distutils.log import FATAL
from operator import pos
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import random
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import pygame
from torch import rand

# Initializing Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))

Grey = (128, 128, 128) #Colors
Red = (255, 0, 0)
Black = (0, 0, 0)
White = (255, 255, 255)

position_pos = (100, 200, 300, 400) #options where player can move
Poslist = [0,0,0,0] #Enemy position 
Colided = False
landRec = pygame.Rect(0, 500, 600, 100) #Land

##########################################################################################################################################################################

def land():
    
    global landRec
    pygame.draw.rect(screen, Grey, pygame.Rect(0, 500, 600, 100))  # Land

class Enemy(object):
    heightY0 = 0
    heightY1 = 0
    heightY2 = 0
    heightY3 = 0
    
    def fall(posE):

            posE = posE
            pos0Rect = pygame.Rect(position_pos[posE], Enemy.heightY0, 100, 20)
            pos1Rect = pygame.Rect(position_pos[posE], Enemy.heightY1, 100, 60)
            pos2Rect = pygame.Rect(position_pos[posE], Enemy.heightY2, 100, 150)
            pos3Rect = pygame.Rect(position_pos[posE], Enemy.heightY3, 100, 80)

            global Poslist
            global Colided           

            StartR = 1
            EndE = 10 #speed range 40

            speed0 = random.randint(StartR,EndE) #Initializing
            speed1 = random.randint(StartR,EndE)
            speed2 = random.randint(StartR,EndE)
            speed3 = random.randint(StartR,EndE)    
         
            player = pygame.Rect(position_pos[posPlayer], 400, 100, 100)

            if posE == 0:

                if pos0Rect.colliderect(player):
                    Colided = True
                else: Colided = False


                if pos0Rect.colliderect(landRec): #Collision
                    Enemy.heightY0 = 0
                    speed0 = random.randint(StartR,EndE)
                    
                pygame.draw.rect(screen, Red, pos0Rect)
                
                Enemy.heightY0 += speed0
                Poslist[0] = Enemy.heightY0

   
            if posE == 1:
                
                if pos1Rect.colliderect(pygame.Rect(position_pos[posPlayer], 400, 100, 100)):
                    Colided = True
                else: Colided = False

                if pos1Rect.colliderect(landRec): #Collision
                    Enemy.heightY1 = 0
                    speed1 = random.randint(StartR,EndE)

                pygame.draw.rect(screen, Red, pos1Rect)
                Enemy.heightY1 += speed1
                Poslist[1] = Enemy.heightY1       

            if posE == 2:
                
                if pos2Rect.colliderect(pygame.Rect(position_pos[posPlayer], 400, 100, 100)):
                    Colided = True
                else: Colided = False

                pygame.draw.rect(screen, Red, pos2Rect)

                if pos2Rect.colliderect(landRec): #Collistion
                    Enemy.heightY2 = 0
                    speed2 = random.randint(StartR,EndE)

                Enemy.heightY2 += speed2
                Poslist[2] = Enemy.heightY2

            if posE == 3:
                
                if pos3Rect.colliderect(pygame.Rect(position_pos[posPlayer], 400, 100, 100)):
                    Colided = True
                else: Colided = False

                pygame.draw.rect(screen, Red, pos3Rect)

                if pos3Rect.colliderect(landRec): #Collision
                    Enemy.heightY3 = 0
                    speed3 = random.randint(StartR,EndE)

                Enemy.heightY3 += speed3
                Poslist[3] = Enemy.heightY3

            return posE

class Player:
    def position(posP):
        global posPlayer
        posPlayer = posP
        pygame.draw.rect(screen, Black, pygame.Rect(position_pos[posP], 490, 50, 10))

        return posP

        
##########################################################################################################################################################################

screen.fill((255, 255, 255))
Player.position(random.randint(0, 3))  

##########################################################################################################################################################################

#Buidling Env
class EscaperEnv(Env):
    def __init__(self):
        # Actions we can take 4
        self.action_space = Discrete(4)
        # color  array
        self.observation_space = MultiDiscrete([600,600,600,600])

        self.state = Poslist
        # Set length
        self.length = 60
        
    def step(self, action):
        # Apply action
        
        screen.fill((255, 255, 255))
        land()
        self.state = Poslist
        # Reduce  length by 1 second
        self.length -= 1 
        player = Player.position(action)
        x = Enemy.fall(0)
        x1 = Enemy.fall(1)
        x2 = Enemy.fall(2)
        x3 = Enemy.fall(3)

        # Calculate reward

        if Colided == True:
            reward = -2
        else: 
            reward = 1         
        # Check if it is done
        if self.length <= 0: 

            done = True
        else:
            done = False
        
        info = {}
        pygame.display.update()
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
       pass
    
    def reset(self):
        # Reset  
        Colided = False
      #  screen.fill((255, 255, 255))
        land()
        Enemy.heightY0 = 0 #Reset obstackle heigt
        Enemy.heightY1 = 0
        Enemy.heightY2 = 0
        Enemy.heightY3 = 0

        self.state = Poslist
        pygame.display.update()
        # Reset time
        self.length = 60 
        return self.state

##########################################################################################################################################################################
env = EscaperEnv()

log_path = os.path.join('Training', 'Logs') # Train
A2C_path = os.path.join('Training', 'Saved Models', 'Trained_Model_100K_Steps') # model path
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

#model = A2C.load(A2C_path, env)   #UnComment IF you Want To Train Again
#model.learn(total_timesteps=100000)   #learn  UnComment IF you Want To Train Again
#model.save(A2C_path) #UnComment IF you Want To Train Again

model = A2C.load(A2C_path, env)

episodes = 5    #Predict
for episode in range(1, episodes+1):  
    
    state = env.reset()
    done = False
    score = 0 
    clock=pygame.time.Clock()  

    while not done:
        clock.tick(50) #Fps
        
        x = Enemy.fall(0)
        x1 = Enemy.fall(1)
        x2 = Enemy.fall(2)
        x3 = Enemy.fall(3)
        action, _states = model.predict(state)
        state, reward, done, info = env.step(action)
       # print(Poslist) #Observation space / obstackle Place
        
        score+=reward
         
    print('Episode:{} Score:{}'.format(episode, score))
    
env.close()