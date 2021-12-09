# KidsCanCode - Game Development with Pygame video series
# Shmup game - part 14
# Video link: https://www.youtube.com/watch?v=Z2K2Yttvr5g
# Game Over Screen
# Frozen Jam by tgfcoder <https://twitter.com/tgfcoder> licensed under CC-BY-3
# Art from Kenney.nl

# Modified version of the kidscancode shmup with a reinforcement learning model

#original code can be found at: https://github.com/kidscancode/pygame_tutorials/blob/master/shmup/shmup-14.py

from numpy.lib.ufunclike import _deprecate_out_named_y
import pygame
import random
from os import path, makedirs
import torch
import torch.nn as nn
import torch.optim as optimal
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from IPython import display
from datetime import datetime
from numpy import savetxt

MAX_MEMORY = 100000
EPOCH_SIZE = 1000
LR = 0.001

img_dir = path.join(path.dirname(__file__), 'img')
snd_dir = path.join(path.dirname(__file__), 'snd')

WIDTH = 480
HEIGHT = 600
FPS = 600
POWERUP_TIME = 5000

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

today = datetime.now()

poss_states = [[min_x, min_y, max_x, max_y, shoot_pow, pow_type] for min_x in range(-1, 2) for min_y in range (-1, 2) for max_x in range(-1, 2) for max_y in range (-1, 2) for  shoot_pow in range(0, 2) for pow_type in range(0,2)]

# initialize pygame and create window
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shmup!")
clock = pygame.time.Clock()

plt.ion()

font_name = pygame.font.match_font('arial')
def draw_text(surf, text, size, x, y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

# new mob adjusted to take in action from model
def newmob(action): 
    m = Mob(action)
    all_sprites.add(m)
    mobs.add(m)

def draw_shield_bar(surf, x, y, pct):
    if pct < 0:
        pct = 0
    BAR_LENGTH = 100
    BAR_HEIGHT = 10
    fill = (pct / 100) * BAR_LENGTH
    outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
    fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
    pygame.draw.rect(surf, GREEN, fill_rect)
    pygame.draw.rect(surf, WHITE, outline_rect, 2)

def draw_lives(surf, x, y, lives, img):
    for i in range(lives):
        img_rect = img.get_rect()
        img_rect.x = x + 30 * i
        img_rect.y = y
        surf.blit(img, img_rect)

# Helper method to plot the data
def plot(sps, avg_sps):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Improvemnt in Score per Second, per Game")
    plt.xlabel("Number of Games Played")
    plt.ylabel("Score per Second")
    plt.plot(sps, label = "Score per Second for Game N (SPS)")
    plt.plot(avg_sps, label = "Average SPS")
    plt.ylim(ymin = 0)
    plt.legend()

# Helper method to save the plots as well as the individual data points
def save_plot(avg_sps, sps):
    data_folder = "./data/plots/RL"
    
    date_name = today.strftime("%d_%m_%H_%M_%S")
    plot_name = "plot_" + date_name + ".png"
    avg_score_name = "avgsps_" + date_name + ".csv"
    score_name = "sps_" + date_name + ".csv"
    folder_name = "game_" + date_name

    data_folder = path.join(data_folder, folder_name)
    if not path.exists(data_folder):
        makedirs(data_folder)


    file_name1 = path.join(data_folder, plot_name)
    file_name2 = path.join(data_folder, avg_score_name) 
    file_name3 = path.join(data_folder, score_name)

    plt.savefig(file_name1)
    savetxt(file_name2, sps, delimiter=",")
    savetxt(file_name3, avg_sps, delimiter=",")

# Helper method ensures that the speed vectors for the mob never go out of their limits
def checkBoundXY(action, min_x, max_x, min_y, max_y):
    if min_x + action[0] < 1:
        min_x = 1
    elif min_x + action[0] > 7:
        min_x = 7
    else:
        min_x += action[0]

    if max_x + action[1] < 2:
        max_x  = 2
    elif max_x  + action[1] > 8:
        max_x  = 8
    else:
        max_x  += action[1]

    if min_y + action[2] < 1:
        min_y = 1
    elif min_y + action[2] > 7:
        min_y = 7
    else:
        min_y += action[2]

    if max_y + action[3] < 2:
        max_y = 2
    elif max_y + action[3] > 8:
        max_y = 8
    else:
        max_y += action[3]

    if min_x > max_x:
        temp = max_x
        max_x = min_x
        min_x = temp
    elif min_x == max_x:
        max_x += 1

    if min_y > max_y:
        temp = max_y
        max_y = min_y
        min_x = temp
    elif min_y == max_y:
        max_y += 1

# Agent class holds the ability to get the state of the game as well as returns new acitons for the mob and contorls the actions of the model and the trainer
class Agent:

    def __init__(self):
        self.exploration = 0
        self.gamma = 0.85
        self.l_rate = 0.001
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = DeepQ(3, 256, 324)
        self.trainer = Trainer(self.model, self.l_rate, self.gamma)

    def get_state(self):
        state = []
        state = [player.shield, player.rect.centerx, player.direction]
        return state


    def save(self, state, action, reward, next, games):
        self.memory.append((state, action, reward, next, games))

    def long_term_train(self):
        if len(self.memory) > EPOCH_SIZE:
            sample = random.sample(self.memory, EPOCH_SIZE)
        else:
            sample = self.memory
        
        states, actions, rewards, next_states, games = zip(*sample)
        self.trainer.train(states, actions, rewards, next_states, games)
        
    def short_term_train(self, state, action, reward, next, games):
        self.trainer.train(state, action, reward, next, games)

    def get_action(self, state):
        self.exploration = 50 - num_games
        final_move = []
        if random.randint(0, 150) < self.exploration:
            final_move = [random.randint(-1, 2), random.randint(-1, 2), random.randint(-1, 2), random.randint(-1, 2), random.randint(0, 2), random.randint(0, 2)]
        else:
            state_naught = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state_naught)
            move = torch.argmax(prediction).item()
            final_move = poss_states[move]
        return final_move

# Contains the forward function 
class DeepQ(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save_results(self, file_name="data.pth"):
        data_folder = "./data"
        if not path.exists(data_folder):
            makedirs(data_folder)
        
        file_name = path.join(data_folder, file_name)
        torch.save(self.state_dict(), file_name)

#  The training alorithm 
class Trainer:
    def __init__(self, model, l_rate, gamma):
        self.l_rate = l_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optimal.Adam(model.parameters(), lr = self.l_rate)
        self.loss_function = nn.MSELoss()

    def train(self, state, action, reward, next, game_over):
        state = torch.tensor(state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)
        next = torch.tensor(next, dtype = torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next = torch.unsqueeze(next, 0)
            game_over = (game_over, )

        predict_Q = self.model(state)

        target = predict_Q.clone()
        for i in range(len(game_over)):
            new_prediction = reward[i]
            if not game_over[i]:
                # Q learning rule
                new_prediction = reward[i] + self.gamma * torch.max(self.model(next[i]))
            
            target[i][torch.argmax(action[i]).item()] = new_prediction
        
        # Set gradients to zero
        self.optimizer.zero_grad()
        # Call loss function
        loss = self.loss_function(predict_Q, target)
        # Call back propigation
        loss.backward()
        # Have optimizer take a step
        self.optimizer.step()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(player_img, (50, 38))
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.radius = 20
        # pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 0
        self.shield = 100
        self.shoot_delay = 250
        self.last_shot = pygame.time.get_ticks()
        self.lives = 1
        self.hidden = False
        self.hide_timer = pygame.time.get_ticks()
        self.power = 1
        self.power_time = pygame.time.get_ticks()
        self.speedx = -8
        self.direction = -1
        self.fire_shot = 0
        self.shot_interval = random.randint(1, 4)

    def update(self):
        # timeout for powerups
        if self.power >= 2 and pygame.time.get_ticks() - self.power_time > POWERUP_TIME:
            self.power -= 1
            self.power_time = pygame.time.get_ticks()

        # unhide if hidden
        if self.hidden and pygame.time.get_ticks() - self.hide_timer > 1000:
            self.hidden = False
            self.rect.centerx = WIDTH / 2
            self.rect.bottom = HEIGHT - 10

        
        #keystate = pygame.key.get_pressed()
        #if keystate[pygame.K_LEFT]:
        #   self.speedx = -8
        #if keystate[pygame.K_RIGHT]:
        #   self.speedx = 8
        #if keystate[pygame.K_SPACE]:
        #   self.shoot()
        
        # Bot functionality
        if self.rect.right >= WIDTH:
            self.speedx *= -1
            self.direction *= -1
        if self.rect.left <= 0:
            self.speedx *= -1
            self.direction *= -1

        self.rect.x += self.speedx
        if self.fire_shot % self.shot_interval == 0:
            self.shoot()
        
        self.fire_shot += 1

    def powerup(self):
        self.power += 1
        self.power_time = pygame.time.get_ticks()

    def shoot(self):
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shoot_delay:
            self.last_shot = now
            if self.power == 1:
                bullet = Bullet(self.rect.centerx, self.rect.top)
                all_sprites.add(bullet)
                bullets.add(bullet)
            if self.power >= 2:
                bullet1 = Bullet(self.rect.left, self.rect.centery)
                bullet2 = Bullet(self.rect.right, self.rect.centery)
                all_sprites.add(bullet1)
                all_sprites.add(bullet2)
                bullets.add(bullet1)
                bullets.add(bullet2)

    def hide(self):
        # hide the player temporarily
        self.hidden = True
        self.hide_timer = pygame.time.get_ticks()
        self.rect.center = (WIDTH / 2, HEIGHT + 200)

class Mob(pygame.sprite.Sprite):
    def __init__(self, action):
        self.speed_y_min = min_y
        self.speed_y_max = max_y
        self.speed_x_min = min_x
        self.speed_x_max = max_y

        self.speed_x = random.randint(self.speed_x_min, self.speed_x_max)
        self.speed_y = random.randint(self.speed_y_min, self.speed_y_max)


        pygame.sprite.Sprite.__init__(self)
        self.image_orig = random.choice(meteor_images)
        self.image_orig.set_colorkey(BLACK)
        self.image = self.image_orig.copy()
        self.rect = self.image.get_rect()
        self.radius = int(self.rect.width * .85 / 2)
        # pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.x = random.randrange(WIDTH - self.rect.width)
        self.rect.bottom = random.randrange(-80, -20)
        self.rot = 0
        self.rot_speed = random.randrange(-8, 8)
        self.last_update = pygame.time.get_ticks()

    def rotate(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > 50:
            self.last_update = now
            self.rot = (self.rot + self.rot_speed) % 360
            new_image = pygame.transform.rotate(self.image_orig, self.rot)
            old_center = self.rect.center
            self.image = new_image
            self.rect = self.image.get_rect()
            self.rect.center = old_center

    def update(self):
        self.rotate()
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        if self.rect.top > HEIGHT + 10 or self.rect.left < -100 or self.rect.right > WIDTH + 100:
            self.rect.x = random.randrange(WIDTH - self.rect.width)
            self.rect.y = random.randrange(-100, -40)

            self.speed_y = random.randint(self.speed_y_min, self.speed_y_max)
            self.speed_x = random.randint(self.speed_x_min, self.speed_x_max)

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speedy = -10

    def update(self):
        self.rect.y += self.speedy
        # kill if it moves off the top of the screen
        if self.rect.bottom < 0:
            self.kill()

class Pow(pygame.sprite.Sprite):
    def __init__(self, center, type):
        pygame.sprite.Sprite.__init__(self)
        if type:
            self.type = 'shield'
        else:
            self.type = 'gun'
        self.image = powerup_images[self.type]
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.speedy = 5

    def update(self):
        self.rect.y += self.speedy
        # kill if it moves off the top of the screen
        if self.rect.top > HEIGHT:
            self.kill()

class Explosion(pygame.sprite.Sprite):
    def __init__(self, center, size):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.image = explosion_anim[self.size][0]
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 75

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame += 1
            if self.frame == len(explosion_anim[self.size]):
                self.kill()
            else:
                center = self.rect.center
                self.image = explosion_anim[self.size][self.frame]
                self.rect = self.image.get_rect()
                self.rect.center = center

def show_go_screen():
    screen.blit(background, background_rect)
    draw_text(screen, "SHMUP!", 64, WIDTH / 2, HEIGHT / 4)
    draw_text(screen, "Arrow keys move, Space to fire", 22,
              WIDTH / 2, HEIGHT / 2)
    draw_text(screen, "Press a key to begin", 18, WIDTH / 2, HEIGHT * 3 / 4)
    pygame.display.flip()
    waiting = True
    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYUP:
                waiting = False

# Load all game graphics
background = pygame.image.load(path.join(img_dir, "starfield.png")).convert()
background_rect = background.get_rect()
player_img = pygame.image.load(path.join(img_dir, "playerShip1_orange.png")).convert()
player_mini_img = pygame.transform.scale(player_img, (25, 19))
player_mini_img.set_colorkey(BLACK)
bullet_img = pygame.image.load(path.join(img_dir, "laserRed16.png")).convert()
meteor_images = []
meteor_list = ['meteorBrown_big1.png', 'meteorBrown_med1.png', 'meteorBrown_med1.png',
               'meteorBrown_med3.png', 'meteorBrown_small1.png', 'meteorBrown_small2.png',
               'meteorBrown_tiny1.png']
for img in meteor_list:
    meteor_images.append(pygame.image.load(path.join(img_dir, img)).convert())
explosion_anim = {}
explosion_anim['lg'] = []
explosion_anim['sm'] = []
explosion_anim['player'] = []
for i in range(9):
    filename = 'regularExplosion0{}.png'.format(i)
    img = pygame.image.load(path.join(img_dir, filename)).convert()
    img.set_colorkey(BLACK)
    img_lg = pygame.transform.scale(img, (75, 75))
    explosion_anim['lg'].append(img_lg)
    img_sm = pygame.transform.scale(img, (32, 32))
    explosion_anim['sm'].append(img_sm)
    filename = 'sonicExplosion0{}.png'.format(i)
    img = pygame.image.load(path.join(img_dir, filename)).convert()
    img.set_colorkey(BLACK)
    explosion_anim['player'].append(img)
powerup_images = {}
powerup_images['shield'] = pygame.image.load(path.join(img_dir, 'shield_gold.png')).convert()
powerup_images['gun'] = pygame.image.load(path.join(img_dir, 'bolt_gold.png')).convert()

# Game loop
game_over = True
running = True
reward = 0
frames = 0
seconds = 0
num_games = -1
scoreperseconds = []
avg_scoresps = []
total_scoreps = 0
record = 0
action = None
player = None
min_x = -3
max_x = 3
min_y = 1
max_y = 8
state_old = []
num_mobs = 8
since_last_hit = 0

agent = Agent()

while running:

    frames += 1
    seconds = (frames /  60.0)

    if game_over or seconds > 180:
        num_games += 1
        reward = 0
        game_over = False
        all_sprites = pygame.sprite.Group()
        mobs = pygame.sprite.Group()
        bullets = pygame.sprite.Group()
        powerups = pygame.sprite.Group()
        player = Player()
        all_sprites.add(player)
        for i in range(8):
            newmob([0, 0, 0, 0])
        
        if num_games > 0:
            agent.long_term_train()

            if score/seconds > record:
                record = score/seconds
                agent.model.save_results()
            
            scoreperseconds.append(score/seconds)
            total_scoreps += score/seconds
            avg_score = total_scoreps/num_games
            avg_scoresps.append(avg_score)

            plot(scoreperseconds, avg_scoresps)

        score = 0
        frames = 0
        seconds = 0
        since_last_hit = 0

        

    # keep loop running at the right speed
    clock.tick(FPS)
    # Process input (events)
    for event in pygame.event.get():
        # check for closing window
        if event.type == pygame.QUIT:
            save_plot(scoreperseconds, avg_scoresps)
            running = False

    # Update
    all_sprites.update()

    state_old = agent.get_state()

    action = agent.get_action(state_old)
    checkBoundXY(action, min_x, max_x, min_y, max_y)
    


    # check to see if a bullet hit a mob
    hits = pygame.sprite.groupcollide(mobs, bullets, True, True)
    for hit in hits:
        if score/seconds > 1.0:
            reward = 1 * (score / seconds)
        else:
            if score > 0:
                reward = -1 * (seconds / score)
            else:
                reward = -100
        score += 50 - hit.radius
        expl = Explosion(hit.rect.center, 'lg')
        all_sprites.add(expl)
        if action[4]:
            pow = Pow(hit.rect.center, action[5])
            all_sprites.add(pow)
            powerups.add(pow)
        
        newmob(action)

    # check to see if a mob hit the player
    hits = pygame.sprite.spritecollide(player, mobs, True, pygame.sprite.collide_circle)
    since_last_hit = seconds
    for hit in hits:
        player.shield -= hit.radius * 2
        expl = Explosion(hit.rect.center, 'sm')
        all_sprites.add(expl)

        newmob(action)

        if player.shield <= 0:
            death_explosion = Explosion(player.rect.center, 'player')
            all_sprites.add(death_explosion)
            player.hide()
            player.lives -= 1
            player.shield = 100
            player_shield = 100

    # check to see if player hit a powerup
    hits = pygame.sprite.spritecollide(player, powerups, True)
    for hit in hits:
        if hit.type == 'shield':
            player.shield += random.randrange(10, 30)
            if player.shield >= 100:
                player.shield = 100
        if hit.type == 'gun':
            player.powerup()

    # if the player died and the explosion has finished playing
    if player.lives == 0 and not death_explosion.alive():
        game_over = True

    # Draw / render
    screen.fill(BLACK)
    screen.blit(background, background_rect)
    all_sprites.draw(screen)
    draw_text(screen, str(score), 18, WIDTH / 2, 10)
    draw_shield_bar(screen, 5, 5, player.shield)
    draw_lives(screen, WIDTH - 100, 5, player.lives, player_mini_img)
    # *after* drawing everything, flip the display
    pygame.display.flip()

    next_state = agent.get_state()

    agent.short_term_train(state_old, action, reward, next_state, game_over)
    agent.save(state_old, action, reward, next_state, game_over)


pygame.quit()
save_plot(scoreperseconds, avg_scoresps)
