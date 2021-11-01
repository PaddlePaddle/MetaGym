import pygame
import gym
import metagym.navigator2d

def movement_control(keys):
    #Keyboard control cases
    if keys[pygame.K_LEFT]:
        return (1, 0)
    if keys[pygame.K_RIGHT]:
        return (0, 1)
    if keys[pygame.K_UP]:
        return (1, 1)
    if keys[pygame.K_DOWN]:
        return (-1, -1)
    return (0, 0)

if __name__=="__main__":
    # running random policies
    game = gym.make("navigator-wr-2D-v0", max_steps=5000)
    game.set_task(game.sample_task())
    game.reset()
    done = False
    sum_reward = 0.0
    while not done:
        pygame.time.delay(50)
        keys = pygame.key.get_pressed()
        action = movement_control(keys)
        obs, r, done, info = game.step(action)
        sum_reward += r
        game.render()
        print(obs, info)
    print("episode is over, sum reward: %f" % sum_reward)
