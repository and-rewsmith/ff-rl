import gymnasium as gym
from minigrid.core.world_object import Goal
from minigrid.wrappers import RGBImgObsWrapper
import pygame
import numpy as np
from typing import Tuple, Any, Dict

def create_env() -> gym.Env:
    # Create the environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="rgb_array")
    # Wrap it with RGB image observation wrapper
    env = RGBImgObsWrapper(env)
    return env

def main() -> None:
    env = create_env()
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Game loop
    running: bool = True
    while running:
        # Get the rendered frame
        frame = env.render()
        
        # Display the frame using pygame
        if 'window' not in locals():
            pygame.init()
            window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption('MiniGrid')
        
        # Convert numpy array to pygame surface and display
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Process keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_LEFT:
                    obs, reward, terminated, truncated, _ = env.step(0)  # Turn left
                elif event.key == pygame.K_RIGHT:
                    obs, reward, terminated, truncated, _ = env.step(1)  # Turn right
                elif event.key == pygame.K_UP:
                    obs, reward, terminated, truncated, _ = env.step(2)  # Move forward
                
                if terminated or truncated:
                    obs, _ = env.reset()
        
        pygame.time.wait(50)  # Small delay to control game speed
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()