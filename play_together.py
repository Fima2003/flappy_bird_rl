import pygame
import numpy as np
import cv2
import os
from stable_baselines3 import DQN
import random
from flappy_bird import FlappyBird

# Constants
GAME_WIDTH = 288
GAME_HEIGHT = 512
FPS = 18

# Model Paths
MODEL_PATH = "models/DQN_FlappyBird/flappy_model_720000_steps"
# MODEL_PATH = "models/DQN_FlappyBird/flappy_model_690000_steps"


class FrameStackWrapper:
    """Manually handles frame stacking for the AI model input"""

    def __init__(self, k=4):
        self.k = k
        self.frames = []

    def reset(self, obs):
        self.frames = [obs for _ in range(self.k)]
        return self._get_obs()

    def step(self, obs):
        self.frames.append(obs)
        if len(self.frames) > self.k:
            self.frames.pop(0)
        return self._get_obs()

    def _get_obs(self):
        # Stack frames along the last axis: (84, 84, 4)
        return np.concatenate(self.frames, axis=-1)


def preprocess_frame(screen_pixels):
    """
    Convert (H, W, 3) -> (W, H, 3) (if needed by game) -> (84, 84, 1) Grayscale
    The game returns (W, H, 3) from get_screen_pixels, but let's check.
    FlappyBird.get_screen_pixels() returns pygame.surfarray.array3d(screen) -> (W, H, 3).
    WE NEED: (H, W, 3) for cv2 processing usually, but let's see what test.py did.
    test.py did: np.transpose(pixels, axes=(1, 0, 2)) in env.get_state()
    then cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    then resize to (84, 84)
    then expand_dims to (84, 84, 1)
    """
    # 1. Transpose: (W, H, C) -> (H, W, C)
    # Pygame uses (Width, Height, Channels), Numpy/CV2 expects (Height, Width, Channels)
    frame = np.transpose(screen_pixels, (1, 0, 2))

    # 2. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 3. Resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # 4. Add Channel dim: (84, 84) -> (84, 84, 1)
    return np.expand_dims(resized, axis=-1)


def draw_text_centered(surface, text, font, color, y_offset=0, bg_color=None):
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(
        center=(surface.get_width() // 2, surface.get_height() // 2 + y_offset)
    )
    if bg_color:
        pygame.draw.rect(surface, bg_color, text_rect.inflate(20, 10))
    surface.blit(text_surf, text_rect)
    return text_rect


def main():
    os.environ["SDL_VIDEODRIVER"] = "cocoa"  # Ensure we use a windowed driver on Mac
    pygame.init()

    # Setup Main Window
    screen = pygame.display.set_mode((GAME_WIDTH * 2, GAME_HEIGHT))
    pygame.display.set_caption("AI vs Human - Flappy Bird")

    # Sub-surfaces
    left_surface = screen.subsurface(pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT))
    right_surface = screen.subsurface(
        pygame.Rect(GAME_WIDTH, 0, GAME_WIDTH, GAME_HEIGHT)
    )

    # Font
    font = pygame.font.Font(None, 36)
    big_font = pygame.font.Font(None, 72)

    # Initialize Games
    # Left = AI, Right = Human
    # 1. Generate a common seed for startup
    start_seed = random.randint(0, 1000000)
    print(f"Initializing with seed: {start_seed}")

    ai_game = FlappyBird(player="AI", surface=left_surface, seed=start_seed)
    human_game = FlappyBird(player="human", surface=right_surface, seed=start_seed)

    # Load AI Model
    print("Loading AI Model...")
    try:
        model = DQN.load(MODEL_PATH)
        print("Model Loaded!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # AI State Management
    ai_stack = FrameStackWrapper(k=4)

    clock = pygame.time.Clock()

    # Game States
    STATE_START = 0
    STATE_PLAYING = 1
    STATE_GAMEOVER = 2

    current_state = STATE_START

    # Status flags
    ai_dead = False
    human_dead = False
    ai_score = 0
    human_score = 0

    def reset_games():
        nonlocal ai_dead, human_dead, ai_score, human_score
        ai_dead = False
        human_dead = False
        ai_score = 0
        human_score = 0

        # Pick new common seed
        new_seed = random.randint(0, 1000000)
        print(f"Resetting with seed: {new_seed}")

        ai_game.reset(seed=new_seed)
        human_game.reset(seed=new_seed)

        # Initialize AI observation (Must render first to get the reset state)
        ai_game.render()
        human_game.render()

        pixels = ai_game.get_screen_pixels()
        obs = preprocess_frame(pixels)
        ai_cur_obs_stacked = ai_stack.reset(obs)
        return ai_cur_obs_stacked

    ai_obs_stacked = reset_games()

    running = True
    ai_frame_counter = 0

    while running:
        # 1. Event Handling
        flap_action = False
        start_game_trigger = False
        restart_trigger = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    if current_state == STATE_START:
                        start_game_trigger = True
                    elif current_state == STATE_PLAYING and not human_dead:
                        flap_action = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                if current_state == STATE_GAMEOVER:
                    # Simple check for restart - any click for now, or check generic button area
                    # We will draw a button, but let's just make any click restart for simplicity first
                    restart_trigger = True
                    # If we decide to use a button rect logic, we can do collision checks here

        # 2. State Logic
        if current_state == STATE_START:
            # Render Games (Static)
            ai_game.render()
            human_game.render()

            # Draw UI
            pygame.draw.line(
                screen, (255, 255, 255), (GAME_WIDTH, 0), (GAME_WIDTH, GAME_HEIGHT), 5
            )
            draw_text_centered(left_surface, "AI BOT", font, (255, 0, 0), y_offset=-100)
            draw_text_centered(right_surface, "HUMAN", font, (0, 0, 255), y_offset=-100)

            # Start Button
            btn_rect = draw_text_centered(
                right_surface,
                "PRESS SPACE",
                big_font,
                (255, 255, 255),
                bg_color=(0, 0, 0, 100),
            )
            draw_text_centered(
                right_surface,
                "TO START",
                big_font,
                (255, 255, 255),
                y_offset=60,
                bg_color=(0, 0, 0, 100),
            )

            if start_game_trigger:
                current_state = STATE_PLAYING
                reset_games()

        elif current_state == STATE_PLAYING:
            # --- PREDICT AI ACTION ---
            if not ai_dead:
                action, _ = model.predict(ai_obs_stacked, deterministic=True)
                ai_should_flap = action == 1
            else:
                ai_should_flap = False

            # --- PHYSICS STEP (Repeated 4 times to match Env Wrapper) ---
            # This aligns the "Speed" and "Reaction Time" with training/test.py
            for i in range(4):

                # AI Logic: Only flap on the first sub-frame
                if i == 0 and ai_should_flap:
                    actual_ai_flap = True
                else:
                    actual_ai_flap = False

                # Human Logic: Only flap on the first sub-frame (if pressed)
                if i == 0 and flap_action:
                    actual_human_flap = True
                else:
                    actual_human_flap = False

                # 1. Step AI
                if not ai_dead:
                    ai_done, ai_passed = ai_game.step(actual_ai_flap)
                    if ai_passed:
                        ai_score += 1
                    if ai_done:
                        ai_dead = True

                # 2. Step Human
                if not human_dead:
                    h_done, h_passed = human_game.step(actual_human_flap)
                    if h_passed:
                        human_score += 1
                    if h_done:
                        human_dead = True

                # Break early if both dead? No, keep running to finish the frame usually, but logic holds.

            # --- RENDER ---
            # Render background and sprites (clears previous UI text)
            ai_game.render()
            human_game.render()

            # --- CAPTURE AI OBSERVATION ---
            # Capture once per render frame (every 4 physics steps)
            if not ai_dead:
                pixels = ai_game.get_screen_pixels()
                obs = preprocess_frame(pixels)
                ai_obs_stacked = ai_stack.step(obs)

            # Draw Separator
            pygame.draw.line(
                screen, (255, 255, 255), (GAME_WIDTH, 0), (GAME_WIDTH, GAME_HEIGHT), 5
            )

            # Draw Scores (Show Human Score Only)
            draw_text_centered(
                right_surface,
                f"Score: {human_score}",
                font,
                (255, 255, 255),
                y_offset=-200,
            )

            # Draw "Dead" labels
            if ai_dead:
                draw_text_centered(left_surface, "AI LOST", big_font, (255, 0, 0))
            if human_dead:
                draw_text_centered(right_surface, "YOU LOST", big_font, (255, 0, 0))

            # Check Game Over trigger
            if ai_dead and human_dead:
                current_state = STATE_GAMEOVER

        elif current_state == STATE_GAMEOVER:
            # Keep Rendering the frozen state (or continue rendering to keep the images there)
            # Since we solve logic in playing, we don't step physics here, just draw
            ai_game.render()
            human_game.render()

            pygame.draw.line(
                screen, (255, 255, 255), (GAME_WIDTH, 0), (GAME_WIDTH, GAME_HEIGHT), 5
            )

            # Draw Scores
            draw_text_centered(
                left_surface,
                f"Final: {ai_score}",
                big_font,
                (255, 255, 255),
                y_offset=-100,
            )
            draw_text_centered(
                right_surface,
                f"Final: {human_score}",
                big_font,
                (255, 255, 255),
                y_offset=-100,
            )

            # Restart Button (Spanning both or just center?)
            # Let's put a big button in the middle
            center_x = GAME_WIDTH  # Middle of the big screen
            center_y = GAME_HEIGHT // 2 + 50

            restart_text = big_font.render("RESTART", True, (0, 0, 0))
            restart_bg = pygame.Rect(0, 0, 300, 80)
            restart_bg.center = (center_x, center_y)

            pygame.draw.rect(screen, (0, 255, 0), restart_bg)
            pygame.draw.rect(screen, (255, 255, 255), restart_bg, 3)

            restart_text_rect = restart_text.get_rect(center=restart_bg.center)
            screen.blit(restart_text, restart_text_rect)

            # Mouse interaction
            if restart_trigger:
                # Check collision if we want specific button click
                mx, my = pygame.mouse.get_pos()
                if restart_bg.collidepoint(mx, my):
                    reset_games()
                    current_state = STATE_START

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
