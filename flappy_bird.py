import pygame
import random
import os


class FlappyBird:
    def __init__(self, player: str = "human", surface=None, seed=None):
        if player == "AI":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        self.rng = random.Random(seed)

        # 1. Load the Background RAW (No .convert yet)
        try:
            # Just load it to read dimensions
            raw_bg = pygame.image.load("assets/bg.png")
        except pygame.error as e:
            print(f"Error loading assets: {e}")
            import sys

            sys.exit()

        # 2. Set the Screen Mode using the raw image data
        self.width = raw_bg.get_width()
        self.height = raw_bg.get_height()

        if surface:
            self.screen = surface
            self.external_render = True
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.external_render = False

        # 3. NOW convert the images (The screen exists!)
        self.bg_im = raw_bg.convert()  # Convert the one we already loaded

        # Load and convert the rest
        try:
            self.fb_im = pygame.image.load(
                "assets/yellowbird-midflap.png"
            ).convert_alpha()
            self.pipe_im_bottom = pygame.image.load(
                "assets/pipe-green.png").convert()
            self.pipe_im_top = pygame.transform.rotate(
                pygame.image.load("assets/pipe-green.png").convert(), 180
            )
        except pygame.error as e:
            print(f"Error loading other assets: {e}")
            sys.exit()

        # Put a mask on top of the player
        self.fb_mask = pygame.mask.from_surface(self.fb_im)
        self.pipe_mask_top = pygame.mask.from_surface(self.pipe_im_top)
        self.pipe_mask_bottom = pygame.mask.from_surface(self.pipe_im_bottom)

        self.fb = self.fb_im.get_rect(center=(50, 256))

        # FB physics
        self.gravity = 0.25
        self.flap_strength = -5
        self.bird_velocity = 0

        # Pipe physics
        self.pipe_vx = 4
        self.dist_between_pairs = 200
        self.inter_dist = 100
        self.gap_center_range = 206
        self.last_gap_y = None  # Track the last pipe's gap position

        self.pipes: list = []

        self.clock = pygame.time.Clock()

        self.running = True
        self.game_active = True

        self.generate_first_10_pipes()

    def check_collision(self):
        # 1. Check Ground Collision (Ground image is 112px tall)
        ground_top = self.height - 112
        if self.fb.bottom >= ground_top:
            return True

        # 2. Check Ceiling Collision
        if self.fb.top <= 0:
            return True

        # 3. Check Pipe Collision
        # Remember: pipe_data is [top_rect, bot_rect, scored]
        for pipe_data in self.pipes:
            top_rect = pipe_data[0]
            bottom_rect = pipe_data[1]

            if self.fb.colliderect(top_rect):
                offset = (top_rect.x - self.fb.x, top_rect.y - self.fb.y)
                if self.fb_mask.overlap(self.pipe_mask_top, offset):
                    return True

            if self.fb.colliderect(bottom_rect):
                offset = (bottom_rect.x - self.fb.x, bottom_rect.y - self.fb.y)
                if self.fb_mask.overlap(self.pipe_mask_bottom, offset):
                    return True

        return False

    def generate_first_10_pipes(self):
        for _ in range(10):
            self.generate_pipe_pair()

    def get_screen_pixels(self):
        return pygame.surfarray.array3d(self.screen)

    def generate_pipe_pair(self):
        # 1. Calculate the Gap Y Position
        half_gap = self.inter_dist / 2
        # Limit gap bounds so we don't expose the top or bottom of the 320px pipe image
        pipe_height = self.pipe_im_top.get_height()
        ground_top = self.height - 112
        min_y = max(100, int(half_gap + ground_top - pipe_height))
        max_y = min(int(self.height - 100), int(pipe_height - half_gap))

        if self.last_gap_y is None:
            # First pipe: completely random within bounds
            gap_center_y = self.rng.randint(min_y, max_y)
        else:
            # Subsequent pipes: constrained by physics (max ascent/descent)
            # A 150px delta is a reasonable approximation for reachability over 200px distance
            delta = 150
            low_bound = max(min_y, self.last_gap_y - delta)
            high_bound = min(max_y, self.last_gap_y + delta)
            gap_center_y = self.rng.randint(int(low_bound), int(high_bound))

        self.last_gap_y = gap_center_y

        # 2. Calculate the X Position
        # If pipes exist, place the new one 'dist_between_pairs' after the last one.
        # If no pipes exist, start off-screen to the right.
        if len(self.pipes) > 0:
            last_pipe_pair = self.pipes[-1]
            # Use .centerx because we are positioning by the center of the pipe
            last_x = last_pipe_pair[0].centerx
            current_x = last_x + self.dist_between_pairs
        else:
            current_x = self.width  # Initial buffer to start off-screen

        # 3. Create Rectangles using Anchors (The Physics Fix)
        # TOP PIPE: The BOTTOM of this pipe touches the top of the gap
        top_rect = self.pipe_im_top.get_rect(
            midbottom=(current_x, gap_center_y - half_gap)
        )

        # BOTTOM PIPE: The TOP of this pipe touches the bottom of the gap
        bottom_rect = self.pipe_im_bottom.get_rect(
            midtop=(current_x, gap_center_y + half_gap)
        )

        # 4. Store the Rect objects directly
        self.pipes.append([top_rect, bottom_rect, False]
                          )  # List (mutable), not Tuple

    def step(self, pressed):
        passed_pipe = False  # Default: we haven't passed anything this frame

        # Loop backwards to allow popping safely
        for i in range(len(self.pipes) - 1, -1, -1):
            pipe_data = self.pipes[i]  # Get the list [top, bot, scored]
            top_pipe = pipe_data[0]
            bot_pipe = pipe_data[1]
            scored = pipe_data[2]

            # 1. Move Pipes
            top_pipe.move_ip(-self.pipe_vx, 0)
            bot_pipe.move_ip(-self.pipe_vx, 0)

            # 2. Check if we just passed this pipe
            # Logic: If bird is to the right of the pipe AND we haven't scored it yet
            if self.fb.left > top_pipe.right and not scored:
                passed_pipe = True
                # Mark as scored so we don't count it again
                pipe_data[2] = True

            if passed_pipe:
                self.generate_pipe_pair()

            # 3. Remove if off-screen
            if top_pipe.right < 0:
                self.pipes.pop(i)

        # Physics
        self.bird_velocity += self.gravity
        if pressed:
            self.bird_velocity = self.flap_strength

        self.fb.move_ip(0, self.bird_velocity)

        # Check Collision
        if self.check_collision():
            return (
                True,
                passed_pipe,
            )  # Game Over, did we pass a pipe right before dying?

        # Render
        # pygame.display.flip() # <-- Moving this to a separate render() function is better practice

        return False, passed_pipe  # Not Done, Passed Status

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        # Reset Bird Physics
        self.fb = self.fb_im.get_rect(center=(50, 256))
        self.bird_velocity = 0

        # Reset Pipes
        self.last_gap_y = None
        self.pipes.clear()
        self.generate_first_10_pipes()  # Or just generate one pair to start

        return False  # Return 'done = False'

    def render(self):
        # 1. Draw Background
        self.screen.blit(self.bg_im, (0, 0))

        # 2. Draw Pipes
        for pipe_pair in self.pipes:
            top_rect = pipe_pair[0]
            bot_rect = pipe_pair[1]
            # Draw the images at the rect coordinates
            self.screen.blit(self.pipe_im_top, top_rect)
            self.screen.blit(self.pipe_im_bottom, bot_rect)

        # 3. Draw Bird
        self.screen.blit(self.fb_im, self.fb)

        # 4. Update Display
        if not self.external_render:
            pygame.display.flip()


def main():
    game = FlappyBird()

    while True:
        # 1. Event Handling (The "Inputs")
        flap_action = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys

                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    flap_action = True

        # 2. Game Logic (The "Step")
        # We pass the boolean action (True if spacebar pressed)
        done, passed_pipe = game.step(flap_action)

        # 3. Render (The "Visuals")
        game.render()

        # 4. Check Game Over
        if done:
            print("Game Over! Restarting...")
            game.reset()

        # 5. Framerate Control
        # This is CRITICAL. Without this, the loop runs at 2000 FPS and gravity kills you instantly.
        game.clock.tick(60)


if __name__ == "__main__":
    main()
