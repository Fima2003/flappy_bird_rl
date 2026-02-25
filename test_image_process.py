import cv2
import numpy as np

def test_image():
    # create a dummy image w=288, h=512, c=3
    img = np.zeros((288, 512, 3), dtype=np.uint8)
    
    # Python Environment Resize Logic
    # 1. flappy_bird_env.get_state():
    # pixels = self.game.get_screen_pixels() # returns (w, h, c)
    # state = np.transpose(pixels, axes=(1, 0, 2)).astype(np.uint8) # returns (h, w, c)
    state = np.transpose(img, axes=(1, 0, 2)).astype(np.uint8)
    
    # 2. ResizeAndGrayscaleWrapper:
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    print("Python Output Shape:", resized.shape)

if __name__ == "__main__":
    test_image()
