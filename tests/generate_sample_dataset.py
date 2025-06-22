import os
import numpy as np
from PIL import Image

def create_binary_digit_image(digit, image_size=(28, 28)):
    """
    Create a simple synthetic binary image for a digit (0-9).
    For simplicity, draws the digit as a white square with digit number pixels.
    """
    img = np.zeros(image_size, dtype=np.uint8)
    size = max(1, digit + 1)  # size varies 1-10 pixels
    img[5:5+size, 5:5+size] = 255
    return img

def generate_dataset(base_path="./tests/sample_data", digits=range(10), samples_per_class=5):
    """
    Generate folders with simple binary images per digit class.
    """
    for split in ['training', 'testing']:
        for digit in digits:
            folder = os.path.join(base_path, split, str(digit))
            os.makedirs(folder, exist_ok=True)
            for i in range(samples_per_class):
                img = create_binary_digit_image(digit)
                img_pil = Image.fromarray(img)
                filename = os.path.join(folder, f"{digit}_{i}.png")
                img_pil.save(filename)
    print(f"Sample dataset created at {base_path}")

if __name__ == "__main__":
    generate_dataset()
