import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

base_path = os.path.join(os.path.dirname(__file__), 'data')
categories = ['normal', 'blackhole', 'wormhole_candidate']

for cat in categories:
    os.makedirs(os.path.join(base_path, cat), exist_ok=True)

def generate_normal(size=64):
    img = np.random.rand(size, size) * 0.3
    img = gaussian_filter(img, sigma=2)
    return img

def generate_blackhole(size=64):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    z = np.exp(-((x**2 + y**2)*20))
    return z

def generate_wormhole(size=64):
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    z = -1 / (np.sqrt(x**2 + y**2 + 0.1)) + 1 / (np.sqrt((x-1.5)**2 + y**2 + 0.1))
    z = (z - z.min()) / (z.max() - z.min())  # Normalize
    return z

def save_images(generator_fn, category, count=300):
    for i in tqdm(range(count), desc=f"Generating {category}"):
        img = generator_fn()
        path = os.path.join(base_path, category, f"{category}_{i}.png")
        plt.imsave(path, img, cmap='plasma')

# Generate 300 images for each
save_images(generate_normal, 'normal')
save_images(generate_blackhole, 'blackhole')
save_images(generate_wormhole, 'wormhole_candidate')

print("✅ All training images generated.")
