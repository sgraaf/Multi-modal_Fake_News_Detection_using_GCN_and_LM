from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageChops, ImageOps

def trim(image):
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def pad(image):
    return ImageOps.expand(image, border=5, fill='white')

plt.style.use('seaborn-white')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'output' / 'results'
STL_FILE = RESULTS_DIR / 'GraphConvolutionalNetwork_results.csv'

df = pd.read_csv(STL_FILE, sep=';', encoding='utf-8')
print(df.keys())

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df['epoch'], df['train_acc'], color='tab:red', marker='.', label='Train')
ax.plot(df['epoch'], df['val_acc'], color='tab:orange', marker='.', label='Validation')
ax.plot(df['epoch'], df['test_acc'], color='tab:purple', label='Test')
ax.axhline(y=0.5, color='tab:gray', linestyle='--', label='Random baseline')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.tick_params('y')
ax.set_ylim([0.4, 1])
ax.legend()
plt.tight_layout()
plt_file = RESULTS_DIR / (STL_FILE.stem + '_accuracy_curves.pdf')
plt.savefig(plt_file)
# I = Image.open(plt_file)
# I = trim(I)
# I = pad(I)
# I.save(plt_file.parent / (plt_file.stem + '_trimmed' + plt_file.suffix))
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df['epoch'], df['val_loss'], color='tab:orange', marker='.', label='Val. loss')
ax.plot(df['epoch'], df['train_loss'], color='tab:red', marker='.', label='Train loss')
ax.plot(df['epoch'], df['test_loss'], color='tab:purple', label='Test loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.tick_params('y')
# ax.set_ylim([0.4, 1])
ax.legend()
plt.tight_layout()
plt_file = RESULTS_DIR / (STL_FILE.stem + '_loss_curves.pdf')
plt.savefig(plt_file)
# I = Image.open(plt_file)
# I = trim(I)
# I = pad(I)
# I.save(plt_file.parent / (plt_file.stem + '_trimmed' + plt_file.suffix))
plt.show()
