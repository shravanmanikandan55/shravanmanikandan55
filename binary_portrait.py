import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = "myphoto.jpg"
OUTPUT_IMAGE = "github_binary_avatar.png"

WIDTH = 900
HEIGHT = 900

# Characters used to build the face
CHARS = ["0", "1"]

# Character size
FONT_SIZE = 9

# Glitch amount
GLITCH_LINES = 35

# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise FileNotFoundError(f"Could not find {INPUT_IMAGE}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crop image to square
h, w = gray.shape
size = min(h, w)

x = (w - size) // 2
y = (h - size) // 2

gray = gray[y:y + size, x:x + size]

# Resize
gray = cv2.resize(gray, (WIDTH, HEIGHT))

# Slight contrast enhancement
gray = cv2.equalizeHist(gray)

# ============================================================
# CREATE DARK BACKGROUND
# ============================================================

canvas = Image.new(
    "RGB",
    (WIDTH, HEIGHT),
    (2, 6, 12)
)

draw = ImageDraw.Draw(canvas)

# Try monospace font
try:
    font = ImageFont.truetype(
        "C:/Windows/Fonts/consola.ttf",
        FONT_SIZE
    )
except:
    font = ImageFont.load_default()

# ============================================================
# CREATE BINARY FACE
# ============================================================

# Characters are drawn every FONT_SIZE pixels
for y in range(0, HEIGHT, FONT_SIZE):
    for x in range(0, WIDTH, FONT_SIZE):

        brightness = gray[y, x]

        # Dark areas = fewer/brighter characters
        # Bright areas = stronger characters

        if brightness < 35:
            continue

        # Convert brightness into density
        probability = brightness / 255

        if random.random() > probability:
            continue

        char = random.choice(CHARS)

        # Brightness controls character intensity
        intensity = int(40 + brightness * 0.85)

        intensity = min(255, intensity)

        # Slight blue/cyan tint
        color = (
            30,
            intensity,
            255
        )

        draw.text(
            (x, y),
            char,
            font=font,
            fill=color
        )

# ============================================================
# ADD STRONG WHITE BINARY AREAS
# ============================================================

# Threshold highlights
_, highlight = cv2.threshold(
    gray,
    175,
    255,
    cv2.THRESH_BINARY
)

for y in range(0, HEIGHT, FONT_SIZE):
    for x in range(0, WIDTH, FONT_SIZE):

        if highlight[y, x] > 0:

            # Don't draw every character
            if random.random() < 0.55:

                char = random.choice(CHARS)

                draw.text(
                    (x, y),
                    char,
                    font=font,
                    fill=(230, 250, 255)
                )

# ============================================================
# GLITCH EFFECT
# ============================================================

pixels = np.array(canvas)

for _ in range(GLITCH_LINES):

    y = random.randint(0, HEIGHT - 5)

    height = random.randint(1, 5)

    shift = random.randint(-80, 80)

    section = pixels[y:y + height].copy()

    if shift > 0:

        pixels[
            y:y + height,
            shift:
        ] = section[:, :-shift]

    elif shift < 0:

        shift = abs(shift)

        pixels[
            y:y + height,
            :-shift
        ] = section[:, shift:]

# ============================================================
# BINARY GLITCH PARTICLES
# ============================================================

glitch_img = Image.fromarray(pixels)
glitch_draw = ImageDraw.Draw(glitch_img)

for _ in range(500):

    x = random.randint(0, WIDTH - 20)
    y = random.randint(0, HEIGHT - 10)

    char = random.choice(["0", "1"])

    # Mostly dark blue/cyan
    color = random.choice([
        (0, 70, 140),
        (0, 120, 220),
        (20, 180, 255),
        (80, 220, 255)
    ])

    glitch_draw.text(
        (x, y),
        char,
        font=font,
        fill=color
    )

# ============================================================
# HORIZONTAL SCANLINES
# ============================================================

final = np.array(glitch_img)

for y in range(0, HEIGHT, 6):

    final[y:y + 1] = (
        final[y:y + 1] * 0.45
    ).astype(np.uint8)

# ============================================================
# SAVE
# ============================================================

output = Image.fromarray(final)

output.save(
    OUTPUT_IMAGE,
    quality=95
)

print("================================")
print("Binary portrait created!")
print(f"Saved as: {OUTPUT_IMAGE}")
print("================================")
