#!/usr/bin/env python3
"""Generate synthetic shelf test images for the demo."""
import cv2
import numpy as np
import os

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "web", "public", "samples")
os.makedirs(out_dir, exist_ok=True)


def generate_shelf(seed, img_size, num_shelves, cols, name):
    rng = np.random.RandomState(seed)
    img = np.full((img_size, img_size, 3), 210, dtype=np.uint8)

    shelf_h = img_size // num_shelves
    for s in range(num_shelves):
        y0 = s * shelf_h
        y1 = min((s + 1) * shelf_h, img_size)
        shade = rng.randint(185, 225)
        img[y0:y1, :] = shade
        cv2.line(img, (0, y1 - 1), (img_size, y1 - 1), (90, 80, 70), 2)

    cell_w = img_size // cols
    for r in range(num_shelves):
        for c in range(cols):
            pw = rng.randint(int(cell_w * 0.45), int(cell_w * 0.85))
            ph = rng.randint(int(shelf_h * 0.45), int(shelf_h * 0.85))
            cx = c * cell_w + cell_w // 2 + rng.randint(-8, 9)
            cy = r * shelf_h + shelf_h // 2 + rng.randint(-5, 6)
            x1 = max(0, cx - pw // 2)
            y1_ = max(0, cy - ph // 2)
            x2 = min(img_size, x1 + pw)
            y2 = min(img_size, y1_ + ph)

            color = tuple(int(v) for v in rng.randint(50, 230, 3))
            cv2.rectangle(img, (x1, y1_), (x2, y2), color, -1)
            border = tuple(int(v) for v in (np.array(color) * 0.5).astype(int))
            cv2.rectangle(img, (x1, y1_), (x2, y2), border, 1)

            if rng.rand() > 0.3:
                lw = max(6, (x2 - x1) // 2)
                lh = max(4, (y2 - y1_) // 4)
                lx = x1 + (x2 - x1 - lw) // 2
                ly = y1_ + (y2 - y1_) // 2 - lh // 2
                cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (240, 240, 240), -1)

    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img)
    print(f"Saved {path}")


# Sparse shelf (few large products)
generate_shelf(seed=99, img_size=500, num_shelves=3, cols=4, name="shelf_sparse.png")

# Dense shelf (many small products)
generate_shelf(seed=42, img_size=500, num_shelves=5, cols=7, name="shelf_dense.png")

# Medium shelf
generate_shelf(seed=77, img_size=500, num_shelves=4, cols=6, name="shelf_medium.png")

print("Done!")
