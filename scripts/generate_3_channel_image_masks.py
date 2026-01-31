import json
import math
from pathlib import Path
import random
from typing import Union

import click
from PIL import Image, ImageDraw

from stardist.src.utils.config import read_json_config

def make_three_object_composites(
    width,
    height,
    n_groups_min=4,
    n_groups_max=8,                   # how many composite triplets to draw
    size_range=(0.05, 0.09),        # each triplet is 5–9% of min(width,height)
    seed=None,
    out_prefix="three_composites_rgb",
    target_type="tif",
    exclusive=True,
    rgb_dir=""
):
    """
    Draw N tiny, non-overlapping composite 3-object instances at random locations on a black RGB canvas.
    Each instance has the same internal layout (rounded-rect background, overlapping
    circle + star) as the reference. The result is saved as an RGB PNG.

      - RGB composite PNG to rgb_dir
      - circle instance mask (8-bit PNG) to circle_dir
      - star   instance mask (8-bit PNG) to star_dir
      - rect   instance mask (8-bit PNG) to rect_dir

    Parameters
    ----------
    width, height : int
        Canvas size in pixels (black RGB background).
    n_groups : int
        Multiplicity (how many composite triplets to place).
    scale_range : (float, float)
        Min/max fraction of min(width, height) that sets each group's outer-box size.
    seed : int | None
        Random seed for reproducibility.
    outfile : str
        Output PNG path.

    Returns
    -------
    PIL.Image.Image  (RGB)
    """
    if seed is not None:
        random.seed(seed)

    # Final RGB canvas
    base_rgb = Image.new("RGB", (width, height), "black")

    # per-class instance masks (L-mode, 8-bit integer; background=0)
    mask_circle = Image.new("L", (width, height), 0)
    mask_star   = Image.new("L", (width, height), 0)
    mask_rect   = Image.new("L", (width, height), 0)
    draw_mc = ImageDraw.Draw(mask_circle)
    draw_ms = ImageDraw.Draw(mask_star)
    draw_mr = ImageDraw.Draw(mask_rect)

    # Colors (RGB with alpha used internally for overlap shading)
    COL_CAT2 = (186, 221, 186, 255)  # rounded rect
    COL_CAT1 = ( 82, 122, 192, 220)  # circle   (slight alpha for overlap)
    COL_CAT3 = (198, 150, 130, 220)  # star     (slight alpha for overlap)

    def star_points(cx, cy, r_outer, r_inner, n=5):
        pts = []
        for i in range(2 * n):
            ang = -math.pi/2 + i * math.pi / n
            r = r_outer if i % 2 == 0 else r_inner
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        return pts

    def make_triplet_layer_and_masks(x0, y0, size, circ_id, star_id, rect_id, exclusive=exclusive):
        """
        Create an RGBA layer with one colored triplet, and draw the three
        instance masks into the provided mask canvases. Return the RGBA layer.
        """
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        if exclusive:
            # Outer rounded-rect (category 2) inside this box, with fixed relative margins/radius
            m = int(0.08 * size)        # inner margin
            rr = int(0.22 * size)        # corner radius
            rect_box = [x0 + m, y0 + m, x0 + size - m, y0 + size - m]
            d.rounded_rectangle(rect_box, radius=int(rr), fill=COL_CAT2, outline=COL_CAT2, width=2)

            # Circle (category 1) — left fixed relative position inside the outer box
            cx1, cy1 = int(x0 + 0.33 * size), int(y0 + 0.53 * size)
            r1 = int(0.12 * size)
            circ_box = [cx1 - r1, cy1 - r1, cx1 + r1, cy1 + r1]
            d.ellipse(circ_box, fill=COL_CAT1, outline=COL_CAT1, width=2)

            # Star (category 3) — right fixed relative position inside the outer box
            cx3, cy3 = int(x0 + 0.62 * size), int(y0 + 0.50 * size)
            r_out, r_in = int(0.12 * size), int(0.06 * size)
            star_poly = star_points(cx3, cy3, r_out, r_in)
            d.polygon(star_poly, fill=COL_CAT3, outline=COL_CAT3)

        else:
            # Outer rounded-rect (category 2) inside this box, with fixed relative margins/radius
            m = int(0.08 * size)        # inner margin
            rr = int(0.22 * size)        # corner radius
            rect_box = [x0 + m, y0 + m, x0 + size - m, y0 + size - m]
            d.rounded_rectangle(rect_box, radius=int(rr), fill=COL_CAT2, outline=COL_CAT2, width=3)

            # Circle (category 1) — fixed relative position inside the outer box
            cx1, cy1 = int(x0 + 0.46 * size), int(y0 + 0.54 * size)
            r1 = int(0.13 * size)
            circ_box = [cx1 - r1, cy1 - r1, cx1 + r1, cy1 + r1]
            d.ellipse(circ_box, fill=COL_CAT1, outline=COL_CAT1, width=3)

            # Star (category 3) — overlaps circle
            cx3, cy3 = int(x0 + 0.56 * size), int(y0 + 0.53 * size)
            r_out, r_in = int(0.12 * size), int(0.06 * size)
            star_poly = star_points(cx3, cy3, r_out, r_in)
            d.polygon(star_poly, fill=COL_CAT3, outline=COL_CAT3)

        # draw instance masks (solid integer labels)
        draw_mr.rounded_rectangle(rect_box, radius=int(rr), fill=int(rect_id))
        draw_mc.ellipse(circ_box, fill=int(circ_id))
        draw_ms.polygon(star_poly, fill=int(star_id))

        return layer

    # Place N groups at random positions; keep each fully inside the canvas
    side_min = min(width, height)
    pad = int(0.1 * side_min)
    n_groups = random.randint(n_groups_min, n_groups_max)
    placed_boxes = []  # list of (x0, y0, x1, y1) for each triplet

    def overlaps(new_box):
        x0, y0, x1, y1 = new_box
        # expanded boxes for padding
        for (a0, b0, a1, b1) in placed_boxes:
            if not (x1 + pad <= a0 - pad or
                    x0 - pad >= a1 + pad or
                    y1 + pad <= b0 - pad or
                    y0 - pad >= b1 + pad):
                return True
        return False

    attempts_limit = 50
    attempts, placed = 0, 0
    circ_id = star_id = rect_id = 0

    while placed < n_groups and attempts < attempts_limit:
        attempts += 1
        size = random.uniform(*size_range) * side_min
        border = int(0.01 * side_min)  # keep outlines away from edges
        max_x0 = max(border, int(width  - size - border))
        max_y0 = max(border, int(height - size - border))
        if max_x0 <= border or max_y0 <= border:
            print ("canvas too small for requested size")
            break  # canvas too small for requested size

        x0 = random.randint(border, max_x0)
        y0 = random.randint(border, max_y0)
        candidate = (x0, y0, x0 + size, y0 + size)

        if overlaps(candidate):
            continue

        # assign new instance IDs (background stays 0)
        circ_id += 1; star_id += 1; rect_id += 1

        # build the RGBA layer + write the masks
        layer = make_triplet_layer_and_masks(x0, y0, size, circ_id, star_id, rect_id)
        
        base_rgb = Image.alpha_composite(base_rgb.convert("RGBA"), layer).convert("RGB")

        placed_boxes.append(candidate)
        placed += 1

    # --- save outputs ---
    
    rgb_path  = Path(rgb_dir) / ("composite_masks") / (out_prefix + "." + target_type)
    circ_path = Path(rgb_dir) / ("circle_masks") / (out_prefix + "." + target_type)
    star_path = Path(rgb_dir) / ("star_masks") / (out_prefix + "." + target_type)
    rect_path = Path(rgb_dir) / ("rounded_rect_masks") / (out_prefix + "." + target_type)

    base_rgb.save(rgb_path, target_type)
    mask_circle.save(circ_path, target_type)  # 8-bit integer tif
    mask_star.save(star_path, target_type)
    mask_rect.save(rect_path, target_type)

    return {
        "rgb": rgb_path,
        "mask_circle": circ_path,
        "mask_star": star_path,
        "mask_rect": rect_path,
        "placed": placed
    }




@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)

    current_dir	= config['project_result_dir']
    if current_dir:
        current_dir = Path(current_dir)
    else:
        current_dir = Path(__file__).parent

    print(current_dir)
    parameters = config['parameters']


    # Example:

    _ = [make_three_object_composites(width=parameters['img_w'], height=parameters['img_h'], 
                                    n_groups_min=parameters['n_obj_inst_min'],
                                    n_groups_max = parameters['n_obj_inst_max'],
                                    seed=random.seed(),
                                    out_prefix=str(i), 
                                    target_type=parameters['target_type'], 
                                    exclusive=parameters['inner_object_exclusive'],
                                    rgb_dir=current_dir) 
                                    for i in range(parameters['n_images'])]
    

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter