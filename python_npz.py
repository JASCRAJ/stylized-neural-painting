# render_npz.py
import argparse, os, cv2, torch, numpy as np
from types import SimpleNamespace
from painter import Painter

# ------------------------------------------------------------------ #
# 1. CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--npz', required=True, help='stroke‑param file produced by demo.py')
parser.add_argument('--ckpt', default='./checkpoints_G_oilpaintbrush',
                    help='dir with Gs.pth / Gr.pth')
args_cli = parser.parse_args()

# ------------------------------------------------------------------ #
# 2. Build fake Painter args
# ------------------------------------------------------------------ #
args = SimpleNamespace(
    img_path='blank.jpg',                 # dummy photo so Painter doesn’t crash
    renderer='oilpaintbrush',
    renderer_checkpoint_dir='checkpoints_G_oilpaintbrush',
    net_G='zou-fusion-net',
    canvas_color='white',
    canvas_size=512,
    m_grid=5,
    max_m_strokes=1500,
    keep_aspect_ratio=False,
    beta_L1=1.0,
    with_ot_loss=False,
    beta_ot=0.1,
    lr=0.002,
    output_dir='./output',
    disable_preview=True,
)

# ------------------------------------------------------------------ #
# 3. Create a blank image for Painter to load
# ------------------------------------------------------------------ #
if not os.path.exists(args.img_path):
    blank = (np.ones((args.canvas_size, args.canvas_size, 3), np.uint8) * 255)
    cv2.imwrite(args.img_path, blank)        # BGR order for OpenCV

# ------------------------------------------------------------------ #
# 4. Initialise Painter & neural renderer
# ------------------------------------------------------------------ #
pt = Painter(args=args)          # will happily load the dummy photo
pt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pt._load_checkpoint()            # loads Gs.pth / Gr.pth
pt.net_G.eval()

# ------------------------------------------------------------------ #
# 5. Load stroke parameters
# ------------------------------------------------------------------ #
data = np.load(args_cli.npz)
x_np = np.concatenate([data['x_ctt'],      # (25,60,5)
                       data['x_color'],    # (25,60,6)
                       data['x_alpha']],   # (25,60,1)
                      axis=-1)             # → (25,60,12)
x = torch.tensor(x_np, dtype=torch.float32, device=pt.device)\
        .unsqueeze(-1).unsqueeze(-1)       # (25,60,12,1,1)
pt.x = x

# ------------------------------------------------------------------ #
# 6. Normalise & render  (skip the problematic shuffle)
# ------------------------------------------------------------------ #
v_n = pt._normalize_strokes(pt.x)          # already patch‑wise
final_img = pt._render(v_n,
                       save_jpgs=True,
                       save_video=False)   # returns PIL.Image

# ------------------------------------------------------------------ #
# 7. Show & save
# ------------------------------------------------------------------ #
final_img.show()                           # visual check
final_img.save('reconstructed_from_npz.png')
print('✓  Rendering finished → reconstructed_from_npz.png')
