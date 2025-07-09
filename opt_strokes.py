# ───────────────────────────────────────────────── demo_prog_init.py ──────
"""
Progressive neural painting – but optionally initialise every patch with a
pre‑existing stroke file (.npz).  If --init_strokes is omitted the behaviour
is identical to the original demo_prog.py (random initialisation).
"""
import argparse, os, torch, torch.optim as optim, numpy as np
from painter import ProgressivePainter
import utils                                      # for img2patches & helpers


# ─────────────── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING w/ init')
parser.add_argument('--img_path',  default='./test_images/apple.jpg')
parser.add_argument('--renderer',  default='oilpaintbrush')
parser.add_argument('--canvas_color', default='black')
parser.add_argument('--canvas_size', type=int, default=512)
parser.add_argument('--max_m_strokes', type=int, default=500)
parser.add_argument('--max_divide',    type=int, default=5)
parser.add_argument('--net_G',         default='zou-fusion-net-light')
parser.add_argument('--renderer_checkpoint_dir',
                    default='./checkpoints_G_oilpaintbrush_light')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--output_dir', default='./output')
parser.add_argument('--disable_preview', action='store_true')
# NEW:
parser.add_argument('--init_strokes', help='optional .npz with x_ctt/x_color/x_alpha')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ─────────────── helper: load & reshape init file ────────────────────────
def load_init_npz(path, pt):
    """
    Returns PARAMS of shape (1, N, 12) ready for pt._render,
    and also the per‑level list used to seed each patch.
    """
    d = np.load(path)
    v = np.concatenate([d['x_ctt'], d['x_color'], d['x_alpha']], axis=-1)

    if v.ndim == 2:
        v = v[np.newaxis, ...]   # adds batch dimension
    assert v.ndim == 3 and v.shape[0] == 1 and v.shape[2] == pt.rderr.d
    PARAMS = v.copy()                       # keep master copy
    # pre‑compute a list of split views: one entry per m_grid level
    splits = []
    start = 0
    for m in range(1, pt.max_divide + 1):
        per_patch   = pt.max_m_strokes // (m * m)
        block_total = m * m * per_patch
        splits.append(PARAMS[:, start:start+block_total, :])
        start += block_total
    return PARAMS, splits


# ─────────────── optimise function (mod of original) ─────────────────────
def optimize_x(pt, init_npz=None):
    pt._load_checkpoint();  pt.net_G.eval()
    print('begin drawing...')

    if init_npz:
        PARAMS, split_views = load_init_npz(init_npz, pt)
    else:
        PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)
        split_views = None                        # not used

    CANVAS_tmp = torch.ones([1,3,pt.net_G.out_size,pt.net_G.out_size],device=device) \
                 if pt.rderr.canvas_color=='white' else \
                 torch.zeros([1,3,pt.net_G.out_size,pt.net_G.out_size],device=device)

    # progressive loops ----------------------------------------------------
    for pt.m_grid in range(1, pt.max_divide+1):
        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
        pt.G_final_pred_canvas = CANVAS_tmp

        if init_npz:
          
          v = split_views[pt.m_grid-1]            # slice for this level
          v = v[0]                                # drop batch dim  → (N,12)

          # reshape to [patch, strokes_per_patch, 12]
          strokes_per_patch = v.shape[0] // (pt.m_grid * pt.m_grid)
          v = v.reshape(pt.m_grid * pt.m_grid, strokes_per_patch, pt.rderr.d)

          # assign to painter
          pt.x = v
          pt.x_ctt   = torch.tensor(v[..., :5],   device=device, requires_grad=True)
          pt.x_color = torch.tensor(v[..., 5:11], device=device, requires_grad=True)
          pt.x_alpha = torch.tensor(v[..., 11:],  device=device, requires_grad=True)
        else:
          
            # original random initialisation
            pt.initialize_params()
            pt.x_ctt.requires_grad = True
            pt.x_color.requires_grad = True
            pt.x_alpha.requires_grad = True

        optim_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha],
                                lr=pt.lr, centered=True)

        pt.step_id = 0
        for pt.anchor_id in range(pt.m_strokes_per_block):
            pt.stroke_sampler(pt.anchor_id)
            for _ in range(int(500/pt.m_strokes_per_block)):
                pt.G_pred_canvas = CANVAS_tmp
                optim_x.zero_grad()
                # clamp & forward
                pt.x_ctt.data.clamp_(0.1, 0.9)
                pt.x_color.data.clamp_(0.0, 1.0)
                pt.x_alpha.data.clamp_(0.0, 1.0)
                pt._forward_pass();  pt._drawing_step_states();  pt._backward_x()
                optim_x.step();      pt.step_id += 1

        v = pt._normalize_strokes(pt.x)
        v = pt._shuffle_strokes_and_reshape(v)
        PARAMS = np.concatenate([PARAMS, v], 1)      # accumulate
        CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid+1, pt.net_G.out_size).to(device)

    pt._save_stroke_params(PARAMS)
    pt._render(PARAMS, save_jpgs=False, save_video=True)
    print('✓ finished; results in', args.output_dir)


# ─────────────── main ────────────────────────────────────────────────────
if __name__ == '__main__':
    pt = ProgressivePainter(args=args)
    optimize_x(pt, init_npz=args.init_strokes)
