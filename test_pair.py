import os
import time

os.chdir("..")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figures, make_matching_figure
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

# You can choose model type in ['full', 'opt']
model_type = 'full'  # 'full' for best quality, 'opt' for best efficiency

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
precision = 'fp32'  # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can also change the default values like thr. and npe (based on input image size)

if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)

if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True

print(_default_cfg)
matcher = LoFTR(config=_default_cfg)

matcher.load_state_dict(torch.load("/home/garik/PycharmProjects/Eloftr/EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
matcher = reparameter(matcher)  # no reparameterization will lead to low performance

if precision == 'fp16':
    matcher = matcher.half()

matcher = matcher.eval().cuda()


# Load example images
img0_pth = "/home/garik/PycharmProjects/Eloftr/EfficientLoFTR/Flight_3/image_2.jpg"
img1_pth = "/home/garik/PycharmProjects/Eloftr/EfficientLoFTR/cropped/2_2_40.22447304688932_43.92158615719963_cropped.png"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640,448))  # input size shuold be divisible by 32
img1_raw = cv2.resize(img1_raw, (640,448))

if precision == 'fp16':
    img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
else:
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}
print(img0.shape, img1.shape)
# Inference with EfficientLoFTR and get prediction
with torch.no_grad():
    now = time.time()
    if precision == 'mp':
        with torch.autocast(enabled=True, device_type='cuda'):
            matcher(batch)
    else:
        matcher(batch)
    print("inference time: " , time.time() - now)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()



if model_type == 'opt':
    print(mconf.max())
    mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]

fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)

conf_thresh = 0.7  # or 0.8, depending on how strict you want to be

# Filter matches based on confidence
mask = mconf > conf_thresh
mkpts0_filtered = mkpts0[mask]
mkpts1_filtered = mkpts1[mask]
mconf_filtered = mconf[mask]
color_filtered = cm.jet(mconf_filtered)

# Update text
text = [
    'LoFTR (filtered)',
    f'Matches (>{conf_thresh:.2f}): {len(mkpts0_filtered)}',
]

fig = make_matching_figure(
    img0_raw, img1_raw, mkpts0_filtered, mkpts1_filtered, color_filtered, text=text
)