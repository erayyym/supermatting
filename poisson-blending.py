import cv2
import numpy as np
from scipy.ndimage import laplace

def poisson_blend(src, mask, dest):
    for _ in range(20000):
        dest = dest + 0.25 * mask * laplace(dest - src)
    return dest.clip(0, 1)

dest = cv2.imread("dest.jpg").astype(np.float64) / 255.0
source = cv2.imread("src.png").astype(np.float64) / 255.0
mask = cv2.imread("mask.jpg", 0) == 255

h, w, three = dest.shape
hei = int(h/4)
wid = int(w/4)
dest_m = cv2.resize(dest, (256, 256))
src_m = cv2.resize(source, (256, 256))
# mask_m = cv2.resize(mask, (wid, hei), interpolation=cv2.INTER_NEAREST) > .5

r = poisson_blend(src_m[:,:,0], mask, dest_m[:,:,0])
g = poisson_blend(src_m[:,:,1], mask, dest_m[:,:,1])
b = poisson_blend(src_m[:,:,2], mask, dest_m[:,:,2])
blended = np.stack([r, g, b], axis=2)

cv2.imwrite("out.png", 255*blended)
