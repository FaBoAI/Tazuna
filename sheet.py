"""Make a contact sheet from view images: python sheet.py out.jpg img1 img2 ..."""
import sys
import cv2
import numpy as np
out, paths = sys.argv[1], sys.argv[2:]
ims = [cv2.imread(p) for p in paths]
h = max(im.shape[0] for im in ims)
w = max(im.shape[1] for im in ims)
cols = min(4, len(ims))
rows = (len(ims) + cols - 1) // cols
sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
for i, (im, p) in enumerate(zip(ims, paths)):
    r, c = divmod(i, cols)
    sheet[r * h:r * h + im.shape[0], c * w:c * w + im.shape[1]] = im
    cv2.putText(sheet, p.split("/")[-1][:28], (c * w + 5, r * h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(sheet.shape)
