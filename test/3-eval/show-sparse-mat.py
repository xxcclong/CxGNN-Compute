import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import cxgnncomp as cxgc
import torch
import cv2




dset = "arxiv"

feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
    dset, feat_len=1, num_head=1, need_edge_index=True, need_feat=False)

# reorder = torch.from_numpy(
#     np.fromfile(f"/home/huangkz/repos/rabbit_order/demo/reorder_{dset}.dat",
#                 dtype=np.int64)).cuda()
# edge_index = reorder[edge_index]

num_v = ptr.shape[0] - 1

size = 10000

shape = (size, size)
edge_index = edge_index / num_v * size
edge_index = edge_index.to(torch.int32)
print(torch.max(edge_index))
edge_index = edge_index.cpu().numpy()
rows = edge_index[0]
cols = edge_index[1]
vals = np.ones_like(rows)

img = np.zeros(shape)
img[rows, cols] = vals
# img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)
img = img * 255
img = img.astype(np.uint8)
cv2.imwrite("arxiv-reorder.png", img)