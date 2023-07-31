import math

num_center = 169343
num_edge = 2332486
block_size = 32
feat_len = 256
deg = num_edge / num_center
num_rel = 1
num_head = 4
deg2 = deg / num_rel
# print(deg2)

# spmm

# vertex-centrc
mem = 2 + deg / block_size + deg * feat_len / block_size + feat_len / block_size
op = deg * feat_len * 2 / block_size
print("vertex-centrc", op / mem)

# edge-centric
mem = 2 + feat_len / block_size * 2
op = feat_len * 2 / block_size
print("edge-centrc", op / mem)

# optimal
mem = deg * feat_len / block_size + feat_len / block_size
op = deg * feat_len * 2 / block_size
print("optimal", op / mem)

# rgcn

# vertex-centrc
mem = 2 + deg / block_size + deg * feat_len / block_size + feat_len / block_size + deg * feat_len * feat_len / block_size
op = (deg * feat_len * feat_len + deg * feat_len) / block_size
print("vertex-centrc", op / mem)

# edge-centric
mem = 2 + feat_len / block_size * 2 + feat_len * feat_len / block_size
op = feat_len * feat_len / block_size + feat_len / block_size
print("edge-centrc", op / mem)

# optimal
mem = deg2 * feat_len / block_size + feat_len / block_size + feat_len * feat_len / block_size
op = deg2 * feat_len * feat_len / block_size + deg2 * feat_len / block_size
print("optimal", op / mem)

# sddmm

# vertex-centrc
mem = 2 + feat_len / block_size + deg / block_size + deg * feat_len * num_head / block_size + deg * num_head
op = deg * feat_len * num_head * 2 / block_size
print("vertex-centrc", op / mem)

# edge-centric
mem = 2 + feat_len * (num_head + 1) / block_size
op = feat_len * num_head * 2 / block_size
print("edge-centrc", op / mem)

# optimal
# mem = deg * feat_len * num_head / block_size + feat_len * num_head / block_size
mem = num_head * feat_len + feat_len * deg + num_head * deg
op = num_head * feat_len * deg * 2
print("optimal", op / mem)

# lstm
