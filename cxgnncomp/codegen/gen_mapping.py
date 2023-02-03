import torch
import cxgnncomp
import cxgnncomp_backend
import time
import random

# num_node = ptr.shape[0] - 1

# output_torch = cxgnncomp.spmm_triton(x, ptr, idx, ptr.shape[0] - 1)

# ptr_for_dgsparse = ptr.to(torch.int32)
# idx_for_dgsparse = idx.to(torch.int32)

# cxgnncomp.prof(f"spmm", "triton",
#                lambda: cxgnncomp.spmm_triton(x, ptr, idx, ptr.shape[0] - 1))
# cxgnncomp.prof(
#     f"spmm", "manual",
#     lambda: cxgnncomp.sage_sum_forward(x, ptr, idx, ptr.shape[0] - 1))
# manual: (307328, 1, 128, 1, 2, 256, 128, 0, 0)
# cxgnncomp.prof(
#     f"spmm", "dgsparse", lambda: cxgnncomp_backend.GSpMM_u(
#         ptr_for_dgsparse, idx_for_dgsparse, x, cxgnncomp_backend.REDUCEOP.SUM))
# dgsparse: (76832, 4, 32, 8, 8, 64, 64, 1, 0)


def validate(cpw, rpb, cpb, block_size, feat_len):
    valid = 1
    if cpb % cpw != 0:
        valid = 0
    if block_size > 1024 or block_size <= 0:
        valid = 0
    if cpb > feat_len:
        valid = 0
    return valid


# params = [(256, 1, num_node, )]
# params = [
#     (76832, 4, 32, 8, 8, 64, 64, 1, 0),
#     (307328, 1, 128, 1, 2, 256, 128, 0, 0),
#     (614656, 2, 32, 1, 1, 128, 128, 1, 0)
# ]
def tune_spmm(num_node,
              num_edge,
              feat_len,
              func,
              run_param,
              performance_param=None,
              patience=50):
    if performance_param is not None:
        res = cxgnncomp.prof("spmm", "config",
                             lambda: func(*run_param, *performance_param))
        return [
            res[1], performance_param,
            func(*run_param, *performance_param)
        ]
    cpws = [32, 64, 128, 256, 512, 1024]
    # rpbs = [2, 4, 8, 16, 32]
    rpbs = [1, 2, 4, 8, 16, 32]
    cpbs = [32, 64, 128, 256, 512, 1024]
    block_xs = [32, 64, 128, 256, 512, 1024]
    grid_ys = [1, 2, 4, 8, 16, 32]
    block_maps = [0, 1]
    grid_maps = [0, 1]

    ceil_feat_len = (feat_len + 31) // 32 * 32
    params = []
    for rpb in rpbs:
        for cpb in cpbs:
            for cpw in cpws:
                block_size = rpb * 32 * (cpb // cpw)
                num_grid = num_node * ceil_feat_len // rpb // cpb
                # print(cpw, rpb, cpb, block_size)
                if not validate(cpw, rpb, cpb, block_size, feat_len):
                    continue
                for block_x in block_xs:
                    if block_x > block_size:
                        continue
                    if block_size % block_x != 0:
                        continue
                    block_y = block_size // block_x
                    for grid_y in grid_ys:
                        grid_x = (num_grid + grid_y - 1) // grid_y
                        grid_x = (grid_x + (feat_len // rpb) -
                                  1) // (feat_len // rpb) * (feat_len // rpb)
                        for block_map in block_maps:
                            for grid_map in grid_maps:
                                if block_x < (cpb // cpw *
                                              32) and block_map == 0:
                                    continue
                                if block_y < (cpb // cpw) and block_map == 1:
                                    continue
                                if grid_y < (ceil_feat_len //
                                             cpb) and grid_map == 1:
                                    continue
                                params.append(
                                    (grid_x, grid_y, block_x, block_y, rpb,
                                     cpb, cpw, grid_map, block_map))
    mmin = 1e9
    best_config = None

    cnt = 0
    random.seed(233)
    random.shuffle(params)

    t0 = time.time()
    no_impv = 0
    for param in params:
        res = cxgnncomp.prof("spmm",
                             "config",
                             lambda: func(*run_param, *param),
                             display=False)

        if res[1] < mmin and param[6] == 128:
            mmin = res[1]
            best_config = param
            no_impv = 0
        else:
            no_impv += 1
        if patience > 0 and no_impv > patience:
            break
        if cnt % 1000 == 0 and cnt != 0:
            print("res", res[1], mmin, param, best_config)
            print(f"progress: {cnt}/{len(params)} {time.time() - t0}")
        cnt += 1
    print("Tuning finish with patience", patience, "tune time",
          time.time() - t0, f"trial-num {cnt}", "best", mmin, mmin / num_edge,
          best_config)
    return [mmin, best_config]


# int32 version
# for param in params:
#     print(param)
#     res = cxgnncomp.prof("spmm", "config", lambda: cxgnncomp_backend.run_spmm_configurable_int32(
#         ptr_for_dgsparse, idx_for_dgsparse, x, ptr.shape[0] - 1, *param))
#     if res[1] < mmin:
#         mmin = res[1]
#         best_config = param
#     print("res", res[1], mmin, param, best_config)
