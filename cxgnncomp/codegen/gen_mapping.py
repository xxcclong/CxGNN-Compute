import torch
import cxgnncomp
import cxgnncomp_backend
import time
import random


def validate(cpw, rpb, cpb, block_size, feat_len):
    valid = 1
    if cpb % cpw != 0:
        valid = 0
    if block_size > 1024 or block_size <= 0:
        valid = 0
    if cpb > feat_len:
        valid = 0
    if cpw > 256:
        valid = 0
    return valid


class Tuner():

    def __init__(self, lazy=False):
        self.cache = {}
        self.cpws = [32, 64, 128, 256, 512, 1024]
        self.rpbs = [1, 2, 4, 8, 16, 32]
        self.cpbs = [32, 64, 128, 256, 512, 1024]
        self.block_xs = [32, 64, 128, 256, 512, 1024]
        self.grid_ys = [1, 2, 4, 8, 16, 32]
        self.block_maps = [0, 1]
        self.grid_maps = [0, 1]
        self.lazy = lazy

    def hash(
        self,
        num_node,
        num_edge,
        feat_len,
        func,
        run_param,
    ):
        if self.lazy:
            s = f"{func.__name__}_lazy"
        else:
            s = f"{num_node}_{num_edge}_{feat_len}_{func.__name__}"
            for item in run_param:
                if isinstance(item, torch.Tensor):
                    s += f"_{item.shape}"
                else:
                    s += f"_{str(item)}"
        return s

    def tune_graph(self,
                   num_node,
                   num_edge,
                   feat_len,
                   func,
                   run_param,
                   patience=50):
        num_node = int(num_node)
        num_edge = int(num_edge)
        feat_len = int(feat_len)
        hash_str = self.hash(num_node, num_edge, feat_len, func, run_param)
        ceil_feat_len = (feat_len + 31) // 32 * 32
        if hash_str in self.cache:
            if self.lazy:
                grid_x, grid_y, block_x, block_y, rpb, cpb, cpw, grid_map, block_map = self.cache[
                    hash_str]
                num_grid = num_node * ceil_feat_len // rpb // cpb
                print(grid_x, type(grid_x))
                grid_x = (num_grid + grid_y - 1) // grid_y
                grid_x = (grid_x + (feat_len // rpb) -
                          1) // (feat_len // rpb) * (feat_len // rpb)
                output = func(*run_param, grid_x, grid_y, block_x, block_y,
                              rpb, cpb, cpw, grid_map, block_map)
            else:
                output = func(*run_param, *self.cache[hash_str])
            return output

        params = []
        for rpb in self.rpbs:
            for cpb in self.cpbs:
                for cpw in self.cpws:
                    block_size = rpb * 32 * (cpb // cpw)
                    num_grid = num_node * ceil_feat_len // rpb // cpb
                    # print(cpw, rpb, cpb, block_size)
                    if not validate(cpw, rpb, cpb, block_size, feat_len):
                        continue
                    for block_x in self.block_xs:
                        if block_x > block_size:
                            continue
                        if block_size % block_x != 0:
                            continue
                        block_y = block_size // block_x
                        for grid_y in self.grid_ys:
                            grid_x = (num_grid + grid_y - 1) // grid_y
                            grid_x = (grid_x + (feat_len // rpb) - 1) // (
                                feat_len // rpb) * (feat_len // rpb)
                            for block_map in self.block_maps:
                                for grid_map in self.grid_maps:
                                    if block_x < (cpb // cpw *
                                                  32) and block_map == 0:
                                        continue
                                    if block_y < (cpb //
                                                  cpw) and block_map == 1:
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
            if res[1] < mmin:
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
              time.time() - t0, f"trial-num {cnt}", "best", mmin,
              mmin / num_edge, best_config)
        assert best_config is not None
        self.cache[hash_str] = best_config
        output = func(*run_param, *best_config)
        return output

    # int32 version
    # for param in params:
    #     print(param)
    #     res = cxgnncomp.prof("spmm", "config", lambda: cxgnncomp_backend.run_spmm_configurable_int32(
    #         ptr_for_dgsparse, idx_for_dgsparse, x, ptr.shape[0] - 1, *param))
    #     if res[1] < mmin:
    #         mmin = res[1]
    #         best_config = param
    #     print("res", res[1], mmin, param, best_config)
