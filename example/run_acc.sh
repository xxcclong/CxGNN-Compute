python3 train.py dl/dataset=arxiv train/model=rgcn dl.type=cxg train.type=cxg \
        dl.loading.feat_mode=uvm dl.sampler.train.batch_size=1000 \
        train.train.num_epochs=30  \
        dl.num_device=1 dl.performance.prof=0 dl.performance.bind_method=0 \
        train.train.eval_begin=15 dl.device=cuda:3
