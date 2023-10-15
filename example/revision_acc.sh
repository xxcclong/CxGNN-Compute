dsets=(arxiv products papers100M)
models=(sage gat)
baselines=(cxg dgl)
for i in {1..4}; do
for model in ${models[@]}; do
for dset in ${dsets[@]}; do
for baseline in ${baselines[@]}; do
        python3 train.py dl/dataset=${dset} train/model=${model} dl.type=cxg train.type=${baseline} \
                dl.loading.feat_mode=uvm dl.sampler.train.batch_size=1000 \
                train.train.num_epochs=30  \
                dl.num_device=1 dl.performance.prof=0 dl.performance.bind_method=0 \
                train.train.eval_begin=20 dl.device=cuda:2
        # python3 train.py dl/dataset=mag240M train/model=gat dl.type=cxg train.type=dgl \
        #         dl.loading.feat_mode=uvm dl.sampler.train.batch_size=1000 \
        #         train.train.num_epochs=40  \
        #         dl.num_device=1 dl.performance.prof=0 dl.performance.bind_method=0 \
        #         train.train.eval_begin=0 dl.device=cuda:3 train.model.heads=1 dl.sampler.train.fanouts=[15,25] \
        #         dl.sampler.train.num_layer=2 dl.sampler.eval.fanouts=[30,30] dl.sampler.eval.num_layer=2 \
        #         train.model.hidden_dim=1024
done
done
done
done
