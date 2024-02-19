dsets=(arxiv)
# dsets=(arxiv products papers100M)
models=(sage)
baselines=(cxg dgl)
# for i in {1..4}; do
for model in ${models[@]}; do
for dset in ${dsets[@]}; do
for baseline in ${baselines[@]}; do
        python train.py dl/dataset=${dset} train/model=${model} dl.type=cxg train.type=${baseline} \
               dl.loading.feat_mode=uvm dl.sampler.train.batch_size=1000 \
               train.train.num_epochs=100  \
               dl.num_device=1 dl.performance.prof=0 dl.performance.bind_method=0 \
               train.train.eval_begin=0 dl.device=cuda:0
done
done
done
