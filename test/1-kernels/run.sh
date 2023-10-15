# dsets=(arxiv products reddit papers100M-sample-1000 friendster-sample-1000)
# dsets=(arxiv papers100M-sample-1000 friendster-sample-1000)
# dsets=(friendster-sample-1000)
# dsets=(products reddit)
# dsets=(products)
dsets=(arxiv-ng)
# models=(GAT SAGE GCN)
# dsets=(products reddit)
# dsets=(papers100M-sample-1000)
models=(RGCN)
# models=(RGCN)
# graph_types=(PyG DGL CSR_Layer)
# graph_types=(PyG)
# graph_types=(DGL)
graph_types=(CSR_Layer)

for graph_type in ${graph_types[@]}; do
    for model in ${models[@]}; do
        for dset in ${dsets[@]}; do
            echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type}"
            python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type}
        done
    done
done

