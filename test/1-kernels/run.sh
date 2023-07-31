# dsets=(papers100M-sample-1000 wiki90m-sample-1000)
dsets=(arxiv products reddit papers100M-sample-1000 wiki90m-sample-1000)
# dsets=(arxiv products reddit)
# models=(RGCN GAT SAGE GCN)
# dsets=(products reddit)
models=(RGCN)
# graph_types=(PyG)
graph_types=(PyG DGL CSR_Layer)
# graph_types=(DGL)

for graph_type in ${graph_types[@]}; do
    for model in ${models[@]}; do
        for dset in ${dsets[@]}; do
            echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type}"
            python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type}
        done
    done
done

