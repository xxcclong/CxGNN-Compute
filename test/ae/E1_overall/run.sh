# dsets=(arxiv products reddit papers100M-sample-1000 friendster-sample-1000)
dsets=(papers100M-sample-1000 friendster-sample-1000)
models=(LSTM)
# models=(GAT SAGE GCN RGCN)
graph_types=(CSR_Layer)
# graph_types=(DGL CSR_Layer)

for graph_type in ${graph_types[@]}; do
    for model in ${models[@]}; do
        for dset in ${dsets[@]}; do
            echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type}"
            python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type}
        done
    done
done

