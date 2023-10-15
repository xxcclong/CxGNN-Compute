dsets=(arxiv products reddit papers-sample friendster-sample)
models=(sage gcn)
for model in ${models[@]}; do
for dset in ${dsets[@]}; do
    python main_tcgnn.py --dataset $dset --dim -1 --hidden 256 --classes -1 --num_layers 3 --model $model
done
done