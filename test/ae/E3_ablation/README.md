## Test data batching

```bash
python test-batch.py --model rgcn | tee output_batch_rgcn.txt
python test-batch.py --model lstm | tee output_batch_lstm.txt
```

After running, use `process-batch-data.ipynb` to visualize the data.

## Test data duplication

Use `test-dup.ipynb` to run the experiments and get the data.

## Test data batching

Use `test-comm-hidden.ipynb` to run the experiments and get the data.
