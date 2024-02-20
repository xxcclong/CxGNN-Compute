Run the tests

```bash
bash run.sh | tee output.txt
```

Get the results

```bash
# It fetches all the execution time for different systems, models, and datasets
# Missing numbers are due to OOM error
cat output.txt | grep "ans"
```