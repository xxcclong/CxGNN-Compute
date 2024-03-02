To reproduce Figure 13

### Run the tests

Before running, change the implementation of PyG in its source code.
In `/PATH/TO/torch_geometric/backend.py`, change line 11 from 
```python
use_segment_matmul: Optional[bool] = None
```
to
```python
use_segment_matmul: Optional[bool] = True
```
else, PyG + RGCN will be very slow.

#### Run all test using `run.sh`

```bash
bash run.sh | tee output.txt
```

### Get the results

```bash
# It fetches all the execution time for different systems, models, and datasets
# Missing numbers are due to OOM error
cat output.txt | grep "ans"
```
