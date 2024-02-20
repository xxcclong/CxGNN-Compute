Run the experiments

```
python tune.py --model gcn | tee gcn.txt
python tune.py --model lstm | tee lstm.txt
python tune.py --model gat | tee gat.txt
python tune.py --model rgcn | tee rgcn.txt
```

Get the results

```
cat rgcn.txt | grep ans 
```