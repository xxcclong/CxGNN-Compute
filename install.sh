# pip install torch==2.0.1
# cd triton/python
# python setup.py build -j32 develop
# cd ../../
# cd ../CxGNN-DL
# python setup.py build -j32 develop
# cd ../CxGNN-Compute
# python setup.py build -j32 develop

pip install  dgl==1.0.0 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install -r requirements.txt
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html