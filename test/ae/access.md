## How to get access to our server

** One can only log in our server using public key, so you need to provide your public key through e-mail or using hotcrp comment **

After providing your public key:

```bash
ssh eurosysae@166.111.68.163 -p 3330
ssh octave
```

Then you will be on our server for testing. The following loads the prepared environment for all experiments.

```bash
source /data/eurosysae/.venv/cxgnn/bin/activate
```