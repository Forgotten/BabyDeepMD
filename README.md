# Baby Deep MD

The code is written in python 3.6 and Tensorflow 2.x (so far 2.1 seems to be the one with fewer issues) 

folder structure: 

-```src/``` : contains the source files
-```run/``` : it contains the examples encoded in a Json file
-```experimental/``` : it contains experimental features being explored

you will need to create another folder data/ where all the data will be stored, i.e., you need to type
```mkdir data```

The codes are built so they require a json file for execution. 
Several examples are provided in the the ```run/``` folder. In addition, I added a few bash files to be used  with slurm 


For starters you can run a simple 1D example by :

```
cd run/Np20_Per
makedir checkpoints 
python ../../src/2BodyForcesDist.py Np20_Per_mu_10.json
```

This will create the dataset (if it wasn't created before), it will create the graph and it will start training 
the network. 

For an optimized version of the same example you can type 
```python ../../src/2BodyForcesDistArray.py Np20_Per_mu_10.json```

(there is a bug in TF 2.0 that does not allow this code to run properly, be sure to have TF 2.1 installed)

This code uses a vectorized implementation which makes the codes much faster.
