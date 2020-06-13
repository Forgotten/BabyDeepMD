# Baby Deep MD

The code is written in python 3.6 and Tensorflow 2.x (so far 2.1 seems to be the one with fewer issues) 

folder structure: 
src/ : contains the source files
run/ : it contains the examples encoded in a Json file

you will need to create another folder data/ where all the data will be stored, i.e., you need to type
mkdir data


For starters you can run a simple 1D example by 
cd run/Np20_Per
makedir checkpoints 
python ../../src/2BodyForcesDist.py Np20_Per_mu_10.json

This will create the dataset (if it wasn't created before), it will create the graph and it will start training 
the network. 

For an optimized version of the same example you can type 
python ../../src/2BodyForcesDistArray.py Np20_Per_mu_10.json

This code uses a TensorArray to concatenate the results, thus helping tensorflow with the tracing and optimization. 
 