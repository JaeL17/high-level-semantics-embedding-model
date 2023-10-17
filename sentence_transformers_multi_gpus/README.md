# Overview
This repository contains the implementation of torch ```nn.DataParallel``` for multi-GPUs training of Sentence Transformers.

## Key Changes 
1. Wrapped the Transformer model with ```nn.DataParallel``` module in the ```Transformer.py```.
2. Modified some config details in ```modules.json``` and ```1_Pooling/config.json``` within the ```SentenceTransformer.py```.
   
