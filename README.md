Main Code for Code Retrieval Model

### Repository Structures

 - `/code`: All code for model training and evaluation
 - `/data`: Directory for data and model checkpoints
 
### Code Strcuture
 - `codesearcher.py`: The main for Code Retrieval Model: 
 - `config.py`: Configurations for models defined in the `models.py`. 
   Each function defines the hyper-parameters for the corresponding model.
 - `data.py`: dataset loader.
 - `utils.py`: Utilities for models and training. 
 
### Usage
  The `/data` folder provides all the datasets
  
  1) Main staQC dataset
  2) CodeNN dataset
  3) CodeNN sanity check dataset (sanity - pad, unk)
  4) staQC dataset to check generalization to Code Annotation (anno)
  5) staQC sanity check dataset (sanity check - code empty, QB empty)
  
  To train and test our model:
  ### Configuration
  1. Edit hyper-parameters and settings in `config.py`
  
 ### Model Using QB( Best Model)
 ```diff
 - Dont run training without changing parameters, it will override saved best models
 ```
 
 #### Both 512 and 256 batchsizes produced similar results. I have placed both the weights in the data directory.
 
  ### Train
   ```bash
   python codesearcher.py --mode train --use_qb 1 --code_enc bilstm --reload -1 --dropout 0.25 --emb_size 200 --lstm_dims 400 --batch_size 256
   ```
   
   ```bash
    python codesearcher.py --mode train --use_qb 1 --code_enc bilstm --reload -1 --dropout 0.25 --emb_size 200 --lstm_dims 400 --batch_size 512
   ```
   
   ### Evaluation
   
   ```bash
   python codesearcher.py --mode eval --use_qb 1 --code_enc bilstm --reload 1 --dropout 0.25 --emb_size 200 --lstm_dims 400 --batch_size 256
   ```
   
   ```bash
   python codesearcher.py --mode eval --use_qb 1 --code_enc bilstm --reload 1 --dropout 0.25 --emb_size 200 --lstm_dims 400 --batch_size 512
   ```
   
   
   
### Model without QB

  ### Train
  ```bash
  python codesearcher.py --mode train --use_qb 0 --code_enc bilstm --reload -1 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 1024
  ```

  ### Evaluation
  
  ```bash
  python codesearcher.py --mode eval --use_qb 0 --code_enc bilstm --reload 1 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 2014
  ```




