# inference_code 

<br/><br/>

## Step 1: Download the required weights: 


 ### Download the weights for MIDAS
   Use the following custom weights for [MIDAS.](https://github.com/pranav-kale-01/inference_code/releases/download/v0.0.1/dpt_swin2_tiny_256.pt). Put the weights in ./midas_weights directory 

 ### Download the weights for YOLO
   Use the following custom weights for [YOLO.](https://github.com/pranav-kale-01/inference_code/releases/download/v0.0.1/yolov5x.pt). Put the weights in the root directory of the project "./"

<br/>

## Step 2: setup environment

  Create a virtual environment, activate the environment, and install the dependencies. Use the following command to install all the required libraries ( Use timm==0.6.12 as specified in requirements ).
  
   ```
   pip install -r requirements.txt
   ```


## run the inference file 

  ```
    python infer.py
  ```
