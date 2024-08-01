# Software Documentation


# How to Setup?

### Step 1: Download the required weights:
<ul>
 <li> Use the following custom ONNX weights for <a href="https://github.com/pranav-kale-01/obstacle-detection-and-avoidance-using-depth-estimation/releases/download/v1.0.1/test.onnx" > MIDAS.</a> and put the weights in ./midas_weights directory </li> 

 <li> Use the following custom weights for <a href="https://github.com/pranav-kale-01/obstacle-detection-and-avoidance-using-depth-estimation/releases/download/v1.0.1/newbestyolov8.pt"> YOLO.</a> and put the weights in the root directory of the project "./" </li>
</ul>

### Step 2: setup environment

  Create a virtual environment, activate the environment, and install the dependencies. Use the following command to install all the required libraries ( Use timm==0.6.12 as specified in requirements ).
  
   ```
   pip install -r requirements.txt
   ```

### Step 3: run the inference file 

  ```
    python infer.py
  ```
