To run the model, at first create the Config file below and then run:
```bash
!python main.py --config_path='/content/config.json'
```
the args can define:
 - --config_path='path': Specify the config path
 - --*train*: Specify if the model should be trained 
 - --*load*: Specify if the last modified model should be loaded 
 - --*lr=0.0001*: Change the learning rate

Before using a model we have to define a Json config file with the following specifications:
 - model_name: the name of the model
 - data_path: the path of the data
 - test_path: the path of the data to predict
 - model_path: the path of the model
 - data_length: the length of the window
 - data_targetSD: the standard deviation of the onset gaussian PDF
 - model_layers: Specify the layers of the UNet
 - model_out: Specify the last convolution of the UNet
 - model_kernel: Specify the kernel used for the convolution
 - model_scale: Specify how much the data is downscaled in every layer
 - lr: learning rate
 - save_time: after how many seconds the model will be saved
 - train_split: the ratio of the train test split
 - num_epochs: the number of epochs that the model should be trained

As an example we have here:
```python
import json

# Create a dictionary to be saved as JSON
data = {
    "model_name": "onset",
    "data_path": "/content/drive/MyDrive/Share/Onset/dataset",
    "test_path": "/content/drive/MyDrive/Share/Onset/test",
    "model_path": "/content/drive/MyDrive/Share/Onset/model",
    "output_path": "/content/drive/MyDrive/Share/Onset/output",
    "data_length": 32768,
    "data_targetSD": 300,
    "model_layers": [32, 64, 96],
    "model_out": [96, 96, 64, 32],
    "model_kernel": 9,
    "model_scale": 2,
    "lr": 0.0001,
    "save_time": 500,
    "train_split": 0.95,
    "num_epochs": 10000
}

# Save dictionary as JSON to a file
with open("config.json", "w") as f:
    json.dump(data, f)
```