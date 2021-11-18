# Simple_3DCNN

Here I use pretrained C3D model on several downstreams tasks. You can click here to obtain the pretrained C3D model.

## 1.Experiment on different tasks

### 1.1.AQA Task

Here I use pretrained C3D model as the backbone, and randomly initialize a two layers fully conneted neural network as the regressor. I use MSELoss and SGD optimizer. I trained the model on UNLV Dive dataset for 100 epochs in two steps. First, since the average loss is so big at the beginning, I set the lr=0.000001 and train the model for 5 epochs and save the checkpoint. Second, I load the checkpoint and increase the learning rate to 0.001 to train the model continuely. After the model has been trained for 100 epochs, the training loss has reduced to xxx, and the testing loss becomes xxxx. The visualization result is shown below.
