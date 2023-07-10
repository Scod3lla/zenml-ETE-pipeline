# zenml-ETE-pipeline


Simple example of a zenml pipeline using Pytorch and Weight and Biases. Enviroment installation:

> conda env create -f zenml.yml

> conda activate zenml


In order to launch the ZenML dashboard:

> zenml up

From here is possible to manage the stacks, to the run the code a stack with Weight and Biases is required.


In order to run the code:

> python run.py

Possible additional parameter are:
- -lr for learning rate
- -ep for epochs
- -bs for batch size