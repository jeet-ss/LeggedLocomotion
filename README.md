# ANN based Controller design for Artificial Leg Prostheses
Implementation of different Artificial Neural Network architectures to design an efficient controller for Artificial Leg prostheses. 

This is done as a coursework for Seminar titled "Legged Locomotion in Robotics" at Friedrich-Alexander University, Erlangen.

Contributors: Jeet Sen Sarma

Special Thanks to: Prof. Dr. Anne Koelewijn


## Files description
1. Different models are in "utils/model.py"
2. Datasets are loaded using "utils/dataloader.py"
3. Helper function to convert to Sequence data is in "utils/helpers.py"
4. Training Methodology is present is "utils/training_script.py"
5. We train the model with our data in "train.py" and store the loss values in a numpy dictionary
6. Evaluation for scores can be found in "scores.py"
7. Generation of plots can be found in "test.ipynb"
    
    #### NB:
    Due to the different architectures of different models, there are some changes that need to be done to run each of them.
    1. In "train.py", there are two mini-batching code for the data, the first one is for FC like models, the second one for models requiring time sequence input.

    2. In utils/training_script.py", the two methods "train_epoch" and "val_epoch" each contain two sets of code for input. Here the input structure is different for each of FC, CNN and LSTM models. Details can be found in the comments at those places. 