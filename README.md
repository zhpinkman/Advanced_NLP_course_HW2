# Advanced_NLP_course_HW2
This assignment is about using the infrastructure you built in HW1, now applying a non-linear classifier (specifically, a one-layer feed-forward network) to the problem. The key is that we want you to implement the model by hand, without using neural network or machine learning frameworks. Here is a nice post on why such exercise of implementing your own forward and backward passes is useful. After you do this, you should compare your implementation to a PyTorch implementation (that you should also write) and to the results you got in HW1.


get the embedding files from [link](https://drive.google.com/file/d/1DVcwpvvP2j8EjqmUROZKzFkzoF8HXDC4/view).


All the experiments, training and evaluation scripts are in `pipeline.sh`, `torch_pipeline.sh`. 

There are a couple of notes that you have to consider before running the models

    The embedding files should contain the UNK token also included as a separate token
    Also if you wanted to run the models on a slurm based cluster you can use the scripts `cluster_pipeline.sh`, `cluster_torch_pipeline.sh`
    