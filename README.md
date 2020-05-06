# GAN-BERT

Code for the paper "GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples" accepted for publication at ACL 2020 - short papers by Danilo Croce (Tor Vergata, University of Rome), Giuseppe Castellucci (Amazon) and Roberto Basili (Tor Vergata, University of Rome).

This code runs the GAN-BERT experiment over the TREC dataset for the fine-grained Question Classification task. We provide in this package the code as well as the data for running an experiment by using 2% of the labeled material (109 examples) and 5343 unlabeled examples.
The test set is composed of 500 annotated examples.

## Requirements

The code is a modification of the original Tensorflow code for BERT (https://github.com/google-research/bert).
It has been tested with Tensorflow 1.14 over a single Nvidia V100 GPU. The code should be compatible with TPUs, but it has not been tested on such architecture or on multiple GPUs.
Moreover, it uses tf_metrics (https://github.com/guillaumegenthial/tf_metrics) to compute some performance measure.

## Installation Instructions
It is suggested to use a python 3.6 environment to run the experiment.
If you're using conda, create a new environment with:

```
conda create --name ganbert python=3.6
```

Activate the newly create environment with:

```
conda activate ganbert
```

And install the required packages by:

```
pip install -r requirements.txt
```

This should install both Tensorflow and tf_metrics.

## How to run an experiment

The run_experiment.sh script contains the necessary steps to run an experiment with both BERT and GANBERT.

The script can be launched with:

```
sh run_experiment.sh
```

The script will first download the BERT-base model, and then it will run the experiments both with GANBERT and with BERT.

After some time (on a Nvidia Tesla V100 it takes about 5 minutes) there will be two files in output: *qc-fine_statistics_BERT0.02.txt* and *qc-fine_statistics_GANBERT0.02.txt*. These two contain the performance measures of BERT and GANBERT, respectively.

## Out-of-memory issues

As the code is based on the original BERT Tensorflow code and that it starts from the BERT-base model, the same batch size and sequence length restrictions apply here based on the GPU that is used to run an experiment.

Please, refer to the BERT github page (https://github.com/google-research/bert#out-of-memory-issues) to find the suggested batch size and sequence length given the amount of GPU memory available.