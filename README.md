## Char-RNN-Tensorflow-Games

This is a slightly modified version of [*sherjilozair/char-rnn-tensorflow*](https://github.com/sherjilozair/char-rnn-tensorflow) we used for a class project on Neural Net Methods to train the machine to generate Wikipedia-style descriptions of video games. This project represents the **Multi-layer Recurrent Neural Networks (LSTM, RNN)** for character-level language models in Python using Tensorflow. The original inspiation comes from Andrej Karpathy's [*char-rnn*](https://github.com/karpathy/char-rnn). We modified the training model so it produces a satisfying result based on the given custom dataset.

## Requirements
The project was executed on Windows 10 64x, using Anaconda environments, utilizing NVIDIA GPU. Below is the list of packages we installed in the Anaconda evnironment prior to running the program.
* **TensorFlow 1.15.0** (tensorflow-gpu==1.15) - We used the following installation guide - [*Anaconda TensorFlow*](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/?highlight=tensorflow).
* Install **pip** - We used [*pip 21.0.1*](https://anaconda.org/conda-forge/pip)
* **Python 3+** - We used Python 3.7.9, but it can work with 2.7, 3.6, 3.5, and 3.4.
* Install **numpy** into your environment. The version we used is 1.20.0

## Basic Usage
To train with default parameters on the games corpus, run `python train.py`. To access all the parameters use `python train.py --help`.

To sample from a checkpointed model, `python sample.py`.
Sampling while the learning is still in progress (to check last checkpoint) works only in CPU or using another GPU.
To force CPU mode, use `export CUDA_VISIBLE_DEVICES=""` and `unset CUDA_VISIBLE_DEVICES` afterward
(resp. `set CUDA_VISIBLE_DEVICES=""` and `set CUDA_VISIBLE_DEVICES=` on Windows).

To continue training after interruption or to run on more epochs, `python train.py --init_from=save`

## Datasets
You can use any plain text file as input. For example you could download [The complete Sherlock Holmes](https://sherlock-holm.es/ascii/) as such:

```bash
cd data
mkdir sherlock
cd sherlock
wget https://sherlock-holm.es/stories/plain-text/cnus.txt
mv cnus.txt input.txt
```

Then start train from the top level directory using `python train.py --data_dir=./data/sherlock/`

A quick tip to concatenate many small disparate `.txt` files into one large training file: `ls *.txt | xargs -L 1 cat >> input.txt`.

## Tuning

Tuning your models is kind of a "dark art" at this point. In general:

1. Start with as much clean input.txt as possible e.g. 50MiB
2. Start by establishing a baseline using the default settings.
3. Use tensorboard to compare all of your runs visually to aid in experimenting.
4. Tweak --rnn_size up somewhat from 128 if you have a lot of input data.
5. Tweak --num_layers from 2 to 3 but no higher unless you have experience.
6. Tweak --seq_length up from 50 based on the length of a valid input string
   (e.g. names are <= 12 characters, sentences may be up to 64 characters, etc).
   An lstm cell will "remember" for durations longer than this sequence, but the effect falls off for longer character distances.
7. Finally once you've done all that, only then would I suggest adding some dropout.
   Start with --output_rate 0.8 and maybe end up with both --input_rate 0.8 --output_rate 0.5 only after exhausting all the above values.

## Tensorboard
To visualize training progress, model graphs, and internal state histograms:  fire up Tensorboard and point it at your `log_dir`.  E.g.:
```bash
$ tensorboard --logdir=./logs/
```

Then open a browser to [http://localhost:6006](http://localhost:6006) or the correct IP/Port specified.


## Roadmap
- [ ] Add explanatory comments
- [ ] Expose more command-line arguments
- [ ] Compare accuracy and performance with char-rnn
- [ ] More Tensorboard instrumentation

## Contributing
Please feel free to:
* Leave feedback in the issues
* Open a Pull Request
* Join the [gittr chat](https://gitter.im/char-rnn-tensorflow/Lobby)
* Share your success stories and data sets!
