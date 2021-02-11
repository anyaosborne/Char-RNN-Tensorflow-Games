# Char-RNN-Tensorflow-Games

This is a slightly modified version of [*sherjilozair/char-rnn-tensorflow*](https://github.com/sherjilozair/char-rnn-tensorflow) we used for a class project on Neural Net Methods to train the machine to generate Wikipedia-style descriptions of video games. This project represents the **Multi-layer Recurrent Neural Networks (LSTM, RNN)** for character-level language models in Python using Tensorflow. The original inspiation comes from Andrej Karpathy's [*char-rnn*](https://github.com/karpathy/char-rnn). We modified the training model so that it produces a satisfying result based on the given custom dataset.

## Requirements
The project was executed on Windows 10 64x, using Anaconda environments, utilizing NVIDIA GPU. Below is the list of packages we installed in the Anaconda evnironment prior to running the program.
* **TensorFlow 1.15.0** (tensorflow-gpu==1.15) - We used the following installation guide - [Anaconda TensorFlow](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/?highlight=tensorflow). 
* Install **pip** - We used [pip 21.0.1](https://anaconda.org/conda-forge/pip)
* **Python 3+** - We used Python 3.7.9, but it can work with 2.7, 3.6, 3.5, and 3.4.
* Install **numpy** into your environment. The version we used is 1.20.0. Here is a guide on [Installing NUMPY](https://numpy.org/install/).
* Lastly, you will likely need to install **six 1.15.0**. [Six](https://pypi.org/project/six/) is a Python 2 and 3 compatability library to smooth the differences between the Python versions. 

## How to Execute The Code
1. To train, run [train.py](https://github.com/anyaosborne/Char-RNN-Tensorflow-Games/blob/main/train.py). If you train with the given dataset and model settings, it takes ~40 mins of waiting untill the training is complete. The command to run the training is `python train.py` (make sure the path in the evnironment is correct). To access all the parameters use `python train.py --help`.
2. In result of training, it will create new [logs](https://github.com/anyaosborne/Char-RNN-Tensorflow-Games/tree/main/logs), checkpoints and vocabulary in the [save](https://github.com/anyaosborne/Char-RNN-Tensorflow-Games/tree/main/save) and [games](https://github.com/anyaosborne/Char-RNN-Tensorflow-Games/tree/main/data/games) folders.
3. The sample output is generated from the saved checkpoints. To receive one, run `python sample.py`. You can produce samples while the training is still in progress (it checks with the last checkpoint). However, it works only in CPU or using another GPU. To force CPU mode, use `export CUDA_VISIBLE_DEVICES=""` and `unset CUDA_VISIBLE_DEVICES` afterward
(resp. `set CUDA_VISIBLE_DEVICES=""` and `set CUDA_VISIBLE_DEVICES=` on Windows). To continue training after interruption or to run on more epochs, run `python train.py --init_from=save`.

## About Dataset
You can use any plain text file as input. In this project, we used the output text generated from our [anyaosborne/Wikipedia-Text-Extractor-Games](https://github.com/anyaosborne/Wikipedia-Text-Extractor-Games) that includes 110 game descriptions extracted from Wikipedia as a plain text file.

## Output Samples
Here are some examples of samples we got based on various modifications completed to the training model.

#### Sample From the Inital Training
This one is based on full descriptions of 110 games:
```
can only direct story moming out of Game of the game is a villuse "the test dead in to the form, while they had screen is not how they 
have blog to the class, a moving months atcess that they could be reward the Campaigoni (on Otton 64 and BioShock 2 was the past
of Japan better up at the same you began your ball of set in the game not this game to terms in hosper been "itsumes the besout to 
sporce the live across the lawmens of the Computing them their story, the use the game's best launchers,
```
#### Sample From a Different Input File
In experimenting with the input file, we decided to use a shorter version with a repetitive vocabulary. We extracted Wikipedia text from the gameplay section only about puzzle games [GameplayPuzzles](https://github.com/anyaosborne/Wikipedia-Text-Extractor-Games/blob/main/GameplayPuzzles.txt). Here is the sample we received.

###### 659/660 (epoch 109), train_loss = 2.161, time/batch = 0.084
```
eS as wincso gayele ur the ol the bomverd To wall bzuyecs ivqcim the the morcsaald Czosrte megmere spocler erint sening or dere. 
thing a saxo fupsaic bactor the an cleecutes the beres. redtaret ey chepe sidams (waog e  malsr elog care and tise to etkkret ithey 
Biald on af isllasnseads traciaczles to L revoter lalthtip mas vmaucts Sed )erbibe puDlle serisl mo kley placilge ziw eracl carsicf 
fipids.sdith the te gker somt.s fiwe semdith ix Goces molings the mond thse cacpes bales on fom silh a whel
```
The train loss during this experiment turned out to be pretty high, so we decided to return to the initial input file.

#### Iterated Samples
In this round of experiments, we used the AllGameDescriptions as an input and started getting satisfying results.

##### 32699/32700 (epoch 149), train_loss = 1.184, time/batch = 0.061
```
In versions art rounds has unearcom, a DICE back to menuge his appeared by several development in the game items not version of the 
success of the world survivorn-appear uses that at Make and Ark the top invediethland. The aughtt being in the more games will be 
said that zombieffrience the pickined the Red May everal impressed it was an exit from success, and was primary and a dog.
```

##### 15259/15260 (epoch 69), train_loss = 0.549, time/batch = 0.153  (rnn size 400)
```
had previous installments, including Square Enix. Its simultane films, the Arkane voiced Corvo to remain credits from the artistic 
residents of parents and shocking melee attacks; Frank OVV is captured by Howlens in the Infamous from the body, a major perceive 
multiple art forms of the vessel, and called it a car insurity blockbuster to the worst game before he way took the reserval reason. 
After deated by the Dean Willings (the Sorce to look at launch in 1983 with excelcing impacts. It is playa
```

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
