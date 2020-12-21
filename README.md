
# PTAN

PTAN stands for PyTorch AgentNet -- reimplementation of
[AgentNet](https://github.com/yandexdataschool/AgentNet) library for
[PyTorch](http://pytorch.org/)

This library was used in ["Deep Reinforcement Learning Hands-On"](https://www.packtpub.com/data/deep-reinforcement-learning-hands-on-second-edition) book, here you can find [sample sources](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On).


## Code branches
The repository is maintained to keep dependency versions up-to-date. 
This requires efforts and time to test all the examples on new versions, so, be patient.

The logic is following: there are several branches of the code, corresponding to 
major pytorch version code was tested. Due to incompatibilities in pytorch and other components,
**code in the printed book might differ from the code in the repo**.

At the moment, there are the following branches available:
* `master`: contains the code with the latest pytorch which was tested. At the moment, it is pytorch 1.7.
* `torch-1.3-book-ed2`: code printed in the book (second edition) with minor bug fixes. Uses pytorch=1.3 which 
is available only on conda repos.
* `torch-1.7`: pytorch 1.7. Merged with master.

All the branches uses python 3.7, more recent versions weren't tested.

## Installation

From sources:
```bash
python setup.py install
```

From pypi:
```bash
pip install ptan
```

From github:
```bash
pip install pip install git+https://github.com/Shmuma/ptan.git 
```

## Requirements

* [PyTorch](http://pytorch.org/): version 1.1.0 is required
* [PyTorch Ignite](https://pytorch.org/ignite/): provides extra bindings for ignite
* [OpenAI Gym](https://gym.openai.com/): ```pip install gym gym[atari]```
* [Python OpenCV](https://pypi.org/project/opencv-python/): ```pip install opencv-python```
* [TensorBoardX](https://github.com/lanpa/tensorboardX): ```pip install tensorboardX```

### Note for [Anaconda Python](https://anaconda.org/anaconda/python) users

To run some of the samples, you will need these modules:
```bash
conda install pytorch torchvision -c pytorch
pip install tensorboard-pytorch
pip install gym
pip install gym[atari]
pip install opencv-python
```

## Documentation

* [Ptan introduction](docs/intro.ipynb)

Random pieces of information

* `ExperienceSource` vs `ExperienceSourceFirstLast`: https://github.com/Shmuma/ptan/issues/17#issuecomment-489584115
