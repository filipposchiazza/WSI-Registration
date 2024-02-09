# Whole Slide Images Registration

This algorithm is used to register whole slide images (WSI) using the OpenSlide library. The registration is based on the extraction of the SIFT features, the matching of the features between the images and the evaluation of the trasformation parameters between the images (scale factor, rotation matrix and translation vector).
The algorithm is implemented in Python and uses the OpenSlide library to read the WSI.

## How to install

Move to the directory where you want to install the repository:
```
$ cd /directory/where/install/repository/
```
To install the repository run:
```
$ git clone https://github.com/filipposchiazza/WSI-Registration
```
Move in the installed directory:
```
$ cd WSI-Registration
```
Use the setup.py file to install the package:
```
$ python setup.py install
```

## How to use

This algorithm performs the following tasks:
- Global images registration at a specific WSI level
- Local images registration at all the WSI levels, with fixed trasformation parameters (S, R, T) obtained from the global registration
- Local images registration at all the WSI levels, with an iterative re-evaluation of the trasformation parameters (S, R, T) at each level

For a tutorial on how to use the algorithm, please refer to [Example.ipynb](Example.ipynb).


## Acknowledgments

The basic idea for the first global registration is taken from [this paper](https://www.nature.com/articles/s41598-022-15962-5)
