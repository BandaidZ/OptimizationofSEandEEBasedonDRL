# OptimizationofSEandEEBasedonDRL
Optimization of SE and EE based on DRL
(Joint Optimization Analysis of SE and EE Considering the secrecy rates of V2V by using DQN model)
## Summary

  - [Getting Started](#getting-started)
  - [Runing the tests](#running-the-tests)
  - [Deployment](#deployment)
  - [Built With](#built-with)
  - [Contributing](#contributing)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started


### Prerequisites

The DQN model is built in TensorFlow 1.9.0 and the GPU model used is GeForce GTX 1080 Ti. 

The python version is 3.7

### Installing

Installing python3 is very simple, I don't believe I need to say more

You can search the installation method of tensorflow through Google. 
If you want to install the version of tensorflow GPU, you also need to check the information of graphics card supported by tensorflow GPU (it should be NVIDIA graphics card), driver information of graphics card, CUDA and cudnn corresponding to different versions. I'm sure you can configure all of these by Google or other ways.

If you use the version of tensorflow-2.0, there may be some small errors due to the version problems. You only need to adjust accordingly. However, in order to run more succinctly, we recommend that you use tensorflow 1.9 or higher (not more than 2.0)

## Running the tests

After all the configurations are installed,

If you want to run the program directly, cd to the "OptimizationofSEandEEbasedonDRL" folder, and directly 

		Python main.py

so that all the settings of the program will be the default values, and dqn network training will be carried out according to the established way, and the training steps will be printed out.

However, I highly recommend that you use 

		Python main.py - h 
		Python main.py --help

to check the documentation to better understand the command line parameters.

Of all the command-line parameters, "Num_vehicles" is the most important one, which represents the total number of vehicles added in the simulation environment.

In general, you only need to use 

		Python main.py --num_vehicles n 

to run the program, where n is the number of vehicles you want to add to the environment, for example "python main.py --num_vehicles 100".
I highly recommend that you try 20 or 40 vehicles first, so that the training speed of the network is not too slow, and you can see the effect quickly.

If you want to know more, please check the code carefully.

## Deployment


## Built With


## Contributing


## Versioning

We use [git](https://git-scm.com/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/BandaidZ/OptimizationofSEandEEBasedonDRL).

## Authors

    [BandaidZ](https://github.com/BandaidZ)

See also the list of
[contributors](https://github.com/BandaidZ/OptimizationofSEandEEBasedonDRL/contributors)
who participated in this project.

## License


## Acknowledgments

  
