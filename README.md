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

### Warning

Because our code was originally based on the tensorflow1.9 framework，and in order to run more succinctly, we recommend that you use tensorflow 1.9 or higher (not more than 2.0).

Of course, our code can also be used in tensorflow2.0 or higher, you only need to do a little modification.
We provide two methods for you to choose，you just need to follow the instructions below.

1、Replace the tensorflow reference "import tensorflow as tf" with the following two sentences

		improt tensorflow.compat.v1 as tf
		tf.disable_v2_behavior()

Only the above, other code does not need to be modified. We think that if you only use the code steadily and have no intention of refactoring, this method is the first choice.

2、TensorFlow 2.0 provides a command line migration tool to automatically convert 1.x code to 2.0 code, use this

		tf_upgrade_v2 --infile first-tf.py --outfile first-tf-v2.py

For example
		tf_upgrade_v2 --infile main.py --outfile main-v2.py

You can choose a certain code file to perform the above operations, or perform an overall conversion operation on our entire project folder, use this

		tf_upgrade_v2 -intree foo_v1 -outtree foo_v2
For example
		
		tf_upgrade_v2 -intree /home/OptimizationofSEandEEbasedonDRL -outtree /home/OptimizationofSEandEEbasedonDRL_v2

Then you can happily use tensorflow2.0 to run our project.

## Running the tests


### How to run

After all the configurations are installed,

If you want to run the program directly, cd to the <OptimizationofSEandEEbasedonDRL> folder, and directly 

		Python main.py

so that all the settings of the program will be the default values, and dqn network training will be carried out according to the established way, and the training steps will be printed out.

However, I highly recommend that you use 

		Python main.py - h 
		Python main.py --help

to check the documentation to better understand the command line parameters.

Of all the command-line parameters, "Num_vehicles" is the most important one, which represents the total number of vehicles added in the simulation environment.

In general, you only need to use 

		Python main.py --num_vehicles n 

to run the program, where n is the number of vehicles you want to add to the environment, for example

		python main.py --num_vehicles 20

I highly recommend that you try 20 or 40 vehicles first, so that the training speed of the network is not too slow, and you can see the effect quickly.

If you want to know more, please check the code carefully.

### Data Record

The data from each run of the program is placed in a folder called <record>, and the data files are named as "the number of vehicles in the environment", "the start time of the program", and "the dqn model". Such as

		record/V_num_20-03-27-11-43_double_q&dueling_q.txt

The record data format is "V2V Efficiency: %f \tV2I Efficiency: %f\tSecurity Rate: %f\tCompound Efficiency: %f\tStep : %d\n"

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

  
