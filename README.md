### Installation

Please follow the pip installation guide from [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

### Usage

To train, please run

````
python train.py --device cuda --proj_name t1 --task t1_odom_parkour --exptid t1_odom_001
````

--proj_name specifies the name of the root directory of each experiment.

--task specifies task config you want to run.

--exptid names the current experiment run.

--resumeid specifies the experiment you want to resume from.

Refer to train.py for additional arguments.  


### Convert URDF to USD

Since IsaacSim only support importing USD file to simulation, you should convert your URDF file to USD format.

To do this, run 

````

./isaaclab.sh -p scripts/tools/convert_urdf.py \
path/to/your.urdf \
path/to/store.usd \
--merge-joints   \
--joint-stiffness 0.0   \
--joint-damping 0.0   \
--joint-target-type none

````

The output is not just a single USD file, so you'd better put them under a directory.
