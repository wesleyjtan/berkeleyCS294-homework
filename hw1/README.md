# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

### Additional
In order to run behavorial cloning or DAgger, the data generated from expert policies is required. Based on [Humanoid-v2] environment, to generate the data, run:

`python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 20`

For behavorial cloning, run:

`    python run_behavioral_cloning.py experts/Humanoid-v2.pkl Humanoid-v2 data_experts/Humanoid-v2_20_data.pkl --num_rollouts 20`

For DAgger algorithm, run:

`    python run_dagger.py experts/Humanoid-v2.pkl Humanoid-v2 data_experts/Humanoid-v2_20_data.pkl --num_rollouts 20`