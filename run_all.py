import subprocess
import json
import argparse

jobs = [
    {
      "epsilon_frames": 10 ** 6,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv8",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3000001,
      "dqn_model": "FSADQNParallel"
    },
    {
      "epsilon_frames": 10 ** 6 / 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv4",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3000001,
      "dqn_model": "FSADQNParallel"
    },
    {
      "epsilon_frames": 10 ** 6 * 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv2",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3000001,
      "dqn_model": "FSADQNParallel"
    },
    {
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00015,
        "gamma": 0.99,
        "fsa": True,
        "machine": "ngcv8",
        "replay_initial": 50000,
        "video_interval": 1000000,
        "frame_stop": 3000001,
        "dqn_model": "FSADQNParallel"
    },
    {
        "epsilon_frames": 10 ** 6 / 2,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00015,
        "gamma": 0.99,
        "fsa": True,
        "machine": "ngcv4",
        "replay_initial": 50000,
        "video_interval": 1000000,
        "frame_stop": 3000001,
        "dqn_model": "FSADQNParallel"
    },
    {
        "epsilon_frames": 10 ** 6 * 2,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00015,
        "gamma": 0.99,
        "fsa": True,
        "machine": "ngcv2",
        "replay_initial": 50000,
        "video_interval": 1000000,
        "frame_stop": 3000001,
        "dqn_model": "FSADQNParallel"
    },
    {
      "epsilon_frames": 10 ** 6,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00025,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv1",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3000001,
      "dqn_model": "FSADQN"
    },
    {
      "epsilon_frames": 10 ** 6 / 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00025,
      "gamma": 0.99,
      "fsa": True,
      "machine": "local",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3000001,
      "dqn_model": "FSADQN"
    },
    {
      "epsilon_frames": 10 ** 6 * 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00025,
      "gamma": 0.99,
      "fsa": True,
      "machine": "local",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3000001,
      "dqn_model": "FSADQN"
    }

] # list of dictionaries (json)

cloud = []
local = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    for job in jobs:
        if "machine" not in job:
            print("Machine not specified in job: ", job)
        elif job["machine"] == "local":
            local.append(job)
        else:
            cloud.append(job)

    with open("cloud.json", "w") as f:
        f.write(json.dumps(cloud))
    with open("local.json", "w") as f:
        f.write(json.dumps(local))

    if args.v:
        subprocess.call("python run_local.py -v --file local.json & "
                        "python run_sequential.py -v -p --file cloud.json", shell=True)
    else:
        subprocess.call("python run_local.py --file local.json & "
                        "python run_sequential.py -p --file cloud.json", shell=True)
