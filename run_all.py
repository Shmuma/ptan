import subprocess
import json
jobs = [
    {
      "epsilon_frames": 10 ** 6,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv8"
    },
    {
      "epsilon_frames": 10 ** 6 / 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv4"
    },
    {
      "epsilon_frames": 10 ** 6 * 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "local"
    },
    {
      "epsilon_frames": 10 ** 6 * 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv4"
    }

]  # list of dictionaries (json)

cloud = []
local = []

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

subprocess.check_output("python run_sequential.py -p --file cloud.json &"
                        " python run_local.py --file local.json &", shell=True)