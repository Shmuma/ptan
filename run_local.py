import subprocess
import json
import time
import argparse

"""
Before running, run 'ngc config set' and set the following:
Debug Mode: False
CLI output format type: json
"""
frame_stop = 5000

jobs = [
    {
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "fsa": True
    },
    {
        "epsilon_frames": 10 ** 6 / 2,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99
    },
    {
        "epsilon_frames": 10 ** 6 * 2,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "fsa": True
    }

]  # list of dictionaries (json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    job_number = 0
    for job in jobs:
        # start the first job
        config = json.dumps(job)
        config = ''.join(config.split())  # remove spaces from config string
        command = "echo '" + config + "' > config.json && python samples/dqn_speedup/05_new_wrappers.py " \
                                      "--cuda --file config.json --stop " + str(frame_stop)
        if args.v:
            print(command)
        result = subprocess.check_output(command, shell=True)

        result = subprocess.check_output("mkdir results/", str(job_number))
        result = subprocess.check_output("mv results/output.txt results/"+str(job_number))
        result = subprocess.check_output("mv results/model results/" + str(job_number))
        result = subprocess.check_output("mv results/video results/" + str(job_number)
        job_number+=1