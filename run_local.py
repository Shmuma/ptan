import subprocess
import json
import argparse
import os

jobs = [
    {
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "fsa": True,
        "frame_stop": 3000
    },
    {
        "epsilon_frames": 10 ** 6 / 2,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "frame_stop": 3000
    },
    {
        "epsilon_frames": 10 ** 6 * 2,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "fsa": True,
        "frame_stop": 3000
    }

]  # list of dictionaries (json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default='', help="Input file")
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            jobs = json.loads(open(args.file, "r").read())

    curdir = os.path.abspath(__file__)
    exec_path = os.path.abspath(os.path.join(curdir, '../samples/dqn_speedup/05_new_wrappers.py'))
    json_path = os.path.abspath(os.path.join(curdir, '../config.json'))

    try:
        with open('job_number.txt', 'r') as f:
            job_number = int(f.read())
    except:
        with open('job_number.txt', 'w+') as f:
            f.write(str('0'))
            job_number = 0

    for job in jobs:
        # start the first job
        config = json.dumps(job)
        config = ''.join(config.split())  # remove spaces from config string
        echo_command = "echo '" + config + "' > " + json_path
        subprocess.call(echo_command, shell=True)

        if 'brandon' in curdir:
            python = "/home/brandon/packages/anaconda2/envs/fsaatari/bin/python3.6"
        else:
            python = "python"
        command = [python, exec_path, "--cuda", "--video", "--file", "config.json"]
        print("Starting local Job #", str(job_number))
        if args.v:
            print(command)
        result = subprocess.check_output(command)

        print("Finished local Job #", str(job_number))

        try:
            result = subprocess.check_output("mkdir results/" + str(job_number), shell=True)
        except subprocess.CalledProcessError:
            pass
        try:
            result = subprocess.check_output("mv results/output.txt results/" + str(job_number), shell=True)
        except subprocess.CalledProcessError:
            pass
        try:
            result = subprocess.check_output("mv results/model results/" + str(job_number), shell=True)
        except subprocess.CalledProcessError:
            pass
        try:
            result = subprocess.check_output("mv results/video results/" + str(job_number), shell=True)
        except subprocess.CalledProcessError:
            pass

        job_number += 1
    f = open('job_number.txt', 'w')
    f.write(str(job_number))
    f.close()
