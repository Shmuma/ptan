import subprocess
import json
import argparse

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
    parser.add_argument("--file", default='', help="Input file")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            jobs = json.loads(open(args.file, "r").read())


    with open('job_number.txt', 'r') as f:
        job_number = int(f.read())

    for job in jobs:
        # start the first job
        config = json.dumps(job)
        config = ''.join(config.split())  # remove spaces from config string
        command = "echo '" + config + "' > config.json && python samples/dqn_speedup/05_new_wrappers.py " \
                                      "--cuda --video --file config.json --stop " + str(frame_stop)
        print(command)
        result = subprocess.check_output(command, shell=True)

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
