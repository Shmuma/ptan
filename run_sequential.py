import subprocess
import json
import time
import argparse

import download_results

"""
Before running, run 'ngc config set' and set the following:
Debug Mode: False
CLI output format type: json
"""

job_name_prefix = "test"

jobs = [
    {
      "epsilon_frames": 10 ** 6,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv1",
      "replay_initial": 500,
      "video_interval": 1000,
      "frame_stop": 3000
    },
    {
      "epsilon_frames": 10 ** 6 / 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv2",
      "replay_initial": 500,
      "video_interval": 1000,
      "frame_stop": 3000
    },
    {
      "epsilon_frames": 10 ** 6 * 2,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv2",
      "replay_initial": 500,
      "video_interval": 1000,
      "frame_stop": 3000
    }

]  # list of dictionaries (json)


class JobControl:
    def __init__(self, job_name, job_list, v):
        self.jobcounter = 0
        self.jobs = job_list
        self.verbose = v
        self.job_name = job_name+'-'
        self.job_list = []

    def get_job(self, name, command, machine="ngcv8"):
        name = '"'+name+'"'
        command = '"' + command + '"'
        return ['ngc batch run', '--name', name, '--image', '"mitnvda18/fsa-atari:latest"', '--ace', 'nv-us-west-2',
                '--instance', machine , '--commandline', command,  '--result', '/results']

    def run_next_job(self):
        if self.jobcounter >= len(self.jobs):  # end of job list
            return None
        print("Running " + self.job_name + str(self.jobcounter))

        config = json.dumps(self.jobs[self.jobcounter])
        config = ''.join(config.split())  # remove spaces from config string
        config = '\\"'.join(config.split('"'))  # escape quotes
        command = "echo '" + config + "' > config.json && " \
                                      "/opt/conda/envs/pytorch-py3.6/bin/python " \
                                      "samples/dqn_speedup/05_new_wrappers.py " \
                                      "--cuda --telemetry --video --file config.json"

        if "machine" in self.jobs[self.jobcounter]:
            runline = self.get_job(self.job_name + str(self.jobcounter), command,
                                   self.jobs[self.jobcounter]["machine"])
        else:
            runline = self.get_job(self.job_name + str(self.jobcounter), command)

        if self.verbose:
            print(' '.join(runline))
        result = subprocess.check_output(' '.join(runline), shell=True)
        if self.verbose:
            print(result)

        if b"Job created." in result:
            data = json.loads(result[13:])
        else:
            data = json.loads(result)
            if len(data) == 1:
                data = data[0]

        self.job_list.append(data["id"])
        print(self.job_name + str(self.jobcounter), "Job Id is ", data["id"])
        self.jobcounter += 1
        return data["id"]

    def run_all_jobs(self):
        while self.jobcounter < len(self.jobs):
            self.run_next_job()
        self.job_list.reverse()

    def get_last_job(self):
        if len(self.job_list) == 0:
            return None
        return self.job_list[-1]

    def pop_job(self):
        if len(self.job_list) == 0:
            return None
        return self.job_list.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")
    parser.add_argument("-p", default=False, action="store_true", help="Enable parallel")
    parser.add_argument("--file", default='', help="Input file")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            jobs = json.loads(open(args.file, "r").read())

    if args.p:
        job_lists = {}
        for job in jobs:
            if "machine" in job:
                if job["machine"] not in job_lists:
                    job_lists[job["machine"]] = []
                job_lists[job["machine"]].append(job)
            else:
                if "none" not in job_lists:
                    job_lists["none"] = []
                job_lists["none"].append(job)
        controls = []
        for machine in job_lists:
            controls.append(JobControl(job_name_prefix+'-'+machine, job_lists[machine], args.v))
    else:
        controls = [JobControl(job_name_prefix, jobs, args.v)]

    # start the first jobs
    for c in controls:
        c.run_all_jobs()

    successful_jobs = []

    while len(controls) > 0:
        # check if running job has finished
        for c in controls:
            job_id = c.get_last_job()
            try:
                result = subprocess.check_output(['ngc', 'batch', 'get', str(job_id)])
            except subprocess.CalledProcessError:
                print("Got an error, retrying in 30")
            if args.v:
                print(result)
            json_data = json.loads(result)

            if len(json_data) == 1:  # hacky fix for different return formats?
                json_data = json_data[0]
            status = json_data["jobStatus"]["status"]
            print("Job #" + str(job_id) + " Status: ", status)

            if status == "FINISHED_SUCCESS" or status == "FAILED" or status == "KILLED_BY_USER":
                if status == "FINISHED_SUCCESS":
                    successful_jobs.append(job_id)
                c.pop_job()

        controls = [c for c in controls if c.get_last_job() is not None]
        time.sleep(30)  # wait a minute before checking again

    check_ip = subprocess.check_output(["ifconfig"])
    if b"128.30.25" in check_ip:
        download_results.get_jobs(successful_jobs)