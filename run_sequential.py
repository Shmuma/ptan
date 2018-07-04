import subprocess
import json
import time
import argparse

"""
Before running, run 'ngc config set' and set the following:
Debug Mode: False
CLI output format type: json
"""
job_names = "testjob"
frame_stop = 5000

jobs = [
    {
      "epsilon_frames": 1000000,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.00005,
      "gamma": 0.99
    },
    {
      "epsilon_frames": 2000000,
      "epsilon_start": 2.0,
      "epsilon_final": 0.2,
      "learning_rate": 0.00006,
      "gamma": 0.98,
      "dqn_model": "FSADQNAppendToFC"
    },
    {
        "epsilon_frames": 3000000,
        "epsilon_start": 3.0,
        "epsilon_final": 0.3,
        "learning_rate": 0.00004,
        "gamma": 0.99
    }

]  # list of dictionaries (json)


class JobControl:
    def __init__(self, job_list, v):
        self.jobcounter = 0
        self.jobs = job_list
        self.verbose = v

    def get_job(self, name, command):
        name = '"'+name+'"'
        command = '"' + command + '"'
        return ['ngc batch run', '--name', name, '--image', '"lucasl_drl_00/fsa-atari:0.1"', '--ace', 'nv-us-west-2',
                '--instance', 'ngcv1', '--commandline', command,  '--result', '/results']

    def run_next_job(self):
        if self.jobcounter >= len(self.jobs):
            return None
        config = json.dumps(self.jobs[self.jobcounter])
        config = ''.join(config.split())  # remove spaces from config string
        config = '\\"'.join(config.split('"'))  # escape quotes
        command = "echo '" + config + "' > config.json && opt/conda/envs/pytorch-py3.6/bin/python " \
                                      "/workspace/ptan/samples/dqn_speedup/05_new_wrappers.py " \
                                      "--cuda --fsa --telemetry --video --file config.json --stop " + str(frame_stop)
        runline = self.get_job(job_names + str(self.jobcounter), command)
        if self.verbose:
            print(' '.join(runline))
        result = subprocess.check_output(' '.join(runline), shell=True)
        if b"Job created." in result:
            result = result[13:]
        self.jobcounter += 1
        if self.verbose:
            print(result)
        data = json.loads(result)
        job_id = data["id"]
        print("Job Id is ", job_id)
        return job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    control = JobControl(jobs, args.v)

    # start the first job
    job_id = control.run_next_job()

    while True:
        # check if running job has finished
        try:
            result = subprocess.check_output(['ngc', 'batch', 'get', str(job_id)])
        except subprocess.CalledProcessError:
            print("Got an error, retrying in 10")
            time.sleep(10)
            continue
        if args.v:
            print(result)
        json_data = json.loads(result)
        status = json_data["jobStatus"]["status"]
        print("Job Status: ", status)

        if status == "FINISHED_SUCCESS" or status == "FAILED":
            job_id = control.run_next_job()
            if job_id == None:
                exit()

        time.sleep(60)  # wait a minute before checking again
