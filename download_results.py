import subprocess
import os

def get_jobs(job_list):
    print("Trying to download results from jobs {}".format(job_list))
    curdir = os.path.abspath(__file__)
    results_path = os.path.abspath(os.path.join(curdir, '../results'))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for job_id in job_list:
        output = subprocess.check_output(['ngc', 'result', 'download', '-d', results_path, str(job_id)])
        if b'Download completed' in output or b'Completed' in output:
            print("Successfully downloaded data for job {}".format(job_id))
        else:
            print("DATA NOT SUCCESSFULLY DOWNLOADED FOR JOB {}".format(job_id))


if __name__ == "__main__":
    job_list = [82109]
    get_jobs(job_list)