import os

def visualize_data(job_list):
    print("Plotting results for jobs {}".format(job_list))
    curdir = os.path.abspath(__file__)
    outfile = "/output.txt"

    for job_id in job_list:
        results_path = os.path.abspath(os.path.join(curdir, '../results/' + str(job_id)))
        with open(results_path + outfile) as f:
            for line in f:
                print(line)



if __name__ == "__main__":
    job_list = [78764, 78763]
    visualize_data(job_list)