import os
import matplotlib.pyplot as plt
import csv

def visualize_data(job_list):
    print("Plotting results for jobs {}".format(job_list))
    curdir = os.path.abspath(__file__)
    outfile = "/output.txt"

    plt.figure(1)
    results = {}
    fieldnames = ['frames', 'games', 'mean reward', 'mean score', 'max score']
    for job_id in job_list:
        results[job_id] = {}
        for field in fieldnames:
            results[job_id][field] = []
        results_path = os.path.abspath(os.path.join(curdir, '../results/' + str(job_id)))
        with open(results_path + outfile) as f:
            reader = csv.DictReader(f)
            for line in reader:
                for field in fieldnames:
                        results[job_id][field].append(float(line[field]))

        i = 0
        for key in results[job_id].keys():
            if key != 'games' and key != 'frames':
                num_keys = str(len(results[job_id].keys()) - 2)
                subplot_num = int(num_keys + '1' + str(i + 1))
                print(subplot_num)
                plt.subplot(subplot_num)
                plt.plot(results[job_id]['games'], results[job_id][key])
                i += 1

    plt.show()



if __name__ == "__main__":
    job_list = [2]
    visualize_data(job_list)