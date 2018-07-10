import os
import matplotlib.pyplot as plt
import csv
import yaml

def visualize_data(job_list):
    print("Plotting results for jobs {}".format(job_list))
    curdir = os.path.abspath(__file__)
    outfile = "/output.txt"
    paramfile = "/params.txt"

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

        params_path = results_path + paramfile
        param_dict = yaml.load(open(params_path))

        i = 0
        for key in results[job_id].keys():
            if key != 'games' and key != 'frames':
                num_keys = str(len(results[job_id].keys()) - 2)
                subplot_num = int(num_keys + '1' + str(i + 1))
                print(subplot_num)
                plt.subplot(subplot_num)
                plt.ylabel(key)
                plt.xlabel('games')
                plt.plot(results[job_id]['games'], results[job_id][key], label="e frames: {:9} | lr: {:10.7}".format(
                    int(param_dict['epsilon_frames']), format(float(param_dict['learning_rate']), 'f')))
                i += 1
    plt.legend()
    plt.show()



if __name__ == "__main__":
    job_list = [9, 10, 79274, 79275, 79319, 79320, 79939, 79940]
    visualize_data(job_list)