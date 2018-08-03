import os
import matplotlib.pyplot as plt
import csv
import yaml
import pickle

def visualize_data(job_list):
    print("Plotting results for jobs {}".format(job_list))
    curdir = os.path.abspath(__file__)
    outfile = "/output.txt"
    paramfile = "/params.txt"
    modelfile = "/model/data.pkl"

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

        model_path = results_path + modelfile
        model_info = pickle.load(open(model_path, "rb"))
        model_name = model_info[2]
        tm_acc = model_info[1]
        ave_score = model_info[0]
        print("{} | {} | TM acc: {} | Ave score: {}".format(job_id, model_name, tm_acc, ave_score))

        params_path = results_path + paramfile
        param_dict = yaml.load(open(params_path))

        i = 0
        for key in results[job_id].keys():
            if key != 'games' and key != 'frames':
                num_keys = str(len(results[job_id].keys()) - 2)
                subplot_num = int(num_keys + '1' + str(i + 1))
                # print(subplot_num)
                plt.subplot(subplot_num)
                plt.ylabel(key)
                plt.xlabel('games')
                plt.plot(results[job_id]['games'], results[job_id][key], label="{} | {}: e frames: {:9} | lr: {:10.7}".format(
                    job_id, model_name, int(param_dict['epsilon_frames']), format(float(param_dict['learning_rate']), 'f')))
                i += 1
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # DQN: 81001, 81002, 81003
    # DQN: 81342, 81343, 81344
    # Best DQN: 81001, lr: 0.001

    # Affine: 81389, 81390, 81391
    # Bias: 80941, 82000, 81999, 81998 | BEST: 81999
    # Scaling: 80940, 80939, 80938

    # 10 million frame runs: 81664, 81665, 81666, 82109

    # Lasers only: 90820 [messed up], 91649
    job_list = [90820, 91649, 91702]
    # job_list = [81664, 81665, 81666, 82109]
    # job_list = [81389, 81390, 81391, 80456]
    # job_list = [81389, 81390, 81391, 81001]
    # job_list = [80380, 81001, 81002]
    # job_list = [80380, 80381, 80453, 80454]
    # job_list = [80379, 80380, 80381, 80382]
    visualize_data(job_list)