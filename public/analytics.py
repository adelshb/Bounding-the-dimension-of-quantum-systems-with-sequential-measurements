import os, json
import matplotlib.pyplot as plt

def plot_ranks(path,
            show_useful = False):

    data = parse_ranks(path)
    mind = min([min(data[key]["dimension"]) for key in data.keys()])
    maxd = max([max(data[key]["dimension"]) for key in data.keys()])

    fig = plt.figure(figsize=(17, 8))
    for MSO in data.keys():

        if show_useful == True:
            if all(x == data[MSO]["ranks"][0] for x in data[MSO]["ranks"]):
                continue

        plt.plot(data[MSO]["dimension"], data[MSO]["ranks"], label=MSO, linewidth = 3, zorder=1)
        plt.scatter(data[MSO]["dimension"], data[MSO]["ranks"], zorder=2)

    plt.xticks(range(mind, maxd+1, 1))
    plt.legend(title='MSO:', fontsize=20)
    plt.xlabel('Dimension', fontsize=20)
    plt.ylabel('Rank', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.show()

def parse_ranks(path):

    data = {}
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]

    for index, js in enumerate(json_files):
        with open(os.path.join(path, js)) as json_file:
            json_temp = json.load(json_file)

            MSO = read_MSO(json_temp)
            if MSO not in data:
                data[MSO] = {}
                data[MSO]["ranks"] = []
                data[MSO]["dimension"] = []

            data[MSO]["ranks"].append(json_temp["rank"])
            data[MSO]["dimension"].append(json_temp["dimension"])

    for MSO in data.keys():

        X = [x for y,x in sorted(zip(data[MSO]["ranks"], data[MSO]["dimension"]))]
        Y = [y for y,x in sorted(zip(data[MSO]["ranks"], data[MSO]["dimension"]))]

        data[MSO]["dimension"] = X
        data[MSO]["ranks"] = Y

    return data

def read_MSO(js):
    MSO = str(js["num of observables"]) +str(js["maximum length of sequences"]) +str(js["num of outcomes"])
    return MSO
