import os, json
import matplotlib.pyplot as plt

def plot_LI(path,
            level=1,
            show_useful = False):

    data = parse_LI(path)
    mind = min([min(data[key]["dimension"]) for key in data.keys()])
    maxd = max([max(data[key]["dimension"]) for key in data.keys()])

    fig = plt.figure(figsize=(17, 8))
    for MLO in data.keys():

        if show_useful == True:
            if all(x == data[MLO]["LI num"][0] for x in data[MLO]["LI num"]):
                continue

        if MLO[-1]==str(level):
            plt.plot(data[MLO]["dimension"], data[MLO]["LI num"], label=MLO[:3], linewidth = 6, zorder=1)
            plt.scatter(data[MLO]["dimension"], data[MLO]["LI num"], s=200, zorder=2)

    plt.xticks(range(mind, maxd+1, 1))
    plt.legend(fontsize=29, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Dimension', fontsize=29)
    plt.ylabel('Number of LI moment matrices', fontsize=29)
    plt.tick_params(axis='both', labelsize=26)
    plt.show()

def parse_LI(path):

    data = {}
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]

    for index, js in enumerate(json_files):
        with open(os.path.join(path, js)) as json_file:
            json_temp = json.load(json_file)

            MLO = read_MLO(json_temp)
            if MLO not in data:
                data[MLO] = {}
                data[MLO]["LI num"] = []
                data[MLO]["dimension"] = []

            data[MLO]["LI num"].append(json_temp["number of LI moment matrices"])
            data[MLO]["dimension"].append(json_temp["dimension"])

    for MLO in data.keys():

        X = [x for y,x in sorted(zip(data[MLO]["LI num"], data[MLO]["dimension"]))]
        Y = [y for y,x in sorted(zip(data[MLO]["LI num"], data[MLO]["dimension"]))]

        data[MLO]["dimension"] = X
        data[MLO]["LI num"] = Y

    return data

def read_MLO(js):
    try:
        MLO = str(js["num of observables"]) + str(js["maximum length of sequences"]) + str(js["num of outcomes"]) + "-" + str(js["level"])
    except:
        MLO = str(js["num of observables"]) + str(js["maximum length of sequences"])  + str(js["num of outcomes"]) + "-1"
    return MLO
