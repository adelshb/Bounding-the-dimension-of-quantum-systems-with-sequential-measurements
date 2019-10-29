import os, json
import matplotlib.pyplot as plt

def plot_ranks(data):

    for MSO in data.keys():
        plt.plot(data[MSO]["dimension"], data[MSO]["ranks"], label=MSO)

    plt.legend(title='MSO:')
    plt.show()

def parse_ranks():

    path = "/data_basis/"

    data = {}
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]

    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json = json.load(json_file)

            MSO = read_scenario(json)
            if MSO no int data:
                data[MSO] = {}
                data[MSO]["ranks"] = []
                data[MSO]["dimension"] = []

            data[MSO]["ranks"].append(json["rank"])
            data[MSO]["dimension"].append(json["dimension"])

    return data

def MSO(json):
    MSO = str(json["maximum length of sequences"]) +str(json["num of observables"]) +str(json["num of outcomes"])
    return MSO
