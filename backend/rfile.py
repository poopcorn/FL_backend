import json

from backend.file import File
from backend.settings import ROUND_EVERY_FILE


class RFile:

    def __init__(self, path):
        self.JSON_PATH = path

    # change layers paramaters from str to list
    def get_layer(self, str):
        if str == -1:
            layers = ['dense']
        else:
            layers = str[1: len(str) - 1]
            layers = layers.split(',')
            for i in range(len(layers)):
                layers[i] = layers[i][1: len(layers[i]) - 1]
        return layers

    # determine gradients from which layers should be used
    def get_grad(self, path, layers, round):

        filename = self.get_filename(round, 'client_grad', 'gradients_')

        file = open(path + 'client_grad/' + filename, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()

        if round == -1:
            round = list(data.keys())[-1]
        data = data[str(round)]

        # get gradients of all clients
        vec = {}
        for key in data:
            vec[key] = []
            for item in layers:
                vec[key] += data[key][item]
        return {'round': int(round), 'data': vec}

    # get the performance of a certain round
    def get_perf(self, path, round, stage):

        file = open(path + 'performance.json', 'r', encoding='utf-8')
        data = json.load(file)
        file.close()

        if round == -1:
            round = list(data.keys())[-1]

        data = data[str(round)][stage]

        return {'round': int(round), 'data': data}

    # get the contribution of a certain round
    def get_con(self, path, round):

        file = open(path + 'contribution.json', 'r', encoding='utf-8')
        data = json.load(file)
        file.close()

        if round == -1:
            round = list(data.keys())[-1]
        data = data[str(round)]

        return {'round': int(round), 'data': data}

    def get_avg_grad(self, path, layers, round):

        filename = self.get_filename(round, 'avg_grad', 'avg_grad_')

        file = open(path + 'avg_grad/' + filename, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()

        if round == -1:
            round = list(data.keys())[-1]
        data = data[str(round)]

        vec = []
        for item in layers:
            vec += data[item]
        return {'round': int(round), 'data': vec}

    def get_filename(self, round, dir_name, file_prefix):
        if round == -1:
            file_obj = File(self.JSON_PATH + dir_name, file_prefix)
            filename = file_obj.latest_file(ROUND_EVERY_FILE)
        else:
            filename = file_prefix + \
                       str((round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE) + '_' + \
                       str((round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE + ROUND_EVERY_FILE - 1) + '.json'
            print(filename, (round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE + ROUND_EVERY_FILE - 1)
        return filename