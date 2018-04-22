import os, json, shutil
import pandas as pd
import numpy as np


class FileLogger(object):
    def __init__(self,
                 path,
                 file_name,
                 model_name,
                 script):

        if script is not True:
            self.path = path + file_name + '/'
            self.load_model = None

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            else:
                while True:
                    overwrite = input('This file already exists. Do you want to overwrite it?'
                                      ' Y(yes) N(no) C(continue writing) E(exit)')
                    if overwrite == 'Y' or overwrite == 'N' or overwrite == 'C':
                        break
                    elif overwrite == 'E':
                        raise SystemExit
                    else:
                        print('Please choose one valid option. '
                              'Y for yes N for no or C for continue writing')

                if overwrite == 'N':
                    name = input('Choose a new name for the file:')
                    assert os.path.exists(path + name + '/') is False, \
                        'Yeah that one is already in use. Sorry dude! Please choose another name'
                    file_name = name
                    self.path = path + file_name + '/'
                    os.makedirs(self.path)
                    print('New directory created at {}'.format(self.path))

                if overwrite == 'Y':
                    while True:
                        overwrite = input('Are you completly fine with this. '
                                          'This will remove all previous files and existing models from this directory? Y(yes) N(no)')
                        if overwrite == 'Y' or overwrite == 'N':
                            break
                    if overwrite == 'Y':
                        shutil.rmtree(self.path)
                        os.makedirs(self.path)

                    else:
                        assert overwrite is None, \
                            'Well then you should check your files before.' \
                            ' You always can choose another file name to this model'

                if overwrite == 'C':
                    assert model_name is not None, \
                        'You didnt choose a model to resume train. ' \
                        'Please try again with a valid name or start a new session.'
                    assert os.path.exists(path + file_name + '/model_checkpoint/') is True, \
                        'You dont have models to resume. Please restart and start a new session'
                    assert os.path.exists(path + file_name + '/model_checkpoint/' + model_name) is True, \
                        'That model name dont exist. Please choose other model to resume'
                    self.load_model = path + file_name + '/model_checkpoint/' + model_name
        else:
            self.path = path + '/' + file_name
            self.file_path = self.path
            self.load_model = None

            if not os.path.exists(self.path):
                os.makedirs(self.path)
            else:
                print('removing')
                shutil.rmtree(self.path)
                os.makedirs(self.path)

        self.file_name = file_name

    def write_train(self,
                    log_interval,
                    step,
                    epoch,
                    batch,
                    loss):

        if batch % log_interval == 0:
            self.update_file(step,
                             epoch,
                             batch,
                             loss,
                             'train_log.txt')

    def write_valid(self,
                    log_interval,
                    step,
                    epoch,
                    batch,
                    loss):

        if batch % log_interval == 0:
            self.update_file(step,
                             epoch,
                             batch,
                             loss,
                             'valid_log.txt')

    def write_test(self,
                   log_interval,
                   step,
                   epoch,
                   batch,
                   loss):

        if batch % log_interval == 0:
            self.update_file(step,
                             epoch,
                             batch,
                             loss,
                             'test_log.txt')

    def write_metadata(self,
                       metadata):

        self.metadataLogger = open(self.path + '/metadata.txt', 'w')
        self.metadataLogger.write(json.dumps(metadata))
        self.metadataLogger.close()

    def open_writers(self):

        data = {'Step': [],
                'Epoch_Number': [],
                'Batch_number': [],
                'Loss': []
                }

        self.trainLogger = open(self.path + '/train_log.txt', 'w')
        self.trainLogger.write(json.dumps(data))
        self.trainLogger.close()
        self.validLogger = open(self.path + '/valid_log.txt', 'w')
        self.validLogger.write(json.dumps(data))
        self.validLogger.close()
        self.testLogger = open(self.path + '/test_log.txt', 'w')
        self.testLogger.write(json.dumps(data))
        self.testLogger.close()

    def update_file(self,
                    step,
                    epoch,
                    batch,
                    loss,
                    file_name):

        data_temp = {
            'Step': [int(step)],
            'Epoch_Number': [int(epoch)],
            'Batch_number': [int(batch)],
            'Loss': [float(loss)]
        }

        with open(self.path + '/' + file_name, 'r') as file:
            data = (json.load(file))

        for key, value in zip(data.items(), data_temp.items()):
            key[1].append(value[1][0])

        with open(self.path + '/' + file_name, 'w') as file:
            json.dump(data, file)

    def read_files(self,
                   file_name):

        with open(self.path + '/' + file_name, 'r') as file:
            data = (json.load(file))

        dataframe = []
        for column in data.keys():
            dataframe.append(pd.DataFrame(data[column], columns=[column]))

        return pd.concat(dataframe, axis=1)

    def start(self,
              name=None):

        if name is not None:

            self.path = self.file_path + '/' + name

            if not os.path.exists(self.path):
                os.makedirs(self.path)
                self.open_writers()
            else:
                print('removing')
                shutil.rmtree(self.path)
                os.makedirs(self.path)
                self.open_writers()
        else:
            self.open_writers()

    def write_results(self,
                      df_test,
                      results,
                      mse,
                      mae,
                      score):

        df_test.to_csv(self.path + '/results.csv')
        results.to_csv(self.path + '/results_target.csv')

        with open(self.path + '/final_results.txt', 'w') as file:
            file.write('Mean Squared Error - {}\n'.format(mse))
            file.write('Mean Absolute Error - {}\n'.format(mae))
            file.write('Score - {}'.format(score))
