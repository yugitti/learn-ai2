import tqdm
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import *

import time
# custom_style = {'axes.labelcolor': 'white',
#                 'xtick.color': 'white',
#                 'ytick.color': 'white'}
sns.set_style("whitegrid")
# sns.set_style("whitegrid", rc=custom_style)
sns.set_context("notebook")
# plt.style.use('dark_background')
# plt.rcParams["font.size"] = 18

class TrainLogger(object):

    def __init__(self, args):
        try:
            os.makedirs(args.out_dir)
        except OSError:
            pass
        self.file = open(os.path.join(args.out_dir, 'log'), 'w')
        self.out_dir = args.out_dir
        self.epoch = args.epochs
        self.iters = args.iters
        self.iter = 0
        self.log_interval = args.log_interval
        self.logs = []
        self.train_accuracy = [0]
        self.valid_accuracy = [0]
        self.train_loss = [0]
        self.valid_loss = [0]
        self.train_log_summary = []
        self.max_accuracy = 0
        self.prefix = '_' + args.prefix

    def write(self, log):
        ## write log
        print('{}'.format(log))
        self.file.write((log + self.prefix + "\n"))

    def state_dict(self):
        ## returns the state of the loggers
        return {'logs': self.logs}

    def load_state_dict(self, state_dict):
        ## load the logger state
        self.logs = state_dict['logs']
        # write logs
        tqdm.write(self.logs[-1])
        for log in self.logs:
            tqdm.write(log, file=self.file)

    def train_log(self, loss, correct, total, iter_id):
        self.iter = iter_id
        accuracy =  100. * correct / total
        self.train_accuracy.append(accuracy)
        self.train_loss.append(loss)
        print('')
        log = 'ITER [{}/{}]: Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            self.iter, self.iters, loss, correct, total, 100. * correct / total)
        self.logs.append(log)
        self.write(log)

    def valid_log(self, loss, correct, total, iter_id):
        self.iter = iter_id
        accuracy = 100. * correct / total
        self.valid_accuracy.append(accuracy)
        self.valid_loss.append(loss)

        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy

        log = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, total, 100. * correct / total)
        self.logs.append(log)
        self.write(log)
        self.plot()

    def save_loss(self):
        np.save(os.path.join(self.out_dir ,'train_accuracy' + self.prefix), self.train_accuracy)
        np.save(os.path.join(self.out_dir ,'valid_accuracy' + self.prefix), self.valid_accuracy)
        np.save(os.path.join(self.out_dir ,'train_loss' + self.prefix), self.train_loss)
        np.save(os.path.join(self.out_dir ,'valid_loss' + self.prefix), self.valid_loss)

    def plot(self):

        self.plot_save(self.train_accuracy, self.valid_accuracy, 'accuracy')
        self.plot_save(self.train_loss, self.valid_loss, 'loss')


    def plot_save(self, y_train, y_valid, plot_type):

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        # X = np.arange(1,len(self.valid_accuracy)+1)
        X = np.arange(0, self.iter + 1, self.log_interval)

        # plot
        ax.plot(X, y_train, marker='x', color='red', label='train')
        ax.plot(X, y_valid, marker='x', color='blue', label='validation')

        # x-axis
        plt.xlim([0, self.iters])

        if self.iters < 10:
            ax.set_xticks(np.arange(0, self.iters + 1))
        else:
            ax.set_xticks(np.arange(0, self.iters + 1, self.log_interval))
        ax.set_xlabel("iterator")
        # ax.xaxis.set_major_locator(MultipleLocator(1))

        # y-axis
        if plot_type is 'accuracy':
            plt.ylim(0, 100)
            ax.set_yticks(np.arange(0, 110, 10))
        ax.set_ylabel(plot_type + ' [%]')

        # legend and title
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # ax.legend(loc='best')
        ax.set_title(plot_type + " curve", fontsize=14)
        plt.subplots_adjust(right=0.8)

        # save
        plt.savefig(os.path.join(self.out_dir, plot_type + self.prefix))

        # close
        plt.close()

    def progress(self, p):
        sys.stdout.write("\rITER [{}/{}]".format(p, self.iters))
        sys.stdout.flush()