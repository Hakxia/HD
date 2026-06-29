import os
import sys
import numpy as np
from pathlib import Path


def write(log, str):
    sys.stdout.flush()
    log.write(str + '\n')
    log.flush()


class Report():
    def __init__(self, save_dir, type):
        filename = os.path.join(save_dir, f'{type}_log.txt')

        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(filename):
            self.logFile = open(filename, 'a')
        else:
            self.logFile = open(filename, 'w')

    def write(self, str):
        print(str)
        write(self.logFile, str)

    def __del__(self):
        if hasattr(self, 'logFile') and not self.logFile.closed:
            self.logFile.close()


class TrainReport():
    def __init__(self):
        self.total_loss = []
        self.completion_loss = []
        self.energy_loss = []
        self.positive_energy = []
        self.negative_energy = []
        self.energy_gap = []
        self.candidate_accuracy = []
        self.num_examples = 0

    def update(
        self,
        batch_size,
        total_loss,
        completion_loss,
        energy_loss,
        positive_energy,
        negative_energy,
        energy_gap,
        candidate_accuracy,
    ):
        self.num_examples += batch_size
        self.total_loss.append(total_loss * batch_size)
        self.completion_loss.append(completion_loss * batch_size)
        self.energy_loss.append(energy_loss * batch_size)
        self.positive_energy.append(positive_energy * batch_size)
        self.negative_energy.append(negative_energy * batch_size)
        self.energy_gap.append(energy_gap * batch_size)
        self.candidate_accuracy.append(candidate_accuracy * batch_size)

    def compute_mean(self):
        if self.num_examples <= 0:
            return
        self.total_loss = np.sum(self.total_loss) / self.num_examples
        self.completion_loss = np.sum(self.completion_loss) / self.num_examples
        self.energy_loss = np.sum(self.energy_loss) / self.num_examples
        self.positive_energy = np.sum(self.positive_energy) / self.num_examples
        self.negative_energy = np.sum(self.negative_energy) / self.num_examples
        self.energy_gap = np.sum(self.energy_gap) / self.num_examples
        self.candidate_accuracy = np.sum(self.candidate_accuracy) / self.num_examples

    def result_str(self, lr, period_time, mask_ratio):
        self.compute_mean()
        result = f'Total Loss: {self.total_loss:.6f}\tLearning rate: {lr:.7f}\tTime: {period_time:.4f}\t'
        result += f'Completion Loss: {self.completion_loss:.6f}\tEnergy Loss: {self.energy_loss:.6f}\t'
        result += f'Positive Energy: {self.positive_energy:.6f}\tNegative Energy: {self.negative_energy:.6f}\t'
        result += f'Energy Gap: {self.energy_gap:.6f}\tCandidate Acc: {self.candidate_accuracy:.6f}\t'
        result += f'Mask Ratio: {mask_ratio:.3f}'
        return result
