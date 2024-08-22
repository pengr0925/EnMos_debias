import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from collections import Counter

import json

import pandas as pd

def plot_3data_2sets_2(indices, data1_1, data1_2, data1_3, data2_1, data2_2, data2_3, name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(indices, data1_1, color='r', label='logits_adj')
    axs[0].plot(indices, data2_1, color='b', label='CE_loss')
    axs[0].set_title('E_y-E+BE')
    axs[0].legend()

    axs[1].plot(indices, data1_2, color='r', label='logits_adj')
    axs[1].plot(indices, data2_2, color='b', label='CE_loss')
    axs[1].set_title('E_y-E-BE')
    axs[1].legend()

    axs[2].plot(indices, data1_3, color='r', label='logits_adj')
    axs[2].plot(indices, data2_3, color='b', label='CE_loss')
    axs[2].set_title('E_y-E=loss')
    axs[2].legend()

    plt.tight_layout()

    plt.subplots_adjust(top=0.85)

    fig.suptitle(name)
    plt.savefig('./compare_2/{}.png'.format(name))

    plt.show()

def plot_3data_2sets_3(indices, data1_1, data1_2, data1_3, data2_1, data2_2, data2_3, name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(indices, data1_1, color='r', label='logits_adj')
    axs[0].plot(indices, data2_1, color='b', label='CE_loss')
    axs[0].set_title('boundary_Energy')
    axs[0].legend()

    axs[1].plot(indices, data1_2, color='r', label='logits_adj')
    axs[1].plot(indices, data2_2, color='b', label='CE_loss')
    axs[1].set_title('Energy')
    axs[1].legend()

    axs[2].plot(indices, data1_3, color='r', label='logits_adj')
    axs[2].plot(indices, data2_3, color='b', label='CE_loss')
    axs[2].set_title('Energy_y')
    axs[2].legend()

    plt.tight_layout()

    plt.subplots_adjust(top=0.85)

    fig.suptitle(name)
    plt.savefig('./compare/{}.png'.format(name))

    plt.show()

def plot_2data_1(indices, data1, data2, name, name_1, name_2):
    fig, ax1 = plt.subplots()
    ax1.plot(indices, data1, color='r', label=name_1)
    ax1.set_xlabel('Index')
    ax1.set_ylabel(name)
    ax1.tick_params(axis='y', labelcolor='r')


    # 绘制第二个数据集
    ax1.plot(indices, data2, color='b', label=name_2)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')

    plt.title(name)

    # 显示图形
    plt.show()

def plot_2data(indices, data1, data2, name):
    fig, ax1 = plt.subplots()
    ax1.plot(indices, data1, label='loss')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('loss')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()

    # 绘制第二个数据集
    ax2.plot(indices, data2, color='b', label='boundary_energy')
    ax2.set_ylabel('boundary_energy', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(name)

    # 显示图形
    plt.show()

def plot_3data(indices, data1, data2, data3, name):

    fig, ax1 = plt.subplots()
    ax1.plot(indices, data1, color='r', label='Probs', lineStyle='dashed')
    ax1.set_xlabel('Class_index')
    ax1.set_ylabel('probs', color='k')
    ax1.tick_params(axis='y', labelcolor='r')



    ax2 = ax1.twinx()
    # 绘制第二个数据集
    ax2.plot(indices, data2, color='g', label='BE-sub-E_y')
    # 绘制第三个数据集
    ax2.plot(indices, data3, color='y', label='-E')
    ax2.set_ylabel('energy', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')


    # plt.title('Logits Adjustment & {}'.format(name))
    plt.title('CE Loss & {}'.format(name))

    # 显示图形
    plt.show()

def plot_4data(indices, data1, data2, data3, data4, name):

    fig, ax1 = plt.subplots()
    ax1.plot(indices, data1, color='r', label='BE', lineStyle='dashed')
    ax1.set_xlabel('Class_index')
    ax1.set_ylabel('value', color='k')
    ax1.tick_params(axis='y', labelcolor='r')

    # 绘制第二个数据集
    ax1.plot(indices, data2, color='y', label='E')
    # 绘制第三个数据集
    ax1.plot(indices, data3, color='c', label='E_y_sub_E')

    ax1.plot(indices, data4, color='g', label='BE_sub_E')

    # # 绘制第四个数据集
    # ax2 = ax1.twinx()
    # ax2.plot(indices, data4, color='g', label='BE_sub_E')
    # ax2.set_ylabel('fraction', color='k')
    # ax2.tick_params(axis='y', labelcolor='k')

    # # 添加图例
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')


    plt.title('Logits Adjustment & {}'.format(name))
    # plt.title('CE Loss & {}'.format(name))

    # 显示图形
    plt.show()


def plot_5data(indices, data1, data2, data3, data4, data5, name):

    fig, ax1 = plt.subplots()
    ax1.plot(indices, data1, color='r', label='BE', lineStyle='dashed')
    ax1.set_xlabel('Class_index')
    ax1.set_ylabel('value', color='k')
    ax1.tick_params(axis='y', labelcolor='r')

    # 绘制第二个数据集
    ax1.plot(indices, data2, color='y', label='E')
    ax1.plot(indices, data4, color='m', label='BE_sub_E')
    ax1.plot(indices, data5, color='g', label='E_y_sub_BE')


    # # 绘制第四个数据集
    ax2 = ax1.twinx()
    ax2.plot(indices, data3, color='c', label='E_y_sub_E')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # # 添加图例
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # ax1.legend(lines1, labels1, loc='upper right')


    plt.title('Logits Adjustment & {}'.format(name))
    # plt.title('CE Loss & {}'.format(name))

    # 显示图形
    plt.show()


def plot_class_energy_correct_incorrect(logits, boundary_energy, energy, energy_y, targets, class_bias):
    probs = F.softmax(logits, dim=1)
    _, preds = probs.max(dim=1)

    class_prob = probs[range(probs.shape[0]), targets]

    class_counter = Counter(targets.numpy().tolist())
    indices = range(len(class_counter.keys()))

    # boundary_energy_mean_correct = []
    # energy_mean_correct = []
    # energy_y_mean_correct = []
    # boundary_energy_mean_incorrect = []
    # energy_mean_incorrect = []
    # energy_y_mean_incorrect = []
    #
    # for key in range(len(class_counter.keys())):
    #     boundary_energy_mean_correct.append(boundary_energy[(preds == targets) & (targets == key)].sum())
    #     energy_mean_correct.append(energy[(preds == targets) & (targets == key)].sum())
    #     energy_y_mean_correct.append(energy_y[(preds == targets) & (targets == key)].sum())
    #
    #     boundary_energy_mean_incorrect.append(boundary_energy[(preds != targets) & (targets == key)].sum())
    #     energy_mean_incorrect.append(energy[(preds != targets) & (targets == key)].sum())
    #     energy_y_mean_incorrect.append(energy_y[(preds != targets) & (targets == key)].sum())

    # plot_4data(indices, energy_mean_correct, boundary_energy_mean_correct, energy_y_mean_correct, class_bias, 'Correct—sum')
    # plot_4data(indices, energy_mean_incorrect, boundary_energy_mean_incorrect, energy_y_mean_incorrect, class_bias, 'InCorrect-sum')

    E_sub_BE_correct = []
    BE_correct = []
    E_sub_E_y_correct = []
    E_sub_BE_incorrect = []
    BE_incorrect = []
    E_sub_E_y_incorrect = []
    class_correct = []
    class_incorrect = []

    for key in range(len(class_counter.keys())):
        class_correct.append([(preds == targets) & (targets == key)][0].sum())
        BE_correct.append(boundary_energy[(preds == targets) & (targets == key)].mean())
        E_sub_E_y_correct.append((energy-energy_y)[(preds == targets) & (targets == key)].mean())

        class_incorrect.append([(preds != targets) & (targets == key)][0].sum())
        BE_incorrect.append(boundary_energy[(preds != targets) & (targets == key)].mean())
        E_sub_E_y_incorrect.append((energy-energy_y)[(preds != targets) & (targets == key)].mean())

    # plot_3data(indices, E_sub_BE_correct, E_sub_E_y_correct, class_correct,'Correct—sum')
    # plot_3data(indices, E_sub_BE_incorrect, E_sub_E_y_incorrect, class_incorrect, 'InCorrect—sum')

    plot_3data(indices, BE_correct, E_sub_E_y_correct, class_correct,'Correct—mean')
    plot_3data(indices, BE_incorrect, E_sub_E_y_incorrect, class_incorrect, 'InCorrect—mean')


def class_energys_loss_correct_incorrect(logits, boundary_energy, energy, energy_y, targets, class_bias):
    probs = F.softmax(logits, dim=1)
    _, preds = probs.max(dim=1)

    class_prob = probs[range(probs.shape[0]), targets]

    class_counter = Counter(targets.numpy().tolist())
    indices = range(len(class_counter.keys()))

    # BE_correct, E_correct, E_y_sub_E_correct, BE_sub_E_correct = [], [], [], []
    # BE_incorrect, E_incorrect, E_y_sub_E_incorrect, BE_sub_E_incorrect = [], [], [], []

    mask_correct = preds == targets
    mask_incorrect = preds != targets


    # correct_probs_sort, correct_probs_sort_index = class_prob[mask_correct].sort(descending=False)
    # BE_correct = boundary_energy[mask_correct][correct_probs_sort_index]
    # E_correct = energy[mask_correct][correct_probs_sort_index]
    # E_y_sub_E_correct = (energy_y - energy)[mask_correct][correct_probs_sort_index]
    # BE_sub_E_correct = (boundary_energy - energy)[mask_correct][correct_probs_sort_index]
    #
    # incorrect_probs_sort, incorrect_probs_sort_index = class_prob[mask_incorrect].sort(descending=False)
    # BE_incorrect = boundary_energy[mask_incorrect][incorrect_probs_sort_index]
    # E_incorrect = energy[mask_incorrect][incorrect_probs_sort_index]
    # E_y_sub_E_incorrect = (energy_y - energy)[mask_incorrect][incorrect_probs_sort_index]
    # BE_sub_E_incorrect = (boundary_energy - energy)[mask_incorrect][incorrect_probs_sort_index]
    #
    # plot_4data(correct_probs_sort, BE_correct, E_correct, E_y_sub_E_correct, BE_sub_E_correct, 'Correct')
    # plot_4data(incorrect_probs_sort, BE_incorrect, E_incorrect, E_y_sub_E_incorrect, BE_sub_E_incorrect, 'InCorrect')

    for key in range(len(class_counter.keys())):
        correct_probs_sort, correct_probs_sort_index = class_prob[mask_correct & (targets == key)].sort(descending=False)
        BE_correct = boundary_energy[mask_correct & (targets == key)][correct_probs_sort_index]
        E_correct = energy[mask_correct & (targets == key)][correct_probs_sort_index]
        E_y_sub_E_correct = (energy_y - energy)[mask_correct & (targets == key)][correct_probs_sort_index]
        BE_sub_E_correct = (boundary_energy - energy)[mask_correct & (targets == key)][correct_probs_sort_index]
        E_y_sub_BE_correct = (energy_y - boundary_energy)[mask_correct & (targets == key)][correct_probs_sort_index]

        incorrect_probs_sort, incorrect_probs_sort_index = class_prob[mask_incorrect & (targets == key)].sort(descending=False)
        BE_incorrect = boundary_energy[mask_incorrect & (targets == key)][incorrect_probs_sort_index]
        E_incorrect = energy[mask_incorrect & (targets == key)][incorrect_probs_sort_index]
        E_y_sub_E_incorrect = (energy_y - energy)[mask_incorrect & (targets == key)][incorrect_probs_sort_index]
        BE_sub_E_incorrect = (boundary_energy - energy)[mask_incorrect & (targets == key)][incorrect_probs_sort_index]
        E_y_sub_BE_incorrect = (energy_y - boundary_energy)[mask_incorrect & (targets == key)][incorrect_probs_sort_index]

        plot_5data(correct_probs_sort, BE_correct, E_correct, E_y_sub_E_correct, BE_sub_E_correct, E_y_sub_BE_correct,
                   'Correct_{}'.format(key))
        plot_5data(incorrect_probs_sort, BE_incorrect, E_incorrect, E_y_sub_E_incorrect, BE_sub_E_incorrect, E_y_sub_BE_incorrect,
                   'InCorrect_{}'.format(key))

def plot_prob_energy(logits, sub, energy, targets):
    probs = F.softmax(logits)
    y_probs = probs[torch.arange(logits.size(0)), targets]

    y_probs_sort, y_probs_idx = y_probs.sort(descending=True)

    plot_3data(range(logits.shape[0]), y_probs_sort, sub[y_probs_idx], energy[y_probs_idx], 'CE_loss')


def plot_4data_2(indices_1, data1_1, data1_2, data1_3, data1_4,
                 indices_2, data2_1, data2_2, data2_3, data2_4, name):
    # head_probs.size(0), head_probs, boundary_energy_head, energy_head, energy_y_head,

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(indices_1, data1_2, color='b', label='BE')
    axs[0].plot(indices_1, data1_3, label='E(x)')
    axs[0].plot(indices_1, data1_4, label='E(x,y)')
    axs[0].set_ylabel('energy')

    ax0_2 = axs[0].twinx()
    ax0_2.plot(indices_1, data1_1, color='r', label='probs')
    ax0_2.set_ylabel('probs')

    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_2.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc='lower center')

    axs[0].set_title('head')
    # axs[0].legend()



    axs[1].plot(indices_2, data2_2, color='b', label='BE')
    axs[1].plot(indices_2, data2_3, label='E(x)')
    axs[1].plot(indices_2, data2_4, label='E(x,y)')

    ax1_2 = axs[1].twinx()
    ax1_2.plot(indices_2, data2_1, color='r', label='probs')

    axs[1].set_title('tail')

    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc='lower center')

    plt.tight_layout()

    plt.subplots_adjust(top=0.85)

    fig.suptitle(name)
    plt.savefig('./compare_3/{}.png'.format(name))

    plt.show()




def find_boundary_bad_samples(logits, boundary_energy, energy, energy_y, targets):
    probs = F.softmax(logits, dim=1)
    _, preds = probs.max(dim=1)

    head_mask = (targets == 0)
    tail_mask = (targets == 9)

    head_probs = torch.masked_select(probs[torch.arange(logits.size(0)), targets], head_mask)
    tail_probs = torch.masked_select(probs[torch.arange(logits.size(0)), targets], tail_mask)

    head_descend_ord = head_probs.sort(descending=True)[1]
    tail_descend_ord = tail_probs.sort(descending=True)[1]

    boundary_energy_head = boundary_energy[head_mask][head_descend_ord]
    energy_head = energy[head_mask][head_descend_ord]
    energy_y_head = energy_y[head_mask][head_descend_ord]

    '''-------------###################----------------------------'''
    # find why boundary_energy is important
    head_mask_true = (targets == 0) & (preds == targets)
    head_mask_false = (targets == 0) & (preds != targets)

    loss_sort, loss_sort_order = (energy_y_head - energy_head).sort(descending=True)
    plot_2data(range(head_probs.size(0)), loss_sort, boundary_energy_head[loss_sort_order], 'CE_loss')

    true_loss, true_loss_sort_order = (energy_y[head_mask_true] - energy[head_mask_true]).sort(descending=True)
    plot_2data(range(true_loss.size(0)), true_loss, boundary_energy[head_mask_true][true_loss_sort_order], 'CE_loss_true')

    false_loss, false_loss_sort_order = (energy_y[head_mask_false] - energy[head_mask_false]).sort(descending=True)
    plot_2data(range(false_loss.size(0)), false_loss, boundary_energy[head_mask_false][false_loss_sort_order], 'CE_loss_false')

    mask_non_target = torch.ones(probs.shape)
    mask_non_target[torch.arange(logits.size(0)), targets] = float('-inf')
    non_targets_max_probs = torch.max(probs * mask_non_target, dim=1)[0]
    probs_sub_sort, sort_order = (probs[torch.arange(logits.size(0)), targets]-non_targets_max_probs)[head_mask].sort(descending=True)
    plot_2data_1(range(probs_sub_sort.size(0)), probs_sub_sort, (boundary_energy - energy_y)[head_mask][sort_order], 'CE_loss', 'probs_sub','BE-E_y')

    energy_sub_sort, energy_sort_order = (boundary_energy - energy_y)[head_mask].sort(descending=True)
    plot_2data_1(range(probs_sub_sort.size(0)), (probs[torch.arange(logits.size(0)), targets]-non_targets_max_probs)[head_mask][energy_sort_order], energy_sub_sort, 'CE_loss', 'BE-E_y', 'probs_sub')




    print('aaaaa')

    '''-------------###################----------------------------'''

    boundary_energy_tail = boundary_energy[tail_mask][tail_descend_ord]
    energy_tail = energy[tail_mask][tail_descend_ord]
    energy_y_tail = energy_y[tail_mask][tail_descend_ord]

    kernel_size = 11  # 平滑滤波器的大小

    # head_probs_smooth = F.conv1d(head_probs[head_descend_ord].unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
    #                       padding=kernel_size // 2).squeeze()
    boundary_energy_head_smooth = F.conv1d(boundary_energy_head.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
                          padding=kernel_size // 2).squeeze()
    energy_head_smooth = F.conv1d(energy_head.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
                          padding=kernel_size // 2).squeeze()
    energy_y_head_smooth = F.conv1d(energy_y_head.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
                          padding=kernel_size // 2).squeeze()

    # tail_probs_smooth = F.conv1d(tail_probs[tail_descend_ord].unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
    #                       padding=kernel_size // 2).squeeze()
    boundary_energy_tail_smooth = F.conv1d(boundary_energy_tail.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
                          padding=kernel_size // 2).squeeze()
    energy_tail_smooth = F.conv1d(energy_tail.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
                          padding=kernel_size // 2).squeeze()
    energy_y_tail_smooth = F.conv1d(energy_y_tail.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, kernel_size) / kernel_size,
                          padding=kernel_size // 2).squeeze()





    plot_4data_2(range(head_probs.size(0)), head_probs[head_descend_ord], boundary_energy_head_smooth, energy_head_smooth, energy_y_head_smooth,
                 range(tail_probs.size(0)), tail_probs[tail_descend_ord], boundary_energy_tail_smooth, energy_tail_smooth, energy_y_tail_smooth, 'CEloss')


def plot_head_tail_correct_energy(logits, boundary_energy, energy, energy_y, targets):
    probs = F.softmax(logits, dim=1)
    _, preds = probs.max(dim=1)

    corrects = (preds == targets).T
    incorrects = (preds != targets).T

    mask_head = (preds != targets) & (targets == 0)
    mask_tail = (preds != targets) & (targets == 9)

    mask_head_true = (preds == targets) & (targets == 0)
    mask_tail_true = (preds == targets) & (targets == 9)

    # for head incorrect
    head_probs_descend = []

    # # boundary_eneryg
    # descend_ord = boundary_energy[mask_head].sort(descending=True)[1]
    # probs
    descend_ord = probs[torch.arange(logits.size(0)), targets][mask_head].sort(descending=True)[1]

    gap_head = []
    y_hat_head = []
    for idx in descend_ord:
        idx = torch.where(boundary_energy == boundary_energy[mask_head][idx])
        head_probs_descend.append(probs[idx].sort(descending=True)[1])
        gap_head.append(probs[idx][0][0] - probs[idx].sort(descending=True)[0][0][0])
        y_hat_head.append(1 - probs[idx][0][0])

    data1 = torch.tensor(gap_head).numpy()
    data2 = boundary_energy[mask_head].sort(descending=True)[0].numpy()
    data3 = torch.tensor(y_hat_head).numpy()

    plot(range(len(gap_head)), data1, data2, name='head_incorrect')

    # for tail incorrect
    tail_probs_descend = []
    descend_ord = boundary_energy[mask_tail].sort(descending=True)[1]
    gap_tail = []
    y_hat_tail = []
    for idx in descend_ord:
        idx = torch.where(boundary_energy == boundary_energy[mask_tail][idx])
        tail_probs_descend.append(probs[idx].sort(descending=True)[1])
        gap_tail.append(probs[idx][0][9] - probs[idx].sort(descending=True)[0][0][0])
        y_hat_tail.append(1 - probs[idx][0][9])

    data1 = torch.tensor(gap_tail).numpy()
    data2 = boundary_energy[mask_tail].sort(descending=True)[0].numpy()
    data3 = torch.tensor(y_hat_tail).numpy()

    plot(range(len(gap_tail)), data1, data2, name='tail_incorrect')

    # for head true
    head_probs_descend_true = []
    descend_ord = boundary_energy[mask_head_true].sort(descending=True)[1]
    gap_head_true = []
    y_hat_head = []
    for idx in descend_ord:
        idx = torch.where(boundary_energy == boundary_energy[mask_head_true][idx])
        head_probs_descend_true.append(probs[idx].sort(descending=True)[1])
        gap_head_true.append(probs[idx][0][0] - probs[idx].sort(descending=True)[0][0][1])
        y_hat_head.append(1 - probs[idx][0][0])

    data1 = torch.tensor(gap_head_true).numpy()
    data2 = boundary_energy[mask_head_true].sort(descending=True)[0].numpy()
    data3 = torch.tensor(y_hat_head).numpy()

    plot(range(len(gap_head_true)), data1, data2, name='head_true')

    # for tail true
    tail_probs_descend_true = []
    descend_ord = boundary_energy[mask_tail_true].sort(descending=True)[1]
    gap_tail_true = []
    y_hat_tail = []
    for idx in descend_ord:
        idx = torch.where(boundary_energy == boundary_energy[mask_tail_true][idx])
        tail_probs_descend_true.append(probs[idx].sort(descending=True)[1])
        gap_tail_true.append(probs[idx][0][9] - probs[idx].sort(descending=True)[0][0][1])
        y_hat_tail.append(1 - probs[idx][0][9])

    data1 = torch.tensor(gap_tail_true).numpy()
    data2 = boundary_energy[mask_tail_true].sort(descending=True)[0].numpy()
    data3 = torch.tensor(y_hat_tail).numpy()

    plot(range(len(gap_tail_true)), data1, data2, name='tail_true')


def plot_class_boundary_energy(targets, boundary_energy, class_bias, preds, name):
    class_counter = Counter(targets.numpy().tolist())

    # calculate the mean boundary_energy for each class
    mean_boundary_energy = []
    class_acc = []
    for key in range(len(class_counter.keys())):
        mean_boundary_energy.append(boundary_energy[torch.where(targets==key)[0]].mean())
        class_acc.append(((preds == targets) & (targets == key)).sum()*1.0 / (targets == key).sum())

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(class_counter.keys())), torch.tensor(mean_boundary_energy).numpy(), color='r', label=name, lineStyle='dashed')
    ax1.set_xlabel('Class_index')
    ax1.set_ylabel(name)
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    # 绘制第二个数据集
    ax2.plot(range(len(class_counter.keys())), torch.tensor(class_bias).numpy(), color='g', label='class—bias')
    ax2.set_ylabel('fraction', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # 绘制第三个数据集
    ax2.plot(range(len(class_counter.keys())), torch.tensor(class_acc).numpy(), color='c', label='class—acc')

    # 添加图例
    # lines1, labels1 = ax1.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    # ax1.legend(loc='upper right')

    plt.title('Logits Adjustment')
    # plt.title('CE Loss')

    # 显示图形
    plt.show()

def classes_acc_energyies(logits, boundary_energy, energy, energy_y, targets, class_bias, preds):
    class_counter = Counter(targets.numpy().tolist())

    class_boundary_energy, class_correct_boundary_energy, class_incorrect_boundary_energy = [], [], []
    class_energy, class_correct_energy, class_incorrect_energy = [], [], []
    class_energy_y, class_correct_energy_y, class_incorrect_energy_y = [], [], []
    class_acc = []
    for key in range(len(class_counter.keys())):
        class_acc.append(((targets == key) & (preds == targets)).sum(dtype=float) / (targets == key).sum())

        class_boundary_energy.append(boundary_energy[targets == key].mean())
        class_correct_boundary_energy.append(boundary_energy[(targets == key) & (preds == targets)].mean())
        class_incorrect_boundary_energy.append(boundary_energy[(targets == key) & (preds != targets)].mean())

        class_energy.append(energy[targets == key].mean())
        class_correct_energy.append(energy[(targets == key) & (preds == targets)].mean())
        class_incorrect_energy.append(energy[(targets == key) & (preds != targets)].mean())

        class_energy_y.append(energy_y[targets == key].mean())
        class_correct_energy_y.append(energy_y[(targets == key) & (preds == targets)].mean())
        class_incorrect_energy_y.append(energy_y[(targets == key) & (preds != targets)].mean())

        print(key)

    data = {
        'key': list(range(len(class_counter.keys()))),
        'class_acc': class_acc,
        'class_boundary_energy': class_boundary_energy,
        'class_correct_boundary_energy': class_correct_boundary_energy,
        'class_incorrect_boundary_energy': class_incorrect_boundary_energy,

        'class_energy': class_energy,
        'class_correct_energy': class_correct_energy,
        'class_incorrect_energy': class_incorrect_energy,

        'class_energy_y': class_energy_y,
        'class_correct_energy_y': class_correct_energy_y,
        'class_incorrect_energy_y': class_incorrect_energy_y
    }

    data['class_acc'] = [a.item() for a in data['class_acc']]
    data['class_boundary_energy'] = [a.item() for a in data['class_boundary_energy']]
    data['class_correct_boundary_energy'] = [a.item() for a in data['class_correct_boundary_energy']]
    data['class_incorrect_boundary_energy'] = [a.item() for a in data['class_incorrect_boundary_energy']]
    data['class_energy'] = [a.item() for a in data['class_energy']]
    data['class_correct_energy'] = [a.item() for a in data['class_correct_energy']]
    data['class_incorrect_energy'] = [a.item() for a in data['class_incorrect_energy']]
    data['class_energy_y'] = [a.item() for a in data['class_energy_y']]
    data['class_correct_energy_y'] = [a.item() for a in data['class_correct_energy_y']]
    data['class_incorrect_energy_y'] = [a.item() for a in data['class_incorrect_energy_y']]

    df = pd.DataFrame(data)
    df.to_excel('Image_net_logits_adj.xlsx', index=False)
    # df.to_excel('Image_net_CE_loss.xlsx', index=False)



def read_excel(results_log, results_CE):

    logits_adj = pd.read_excel(results_log, engine='openpyxl')
    CE = pd.read_excel(results_CE, engine='openpyxl')

    indices = CE.key.tolist()

    # plot_2data_1(indices, logits_adj.class_acc.tolist(), CE.class_acc.tolist(),'class_acc')

    # plot_2data_1(indices, logits_adj.class_boundary_energy.tolist(), CE.class_boundary_energy.tolist(),'class_boundary_energy')
    # plot_2data_1(indices, logits_adj.class_correct_boundary_energy.tolist(), CE.class_correct_boundary_energy.tolist(),'class_correct_boundary_energy')
    # plot_2data_1(indices, logits_adj.class_incorrect_boundary_energy.tolist(), CE.class_incorrect_boundary_energy.tolist(),'class_incorrect_boundary_energy')

    # plot_2data_1(indices, logits_adj.class_energy.tolist(), CE.class_energy.tolist(),'class_energy')
    # plot_2data_1(indices, logits_adj.class_correct_energy.tolist(), CE.class_correct_energy.tolist(),'class_correct_energy')
    # plot_2data_1(indices, logits_adj.class_incorrect_energy.tolist(), CE.class_incorrect_energy.tolist(),'class_incorrect_energy')

    # plot_2data_1(indices, logits_adj.class_energy_y.tolist(), CE.class_energy_y.tolist(),'class_energy_y')
    # plot_2data_1(indices, logits_adj.class_correct_energy_y.tolist(), CE.class_correct_energy_y.tolist(),'class_correct_energy_y')
    # plot_2data_1(indices, logits_adj.class_incorrect_energy_y.tolist(), CE.class_incorrect_energy_y.tolist(),'class_incorrect_energy_y')

    plot_3data_2sets_2(indices,
                       [a-b + c for a, b, c in zip(logits_adj.class_energy_y.tolist(), logits_adj.class_energy.tolist(), logits_adj.class_boundary_energy.tolist())],
                       [a-b - c for a, b, c in zip(logits_adj.class_energy_y.tolist(), logits_adj.class_energy.tolist(), logits_adj.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(logits_adj.class_energy_y.tolist(), logits_adj.class_energy.tolist())],
                       [a - b + c for a, b, c in zip(CE.class_energy_y.tolist(), CE.class_energy.tolist(), CE.class_boundary_energy.tolist())],
                       [a - b - c for a, b, c in zip(CE.class_energy_y.tolist(), CE.class_energy.tolist(), CE.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_energy_y.tolist(), CE.class_energy.tolist())],
                       'total')



    plot_3data_2sets_3(indices,
                       logits_adj.class_boundary_energy.tolist(), logits_adj.class_energy.tolist(), logits_adj.class_energy_y.tolist(),
                       CE.class_boundary_energy.tolist(), CE.class_energy.tolist(), CE.class_energy_y.tolist(),
                       'total')

    plot_3data_2sets_3(indices,
                       logits_adj.class_correct_boundary_energy.tolist(), logits_adj.class_correct_energy.tolist(), logits_adj.class_correct_energy_y.tolist(),
                       CE.class_correct_boundary_energy.tolist(), CE.class_correct_energy.tolist(), CE.class_correct_energy_y.tolist(),
                       'correct')

    plot_3data_2sets_3(indices,
                       logits_adj.class_incorrect_boundary_energy.tolist(), logits_adj.class_incorrect_energy.tolist(), logits_adj.class_incorrect_energy_y.tolist(),
                       CE.class_incorrect_boundary_energy.tolist(), CE.class_incorrect_energy.tolist(), CE.class_incorrect_energy_y.tolist(),
                       'incorrect')

    # sub -- energy
    plot_3data_2sets_3(indices,
                       [a-b for a, b in zip(logits_adj.class_boundary_energy.tolist(), logits_adj.class_energy.tolist())],
                       [a-b for a, b in zip(logits_adj.class_energy.tolist(), logits_adj.class_energy.tolist())],
                       [a-b for a, b in zip(logits_adj.class_energy_y.tolist(),logits_adj.class_energy.tolist())],
                       [a-b for a,b in zip(CE.class_boundary_energy.tolist() , CE.class_energy.tolist())],
                       [a-b for a,b in zip(CE.class_energy.tolist() , CE.class_energy.tolist())],
                       [a-b for a, b in zip(CE.class_energy_y.tolist() , CE.class_energy.tolist())],
                       'total--sub--energy')

    plot_3data_2sets_3(indices,
                       [a-b for a, b in zip(logits_adj.class_correct_boundary_energy.tolist(), logits_adj.class_correct_energy.tolist())],
                       [a-b for a, b in zip(logits_adj.class_correct_energy.tolist(), logits_adj.class_correct_energy.tolist())],
                       [a-b for a, b in zip(logits_adj.class_correct_energy_y.tolist(), logits_adj.class_correct_energy.tolist())],
                       [a-b for a, b in zip(CE.class_correct_boundary_energy.tolist(), CE.class_correct_energy.tolist())],
                       [a-b for a, b in zip(CE.class_correct_energy.tolist(), CE.class_correct_energy.tolist())],
                       [a-b for a, b in zip(CE.class_correct_energy_y.tolist(), CE.class_correct_energy.tolist())],
                       'correct--sub--energy')

    plot_3data_2sets_3(indices,
                       [a-b for a, b in zip(logits_adj.class_incorrect_boundary_energy.tolist(), logits_adj.class_incorrect_energy.tolist())],
                       [a-b for a, b in zip(logits_adj.class_incorrect_energy.tolist(), logits_adj.class_incorrect_energy.tolist())],
                       [a-b for a, b in zip(logits_adj.class_incorrect_energy_y.tolist(), logits_adj.class_incorrect_energy.tolist())],
                       [a-b for a, b in zip(CE.class_incorrect_boundary_energy.tolist(), CE.class_incorrect_energy.tolist())],
                       [a-b for a, b in zip(CE.class_incorrect_energy.tolist(), CE.class_incorrect_energy.tolist())],
                       [a-b for a, b in zip(CE.class_incorrect_energy_y.tolist(), CE.class_incorrect_energy.tolist())],
                       'incorrect--sub--energy')

    # sub -- boundary_energy
    plot_3data_2sets_3(indices,
                       [a - b for a, b in zip(logits_adj.class_boundary_energy.tolist(), logits_adj.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(logits_adj.class_energy.tolist(), logits_adj.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(logits_adj.class_energy_y.tolist(), logits_adj.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_boundary_energy.tolist(), CE.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_energy.tolist(), CE.class_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_energy_y.tolist(), CE.class_boundary_energy.tolist())],
                       'total--sub--boundary_energy')

    plot_3data_2sets_3(indices,
                       [a - b for a, b in zip(logits_adj.class_correct_boundary_energy.tolist(), logits_adj.class_correct_boundary_energy.tolist())],
                       [a - b for a, b in zip(logits_adj.class_correct_energy.tolist(), logits_adj.class_correct_boundary_energy.tolist())],
                       [a - b for a, b in zip(logits_adj.class_correct_energy_y.tolist(), logits_adj.class_correct_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_correct_boundary_energy.tolist(), CE.class_correct_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_correct_energy.tolist(), CE.class_correct_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_correct_energy_y.tolist(), CE.class_correct_boundary_energy.tolist())],
                       'correct--sub--boundary_energy')

    plot_3data_2sets_3(indices,
                       [a - b for a, b in zip(logits_adj.class_incorrect_boundary_energy.tolist(), logits_adj.class_incorrect_boundary_energy.tolist())],
                       [a - b for a, b in
                        zip(logits_adj.class_incorrect_energy.tolist(), logits_adj.class_incorrect_boundary_energy.tolist())],
                       [a - b for a, b in
                        zip(logits_adj.class_incorrect_energy_y.tolist(), logits_adj.class_incorrect_boundary_energy.tolist())],
                       [a - b for a, b in
                        zip(CE.class_incorrect_boundary_energy.tolist(), CE.class_incorrect_boundary_energy.tolist())],
                       [a - b for a, b in zip(CE.class_incorrect_energy.tolist(), CE.class_incorrect_boundary_energy.tolist())],
                       [a - b for a, b in
                        zip(CE.class_incorrect_energy_y.tolist(), CE.class_incorrect_boundary_energy.tolist())],
                       'incorrect--sub--boundary_energy')


    # sub -- energy
    plot_3data_2sets_3(indices,
                       [a-b for a, b in zip(logits_adj.class_boundary_energy.tolist(), logits_adj.class_energy_y.tolist())],
                       [a-b for a, b in zip(logits_adj.class_energy.tolist(), logits_adj.class_energy_y.tolist())],
                       [a-b for a, b in zip(logits_adj.class_energy_y.tolist(),logits_adj.class_energy_y.tolist())],
                       [a-b for a,b in zip(CE.class_boundary_energy.tolist() , CE.class_energy_y.tolist())],
                       [a-b for a,b in zip(CE.class_energy.tolist() , CE.class_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_energy_y.tolist() , CE.class_energy_y.tolist())],
                       'total--sub--energy_y')

    plot_3data_2sets_3(indices,
                       [a-b for a, b in zip(logits_adj.class_correct_boundary_energy.tolist(), logits_adj.class_correct_energy_y.tolist())],
                       [a-b for a, b in zip(logits_adj.class_correct_energy.tolist(), logits_adj.class_correct_energy_y.tolist())],
                       [a-b for a, b in zip(logits_adj.class_correct_energy_y.tolist(), logits_adj.class_correct_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_correct_boundary_energy.tolist(), CE.class_correct_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_correct_energy.tolist(), CE.class_correct_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_correct_energy_y.tolist(), CE.class_correct_energy_y.tolist())],
                       'correct--sub--energy_y')

    plot_3data_2sets_3(indices,
                       [a-b for a, b in zip(logits_adj.class_incorrect_boundary_energy.tolist(), logits_adj.class_incorrect_energy_y.tolist())],
                       [a-b for a, b in zip(logits_adj.class_incorrect_energy.tolist(), logits_adj.class_incorrect_energy_y.tolist())],
                       [a-b for a, b in zip(logits_adj.class_incorrect_energy_y.tolist(), logits_adj.class_incorrect_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_incorrect_boundary_energy.tolist(), CE.class_incorrect_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_incorrect_energy.tolist(), CE.class_incorrect_energy_y.tolist())],
                       [a-b for a, b in zip(CE.class_incorrect_energy_y.tolist(), CE.class_incorrect_energy_y.tolist())],
                       'incorrect--sub--energy_y')

    print('aaaa')





def cal_boundary_energy(logits, targets, class_bias, preds):

    logits_wo_y = torch.cat([torch.cat((logits[i][0:j],logits[i][j+1:])) for i, j in enumerate(targets)]
                      ).view(logits.shape[0], logits.shape[1]-1)
    boundary_energy = -torch.log(torch.exp(logits_wo_y).sum(dim=1))
    # torch.max(boundary_energy[boundary_energy != 0])
    energy = - torch.log(torch.exp(logits).sum(dim=1))
    energy_y = - logits[torch.arange(logits.size(0)), targets]

    # plot

    # plot_class_boundary_energy(targets, boundary_energy, class_bias, preds,'boundary_energy')
    # plot_class_boundary_energy(targets, boundary_energy/energy, class_bias, preds, 'BE_div_E')
    # plot_class_boundary_energy(targets, energy, class_bias, preds, 'energy')
    # plot_class_boundary_energy(targets, torch.tanh((energy_y - energy - boundary_energy) / 10.0), class_bias, preds,
    #                            'sub')

    # plot_head_tail_correct_energy(logits, boundary_energy, energy, energy_y, targets)

    # plot_class_boundary_energy(targets, (energy - energy_y) / boundary_energy, class_bias, preds, 'sub_div')
    # plot_class_boundary_energy(targets, boundary_energy - energy, class_bias, preds, 'BE_sub_E')
    # plot_class_boundary_energy(targets, torch.exp(boundary_energy / energy), class_bias, preds, 'exp(BE/E)')

    # plot_class_boundary_energy(targets, energy_y - boundary_energy - energy, class_bias, preds, 'E_y-BE-E')
    # plot_class_boundary_energy(targets, energy_y - energy, class_bias, preds, 'E_y-E')
    # plot_class_boundary_energy(targets, - boundary_energy - energy, class_bias, preds, '-BE-E')
    # plot_class_boundary_energy(targets, boundary_energy - energy, class_bias, preds, 'BE_sub_E')

    # plot_class_energy_correct_incorrect(logits, boundary_energy, energy, energy_y, targets, class_bias)

    # class_energys_loss_correct_incorrect(logits, boundary_energy, energy, energy_y, targets, class_bias)
    # plot_prob_energy(logits, boundary_energy-energy_y, -energy, targets)

    # classes_acc_energyies(logits, boundary_energy, energy, energy_y, targets, class_bias, preds)

    # read_excel('Image_net_logits_adj.xlsx', 'Image_net_CE_loss.xlsx')
    # read_excel('logits_adj.xlsx', 'CE_loss.xlsx')

    # find_boundary_bad_samples(logits, boundary_energy, energy, energy_y, targets, save_path)


    return boundary_energy