import matplotlib.pyplot as plt
import numpy as np

# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
colors = [
    ['steelblue'],
    ['steelblue', 'firebrick'],
    ['orange', 'lightskyblue', 'firebrick'],
    ['orange', 'lightskyblue', 'firebrick', 'limegreen'],
    ['orange', 'lightskyblue', 'firebrick', 'limegreen', 'darkviolet'],
    ['orange', 'lightskyblue', 'firebrick', 'limegreen', 'darkviolet', 'deeppink'],
]


def plot(values, title, output_dir, output_name,
         xlabel='iterations', ylabel='loss', linewidth=1.5, marker=None, figsize=(10, 5), upperbound=0.0,
         xticks=None, xticklabels=None, yticks=None, yticklabels=None, areas=None):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.plot(values, colors[0][0], linewidth=linewidth, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=.75)

    if xticks is not None:
        plt.xticks(xticks, xticklabels if xticklabels is not None else xticks)

    if yticks is not None:
        plt.yticks(yticks, yticklabels if yticklabels is not None else yticks)

    if areas is not None:
        for a in areas:
            plt.axvspan(a['min'], a['max'], color=a['color'], alpha=a['alpha'])

    if upperbound > 0:
        plt.axhline(upperbound, color='k', linestyle='--', linewidth=1.)

    plt.savefig(f'{output_dir}/{output_name}.png')
    plt.close('all')


def plot_multi(list_of_values, title, labels, output_dir, output_name,
               xlabel='iterations', ylabel='loss', linewidth=1.5, markers=None, figsize=(10, 5), upperbound=0.0,
               xticks=None, xticklabels=None, yticks=None, yticklabels=None, areas=None):
    plt.figure(figsize=figsize)
    plt.title(title)
    n = len(list_of_values) - 1
    for i in range(len(list_of_values)):
        if markers is None:
            m = None
        else:
            m = markers[i]
        if n < 3:
            plt.plot(list_of_values[i], label=labels[i], color=colors[n][i], linewidth=linewidth, marker=m)
        else:
            plt.plot(list_of_values[i], label=labels[i], linewidth=linewidth, marker=m)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=.75)

    if xticks is not None:
        plt.xticks(xticks, xticklabels if xticklabels is not None else xticks)

    if yticks is not None:
        plt.yticks(yticks, yticklabels if yticklabels is not None else yticks)

    if areas is not None:
        for a in areas:
            plt.axvspan(a['min'], a['max'], color=a['color'], alpha=a['alpha'])

    if upperbound > 0:
        plt.axhline(upperbound, color='k', linestyle='--', linewidth=1.)

    plt.savefig(f'{output_dir}/{output_name}.png')
    plt.close('all')

    save_obj = {}
    for i in range(len(list_of_values)):
        save_obj[labels[i]] = list_of_values[i]
    np.save(f'{output_dir}/{output_name}.npy', save_obj)
