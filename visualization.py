import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# https://matplotlib.org/stable/gallery/color/named_colors.html
colors = [
    ['steelblue'],
    ['steelblue', 'red'],
    ['darkorange', 'deepskyblue', 'red'],
    ['darkorange', 'deepskyblue', 'red', 'limegreen'],
    ['darkorange', 'deepskyblue', 'red', 'limegreen', 'darkviolet'],
    ['darkorange', 'deepskyblue', 'red', 'limegreen', 'darkviolet', 'cyan']
]
MAX_N_COLORS = 6


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
        if n < MAX_N_COLORS:
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


def export_label(label, path):
    mx = label.max()
    scale = 255 if mx <= 1 else 1

    img = Image.fromarray((label * scale).astype(np.uint8)).squeeze()
    img.save(path)
