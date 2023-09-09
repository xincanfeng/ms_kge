# example: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib import font_manager as fm
import datetime

def data_for_each_chart(j):
    if j==1:
        # FB15k-237, RotatE, MRR
        table_title = 'FB15k-237, RotatE, MRR'
        table_ylabel = 'MRR'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (33.7, 34.1, 34.0),
            'MBS': (33.9, 34.3, 34.3),
            'MIX': (34.0, 34.6, 34.6),
        }
        ylim_start, ylim_end = 33, 35
    elif j==2:
        # FB15k-237, RotatE, Hits@1
        table_title = 'FB15k-237, RotatE, Hits@1'
        table_ylabel = 'Hits@1'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (53.3, 53.2, 53.0),
            'MBS': (53.5, 53.6, 53.6),
            'MIX': (53.4, 53.6, 53.7),
        }
        ylim_start, ylim_end = 52, 55
    elif j==3:
        # FB15k-237, RotatE, Hits@3
        table_title = 'FB15k-237, RotatE, Hits@3'
        table_ylabel = 'Hits@3'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (37.6, 37.6, 37.5),
            'MBS': (37.5, 38.0, 37.9),
            'MIX': (37.6, 38.1, 38.4),
        }
        ylim_start, ylim_end = 37, 39
    elif j==4:
        # FB15k-237, RotatE, Hits@10
        table_title = 'FB15k-237, RotatE, Hits@10'
        table_ylabel = 'Hits@10'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (53.3, 53.2, 53.0),
            'MBS': (53.5, 53.6, 53.6),
            'MIX': (53.4, 53.6, 53.7),
        }
        ylim_start, ylim_end = 52, 55
    elif j==5:
        # FB15k-237, ComplEx, MRR
        table_title = 'FB15k-237, ComplEx, MRR'
        table_ylabel = 'MRR'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (32.2, 32.7, 32.6),
            'MBS': (31.3, 31.8, 31.9),
            'MIX': (32.5, 32.8, 32.6),
        }
        ylim_start, ylim_end = 31, 34
    elif j==6:
        # FB15k-237, ComplEx, Hits@1
        table_title = 'FB15k-237, ComplEx, Hits@1'
        table_ylabel = 'Hits@1'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (51.2, 51.2, 51.0),
            'MBS': (51.7, 49.9, 50.6),
            'MIX': (52.3, 51.4, 51.2),
        }
        ylim_start, ylim_end = 49, 53
    elif j==7:
        # FB15k-237, ComplEx, Hits@3
        table_title = 'FB15k-237, ComplEx, Hits@3'
        table_ylabel = 'Hits@3'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (35.4, 36.0, 35.8),
            'MBS': (34.8, 34.9, 34.9),
            'MIX': (35.8, 36.2, 35.6),
        }
        ylim_start, ylim_end = 34, 37
    elif j==8: 
        # FB15k-237, ComplEx, Hits@10
        table_title = 'FB15k-237, ComplEx, Hits@10'
        table_ylabel = 'Hits@10'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (51.2, 51.2, 51.0),
            'MBS': (51.7, 49.9, 50.6),
            'MIX': (52.3, 51.4, 51.2),
        }
        ylim_start, ylim_end = 49, 53 
    elif j==9:
        # FB15k-237, HAKE, MRR
        table_title = 'FB15k-237, HAKE, MRR'
        table_ylabel = 'MRR'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (34.6, 35.2, 35.2),
            'MBS': (34.4, 35.3, 35.3),
            'MIX': (34.5, 35.3, 35.4),
        }
        ylim_start, ylim_end = 34, 36
    elif j==10:
        # FB15k-237, HAKE, Hits@1
        table_title = 'FB15k-237, HAKE, Hits@1'
        table_ylabel = 'Hits@1'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (54.1, 54.5, 54.6),
            'MBS': (54.2, 54.4, 54.7),
            'MIX': (54.2, 54.5, 54.6),
        }
        ylim_start, ylim_end = 54, 55
    elif j==11:
        # FB15k-237, HAKE, Hits@3
        table_title = 'FB15k-237, HAKE, Hits@3'
        table_ylabel = 'Hits@3'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (38.4, 38.7, 38.7),
            'MBS': (38.0, 39.1, 39.2),
            'MIX': (38.5, 39.0, 39.0),
        }
        ylim_start, ylim_end = 37, 40
    elif j==12:
        # FB15k-237, HAKE, Hits@10
        table_title = 'FB15k-237, HAKE, Hits@10'
        table_ylabel = 'Hits@10'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (54.1, 54.5, 54.6),
            'MBS': (54.2, 54.4, 54.7),
            'MIX': (54.2, 54.5, 54.6),
        }
        ylim_start, ylim_end = 54, 55
    elif j==13:
        # WN18RR, RotatE, MRR
        table_title = 'WN18RR, RotatE, MRR'
        table_ylabel = 'MRR'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (47.9, 47.9, 47.9),
            'MBS': (48.0, 48.0, 48.0),
            'MIX': (47.9, 47.9, 48.2),
        }
        ylim_start, ylim_end = 47, 49
    elif j==14:
        # WN18RR, RotatE, Hits@1
        table_title = 'WN18RR, RotatE, Hits@1'
        table_ylabel = 'Hits@1'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (55.7, 56.8, 57.3),
            'MBS': (57.4, 57.0, 56.8),
            'MIX': (56.7, 56.9, 56.8),
        }
        ylim_start, ylim_end = 55, 58 
    elif j==15:
        # WN18RR, RotatE, Hits@3
        table_title = 'WN18RR, RotatE, Hits@3'
        table_ylabel = 'Hits@3'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (49.5, 49.6, 49.7),
            'MBS': (49.8, 49.8, 49.6),
            'MIX': (49.7, 49.5, 49.8),
        }
        ylim_start, ylim_end = 49, 50
    elif j==16:
        # WN18RR, RotatE, Hits@10
        table_title = 'WN18RR, RotatE, Hits@10'
        table_ylabel = 'Hits@10'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (55.7, 56.8, 57.3),
            'MBS': (57.4, 57.0, 56.8),
            'MIX': (56.7, 56.9, 56.8),
        }
        ylim_start, ylim_end = 55, 58
    elif j==17:
        # WN18RR, ComplEx, MRR
        table_title = 'WN18RR, ComplEx, MRR'
        table_ylabel = 'MRR'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (46.8, 47.2, 47.5),
            'MBS': (47.2, 48.6, 48.3),
            'MIX': (47.3, 48.6, 48.4),
        }
        ylim_start, ylim_end = 46, 50
    elif j==18:
        # WN18RR, ComplEx, Hits@1
        table_title = 'WN18RR, ComplEx, Hits@1'
        table_ylabel = 'Hits@1'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (55.6, 56.2, 56.2),
            'MBS': (54.7, 56.8, 56.5),
            'MIX': (55.3, 57.0, 56.4),
        }
        ylim_start, ylim_end = 54, 58
    elif j==19:
        # WN18RR, ComplEx, Hits@3
        table_title = 'WN18RR, ComplEx, Hits@3'
        table_ylabel = 'Hits@3'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (48.5, 49.3, 49.0),
            'MBS': (48.8, 50.4, 49.9),
            'MIX': (49.0, 50.3, 49.9),
        }
        ylim_start, ylim_end = 48, 51
    elif j==20:
        # WN18RR, ComplEx, Hits@10
        table_title = 'WN18RR, ComplEx, Hits@10'
        table_ylabel = 'Hits@10'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (55.6, 56.2, 56.2),
            'MBS': (54.7, 56.8, 56.5),
            'MIX': (55.3, 57.0, 56.4),
        }
        ylim_start, ylim_end = 54, 58 
    elif j==21:
        # WN18RR, HAKE, MRR
        table_title = 'WN18RR, HAKE, MRR'
        table_ylabel = 'MRR'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (49.5, 50.0, 49.7),
            'MBS': (49.1, 49.9, 50.1),
            'MIX': (49.4, 49.8, 50.0),
        }
        ylim_start, ylim_end = 49, 51 
    elif j==22:
        # WN18RR, HAKE, Hits@1
        table_title = 'WN18RR, HAKE, Hits@1'
        table_ylabel = 'Hits@1'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (58.0, 58.2, 58.2),
            'MBS': (57.9, 58.3, 58.5),
            'MIX': (58.4, 58.1, 58.5),
        }
        ylim_start, ylim_end = 57, 59
    elif j==23:
        # WN18RR, HAKE, Hits@3
        table_title = 'WN18RR, HAKE, Hits@3'
        table_ylabel = 'Hits@3'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (51.3, 51.9, 51.5),
            'MBS': (51.3, 52.0, 51.8),
            'MIX': (51.3, 51.6, 51.7),
        }
        ylim_start, ylim_end = 51, 53
    elif j==24:
        # WN18RR, HAKE, Hits@10
        table_title = 'WN18RR, HAKE, Hits@10'
        table_ylabel = 'Hits@10'
        species = ("Base", "Freq", "Uniq")
        penguin_means = {
            'CBS': (58.0, 58.2, 58.2),
            'MBS': (57.9, 58.3, 58.5),
            'MIX': (58.4, 58.1, 58.5),
        }
        ylim_start, ylim_end = 57, 59
    return table_title, table_ylabel, species, penguin_means, ylim_start, ylim_end

j = 1
for j in range(1,25):
    table_title, table_ylabel, species, penguin_means, ylim_start, ylim_end = data_for_each_chart(j)

    # color_map = cm.get_cmap('tab20c', len(penguin_means))
    colors = ['#666666', '#33CCCC', '#6600FF'] 

    x = np.arange(len(species))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for i, (attribute, measurement) in enumerate(penguin_means.items()):
        offset = width * multiplier
        # rects = ax.bar(x + offset, measurement, width, label=attribute, color=color_map(i))
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    fm.fontManager.addfont('/cl/home/xincan-f/ceci_workspace/kge/MS-KGE/Nanum_Gothic/NanumGothic-Regular.ttf')
    fm.fontManager.addfont('/cl/home/xincan-f/ceci_workspace/kge/MS-KGE/Nanum_Gothic/NanumGothic-Bold.ttf')

    font_big = {'family': 'NANUMGOTHIC', 'size':20, 'weight':'bold'}
    font_medium = {'family': 'NANUMGOTHIC', 'size':18}
    font_small = {'family': 'NANUMGOTHIC', 'size':16}

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(table_ylabel, fontdict=font_medium)
    # Name using 'dataset, model, evaluation_index'
    ax.set_title(table_title, fontdict=font_big)
    ax.set_xticks(x + width, species, fontdict=font_medium)
    ax.legend(loc='upper right', ncols=3, prop=font_small)
    ax.set_ylim(ylim_start, ylim_end)

    # plt.savefig('results/result.png')
    now = datetime.datetime.now()
    # plt.savefig('results/result_{}.png'.format(now))
    plt.savefig('charts/chart_{}.png'.format(table_title))
    j += 1

print("作图成功，现在的时间是:", now)