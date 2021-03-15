import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def get_results(filename):
    res = list()
    with open("vugraph/{}".format(filename), 'r') as f:
        for line in f.readlines():
            res.append(float(line))
    return res

def plot_diff(a, b, title):
    bins = np.zeros(9)
    diffs = np.array([b[i] - a[i] for i in range(len(b))], dtype=np.int32)
    for diff in diffs:
        if diff <= -4:
            bins[0] +=1
        elif diff >= 4:
            bins[8] +=1
        else:
            bins[diff+4] += 1

    fig, ax = plt.subplots()
    bins = np.divide(bins, len(b))
    ax.bar(np.arange(9)-4, bins)
    plt.xticks(np.arange(9)-4, labels=['<=-4', '-3', '-2', '-1', '0', '1', '2', '3', '>=4'])
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{round(height, 3)}', (x + width/2, y + height*1.02), ha='center')
    plt.xlabel("DDS tricks - Expert tricks", size=14)
    plt.ylim(0, 0.6)
    plt.ylabel("Ratio",size=14)
    plt.savefig(os.path.join('vugraph', 'results', title), bbox_inches='tight')
    # plt.show()

def plot_time_diff(all_times, search_times, belief_times, labels):
    fig, ax = plt.subplots()
    averages = [np.average(a) for a in all_times]
    search_averages = [np.average(a) for a in search_times]
    belief_averages = [np.average(a) for a in belief_times]
    ax.bar(labels[0], averages[0])
    ax.bar(labels[1:], search_averages, label='Search time')
    ax.bar(labels[1:], belief_averages, bottom=search_averages, label='Belief time')
    for p_idx, p in enumerate(ax.patches):
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        if p_idx > 0 and y == 0:
            continue
        ax.annotate(f'{round(height + y, 3)}', (x + width/2, y + height*1.02), ha='center')
    plt.legend()
    plt.title('Average runtime for various POMCP simulations and particles',size=22)
    plt.xlabel("Number of simulations s and number of particles p",size=14)
    plt.ylabel("Average runtime (s)",size=14)
    plt.show()




true_tricks = get_results("true_results.txt")
dds_tricks = get_results("dds_results.txt")
all_times = list()
dds_times = get_results("dds_times.txt")
all_times.append(dds_times)
pomcp_tricks = list()
pomcp_times = list()
pomcp_search_times = list()
pomcp_belief_times = list()
pomcp_names = list()
for sims in [100,1000,10000]:
    for particles in [100,1000,10000]:
        pomcp_tricks.append(get_results("pomcp_results_{}_{}.txt".format(sims, particles)))
        all_times.append(get_results("pomcp_times_{}_{}.txt".format(sims, particles)))
        pomcp_search_times.append(get_results("pomcp_search_times_{}_{}.txt".format(sims, particles)))
        pomcp_belief_times.append(get_results("pomcp_belief_times_{}_{}.txt".format(sims, particles)))
        pomcp_names.append("s{}_p{}".format(sims, particles))

# for pomcp_idx, pomcp in enumerate(pomcp_tricks):
#     plot_diff(true_tricks, pomcp, pomcp_names[pomcp_idx])

plot_diff(true_tricks, dds_tricks, 'DDS')

# plot_time_diff(all_times, pomcp_search_times, pomcp_belief_times, ['dds_times'] + pomcp_names)
