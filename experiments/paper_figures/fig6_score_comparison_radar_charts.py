import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import statistics

def compute_avg_stdev(filename, sheet_name):
    data = pd.read_excel(open(filename, 'rb'), sheet_name=sheet_name)
    frame = pd.DataFrame(data)
    Tasks = frame.Task.tolist()
    Datasets = frame.Dataset.tolist()
    KGpipFLAML_data = frame.KGpipFLAML.tolist()

    bc_KGpipFLAML = []
    mc_KGpipFLAML = []
    r_KGpipFLAML = []
    for i in range(0, len(Tasks)):
        if Tasks[i] == 'binary-classification':
            bc_KGpipFLAML.append(KGpipFLAML_data[i])
        elif Tasks[i] == 'multi-classification':
            mc_KGpipFLAML.append(KGpipFLAML_data[i])
        elif Tasks[i] == 'regression':
            r_KGpipFLAML.append(KGpipFLAML_data[i])

    print(
        f'KGPip+FLAML: \n\tbinary classification (mean/stdev): {statistics.mean(bc_KGpipFLAML):.2f} ({statistics.stdev(bc_KGpipFLAML):.2f})'
        f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_KGpipFLAML):.2f} ({statistics.stdev(mc_KGpipFLAML):.2f})'
        f'\n\tregression (mean/stdev): {statistics.mean(r_KGpipFLAML):.2f} ({statistics.stdev(r_KGpipFLAML):.2f})')


def plot(task, show_legend = False):
    FLAML = []
    KGpipAutoSklearn = []
    AutoSklearn = []
    KGpipFLAML = []
    volcanoML = []
    AL = []
    categories = []

    for i in range(0, len(Tasks)):
        if Tasks[i] == task:
            categories.append(int(Datasets[i]))#[:max_len_dataset])
            FLAML.append(FLAML_data[i])
            KGpipAutoSklearn.append(KGpipAutoSklearn_data[i])
            AutoSklearn.append(AutoSklearn_data[i])
            KGpipFLAML.append(KGpipFLAML_data[i])
            volcanoML.append(volcanoML_data[i])
            AL.append(AL_data[i])
    categories = [*categories, categories[0]]
    FLAML = [*FLAML, FLAML[0]]
    KGPip_FLAML = [*KGpipFLAML, KGpipFLAML[0]]
    AutoSklearn = [*AutoSklearn, AutoSklearn[0]]
    KGpipAutoSklearn = [*KGpipAutoSklearn, KGpipAutoSklearn[0]]
    volcanoML = [*volcanoML, volcanoML[0]]
    AL = [*AL, AL[0]]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(KGpipAutoSklearn))

    fig = plt.figure(figsize=(10, 8))
    # fig = plt.figure(figsize=(10, 5))

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    matplotlib.rc('font', **font)

    ax = plt.subplot(polar=True)
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8])
    # plt.xticks(fontsize=10)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.plot(label_loc, FLAML, label='FLAML', linewidth=3)
    plt.plot(label_loc, KGPip_FLAML, label='KGpipFLAML', linewidth=3)
    plt.plot(label_loc, AutoSklearn, label='AutoSklearn', linewidth=3)
    plt.plot(label_loc, KGpipAutoSklearn, label='KGPipAutoSklearn', linewidth=3)
    plt.plot(label_loc, volcanoML, label='Volcano', linewidth=3)
    plt.plot(label_loc, AL, label='AL', linewidth=3)

    if task == 'binary':
        name = "Binary Classification"
    elif task == 'multi-class':
        name = "Multi-Class Classification"
    else:
        name = "Regression"
    plt.title(name, size=18, loc="center")
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    if show_legend:
        # plt.legend(loc="lower right", mode = 'extend', borderaxespad=0, ncol=8,
        #            handletextpad=0.1, prop={'size': 13}, bbox_to_anchor=(0.3, 0.96, 1, 0.2))
        plt.legend(loc="lower right", mode = 'extend', borderaxespad=0, ncol=8,
                   handletextpad=0.1, prop={'size': 13}, bbox_to_anchor=(0.3, -0.12, 1, 0.2))

        #plt.legend(bbox_to_anchor=(x0, y0, width, height), loc=)
        # plt.legend(mode = 'extend', borderaxespad=0, ncol=12)#, bbox_to_anchor=(-0.25, 0.95, 0.1, 0.2), loc='upper left')#, labelspacing = 0)
    # plt.tight_layout()
    plt.show()
    fig.savefig(task + '.pdf', dpi=fig.dpi)


data = pd.read_excel(open('KGpipResults vs. VolcanoML.xlsx', 'rb'), sheet_name='avg. 1h_77_datasets')
frame = pd.DataFrame(data)
#ID	Dataset	FLAML	KGpipFLAML	KGpipAutoSklearn	AutoSklearn	VolcanoWE	AL	Task
frame = frame.sort_values('KGpipAutoSklearn')

Tasks = frame.Task.tolist()
Datasets = frame.ID.tolist()

FLAML_data = frame.FLAML.tolist()
KGpipFLAML_data = frame.KGpipFLAML.tolist()
KGpipAutoSklearn_data = frame.KGpipAutoSklearn.tolist()
AutoSklearn_data = frame.AutoSklearn.tolist()
volcanoML_data = frame.VolcanoWE.tolist()
AL_data = frame.AL.tolist()
# max_len_dataset = 5

plot('binary')
plot('multi-class', show_legend=True)
plot('regression')

# print('3 Graphs')
# compute_avg_stdev('KGpipResults.xlsx', '30m_r1_3graphs_extra')
# print('*'*20)
# print('5 Graphs')
# compute_avg_stdev('KGpipResults.xlsx', '30m_r1_5graphs_extra')
# print('*'*20)
# print('7 Graphs')
# compute_avg_stdev('KGpipResults.xlsx', '30m_r1_extra')
# print('*'*20)
