# based on: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import statistics, matplotlib

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False) #added by ibrahim
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        ('Basecase', [
            [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00],
            [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
            [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
            [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
            [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]]),
        ('With CO', [
            [0.88, 0.02, 0.02, 0.02, 0.00, 0.05, 0.00, 0.05, 0.00],
            [0.08, 0.94, 0.04, 0.02, 0.00, 0.01, 0.12, 0.04, 0.00],
            [0.01, 0.01, 0.79, 0.10, 0.00, 0.05, 0.00, 0.31, 0.00],
            [0.00, 0.02, 0.03, 0.38, 0.31, 0.31, 0.00, 0.59, 0.00],
            [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.00, 0.00]]),
        ('With O3', [
            [0.89, 0.01, 0.07, 0.00, 0.00, 0.05, 0.00, 0.00, 0.03],
            [0.07, 0.95, 0.05, 0.04, 0.00, 0.02, 0.12, 0.00, 0.00],
            [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.00, 0.00, 0.00],
            [0.01, 0.03, 0.00, 0.32, 0.29, 0.27, 0.00, 0.00, 0.95],
            [0.02, 0.00, 0.03, 0.37, 0.56, 0.47, 0.87, 0.00, 0.00]]),
        ('CO & O3', [
            [0.87, 0.01, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.01],
            [0.09, 0.95, 0.02, 0.03, 0.00, 0.01, 0.13, 0.06, 0.00],
            [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.00, 0.50, 0.00],
            [0.01, 0.03, 0.00, 0.28, 0.24, 0.23, 0.00, 0.44, 0.88],
            [0.02, 0.00, 0.18, 0.45, 0.64, 0.55, 0.86, 0.00, 0.16]])
    ]
    return data

def get_task_data(task_name, Tasks, Datasets, FLAML_data,KGpipAutoSklearn_data, AutoSklearn_data, KGpipFLAML_data, AL_data = None):
    FLAML = []
    KGpipAutoSklearn = []
    AutoSklearn = []
    KGpipFLAML = []
    AL = []

    categories = []
    max_len_dataset = 7
    for i in range(0, len(Tasks)):
        if Tasks[i] == task_name:
            categories.append(str(Datasets[i])[:max_len_dataset])
            FLAML.append(FLAML_data[i])
            KGpipAutoSklearn.append(KGpipAutoSklearn_data[i])
            AutoSklearn.append(AutoSklearn_data[i])
            KGpipFLAML.append(KGpipFLAML_data[i])
            if AL_data:
                AL.append(AL_data[i])

    return categories, FLAML, KGpipAutoSklearn, AutoSklearn, KGpipFLAML, AL

def draw_all_coulmns(sheet_name):
    data = pd.read_excel(open('KGpipResults.xlsx', 'rb'), sheet_name=sheet_name)
    frame = pd.DataFrame(data)
    Tasks = frame.Task.tolist()
    Datasets = frame.Dataset.tolist()

    FLAML_data = frame.FLAML.tolist()
    KGpipFLAML_data = frame.KGpipFLAML.tolist()
    KGpipAutoSklearn_data = frame.KGpipAutoSklearn.tolist()
    AutoSklearn_data = frame.AutoSklearn.tolist()
    max_len_dataset = 5

    r_categories, r_FLAML, r_KGpipAutoSklearn, r_AutoSklearn, r_KGpipFLAML, AL = \
        get_task_data('regression', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data)
    bc_categories, bc_FLAML, bc_KGpipAutoSklearn, bc_AutoSklearn, bc_KGpipFLAML, AL = \
        get_task_data('binary-classification', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data)
    mc_categories, mc_FLAML, mc_KGpipAutoSklearn, mc_AutoSklearn, mc_KGpipFLAML, AL = \
        get_task_data('multi-classification', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data)

    print(f'FLAML: \n\tbinary classification (mean/stdev): {statistics.mean(bc_FLAML):.2f} ({statistics.stdev(bc_FLAML):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_FLAML):.2f} ({statistics.stdev(mc_FLAML):.2f})'
                 f'\n\tregression (mean/stdev): {statistics.mean(r_FLAML):.2f} ({statistics.stdev(r_FLAML):.2f})')

    print(f'KGPip+FLAML: \n\tbinary classification (mean/stdev): {statistics.mean(bc_KGpipFLAML):.2f} ({statistics.stdev(bc_KGpipFLAML):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_KGpipFLAML):.2f} ({statistics.stdev(mc_KGpipFLAML):.2f})'
                 f'\n\tregression (mean/stdev): {statistics.mean(r_KGpipFLAML):.2f} ({statistics.stdev(r_KGpipFLAML):.2f})')

    print(f'AutoSklearn: \n\tbinary classification (mean/stdev): {statistics.mean(bc_AutoSklearn):.2f} ({statistics.stdev(bc_AutoSklearn):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_AutoSklearn):.2f} ({statistics.stdev(mc_AutoSklearn):.2f})'
                 f'\n\tregression (mean/stdev): {statistics.mean(r_AutoSklearn):.2f} ({statistics.stdev(r_AutoSklearn):.2f})')

    print(f'KGPip+AutoSklearn: \n\tbinary classification (mean/stdev): {statistics.mean(bc_KGpipAutoSklearn):.2f} ({statistics.stdev(bc_KGpipAutoSklearn):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_KGpipAutoSklearn):.2f} ({statistics.stdev(mc_KGpipAutoSklearn):.2f})'
                 f'\n\tregression (mean/stdev): {statistics.mean(r_KGpipAutoSklearn):.2f} ({statistics.stdev(r_KGpipAutoSklearn):.2f})')

    data = [
        ('Binary Classification', [
            bc_categories, bc_FLAML, bc_KGpipAutoSklearn, bc_AutoSklearn, bc_KGpipFLAML
        ]),
        ('Multi-class Classification', [
            mc_categories, mc_FLAML, mc_KGpipAutoSklearn, mc_AutoSklearn, mc_KGpipFLAML
        ]),
        ('Regression', [
            r_categories, r_FLAML, r_KGpipAutoSklearn, r_AutoSklearn, r_KGpipFLAML
        ])

    ]
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 14}
    #
    # matplotlib.rc('font', **font)
    # plt.style.use('ggplot')
    N = 9
    theta = radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(figsize=(16, 6), nrows=1, ncols=3, subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.subplots_adjust(wspace=0.3, hspace=0., top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm']  # , 'y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        categories, FLAML, KGpipAutoSklearn, AutoSklearn, KGpipFLAML = case_data
        theta = radar_factory(len(categories), frame='polygon')
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data[1:], colors):
            ax.plot(theta, d, color=color)
            # ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(categories)
        # label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
        # lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)

    # add legend relative to top-left plot
    labels = ('FLAML', 'KGpipAutoSklearn', 'AutoSklearn', 'KGpipFLAML')
    legend = axs[0].legend(labels, loc=(0.8, 1.2),
                           labelspacing=0.1, fontsize='large', mode='extend', borderaxespad=0, ncol=8)

    # fig.text(0.5, 0.965, '1 Hour time limit',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    plt.show()
    fig.savefig(sheet_name.replace(' ', '_') + '.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)


def get_sheet_data(sheet_name):
    data = pd.read_excel(open('KGpipResults.xlsx', 'rb'), sheet_name=sheet_name)
    frame = pd.DataFrame(data)
    Tasks = frame.Task.tolist()
    Datasets = frame.Dataset.tolist()

    FLAML_data = frame.FLAML.tolist()
    KGpipFLAML_data = frame.KGpipFLAML.tolist()
    KGpipAutoSklearn_data = frame.KGpipAutoSklearn.tolist()
    AutoSklearn_data = frame.AutoSklearn.tolist()
    max_len_dataset = 5

    r_categories, r_FLAML, r_KGpipAutoSklearn, r_AutoSklearn, r_KGpipFLAML = \
        get_task_data('regression', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data)
    bc_categories, bc_FLAML, bc_KGpipAutoSklearn, bc_AutoSklearn, bc_KGpipFLAML = \
        get_task_data('binary-classification', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data)
    mc_categories, mc_FLAML, mc_KGpipAutoSklearn, mc_AutoSklearn, mc_KGpipFLAML = \
        get_task_data('multi-classification', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data)
    return r_categories, r_FLAML, r_KGpipAutoSklearn, r_AutoSklearn, r_KGpipFLAML, \
           bc_categories, bc_FLAML, bc_KGpipAutoSklearn, bc_AutoSklearn, bc_KGpipFLAML, \
           mc_categories, mc_FLAML, mc_KGpipAutoSklearn, mc_AutoSklearn, mc_KGpipFLAML

def draw_all_coulmns_with_AL(sheet_name):
    data = pd.read_excel(open('KGpipResults.xlsx', 'rb'), sheet_name=sheet_name)
    frame = pd.DataFrame(data)
    Tasks = frame.Task.tolist()
    Datasets = frame.Dataset.tolist()

    FLAML_data = frame.FLAML.tolist()
    KGpipFLAML_data = frame.KGpipFLAML.tolist()
    KGpipAutoSklearn_data = frame.KGpipAutoSklearn.tolist()
    AutoSklearn_data = frame.AutoSklearn.tolist()
    max_len_dataset = 5
    AL_data = frame.AL.tolist()

    r_categories, r_FLAML, r_KGpipAutoSklearn, r_AutoSklearn, r_KGpipFLAML, r_AL = \
        get_task_data('regression', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data, AL_data)
    bc_categories, bc_FLAML, bc_KGpipAutoSklearn, bc_AutoSklearn, bc_KGpipFLAML, bc_AL = \
        get_task_data('binary-classification', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data, AL_data)
    mc_categories, mc_FLAML, mc_KGpipAutoSklearn, mc_AutoSklearn, mc_KGpipFLAML, mc_AL = \
        get_task_data('multi-classification', Tasks, Datasets, FLAML_data, KGpipAutoSklearn_data, AutoSklearn_data,
                      KGpipFLAML_data, AL_data)

    print(f'FLAML: \n\tbinary classification (mean/stdev): {statistics.mean(bc_FLAML):.2f} ({statistics.stdev(bc_FLAML):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_FLAML):.2f} ({statistics.stdev(mc_FLAML):.2f})')

    print(f'KGPip+FLAML: \n\tbinary classification (mean/stdev): {statistics.mean(bc_KGpipFLAML):.2f} ({statistics.stdev(bc_KGpipFLAML):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_KGpipFLAML):.2f} ({statistics.stdev(mc_KGpipFLAML):.2f})')

    print(f'AutoSklearn: \n\tbinary classification (mean/stdev): {statistics.mean(bc_AutoSklearn):.2f} ({statistics.stdev(bc_AutoSklearn):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_AutoSklearn):.2f} ({statistics.stdev(mc_AutoSklearn):.2f})')

    print(f'KGPip+AutoSklearn: \n\tbinary classification (mean/stdev): {statistics.mean(bc_KGpipAutoSklearn):.2f} ({statistics.stdev(bc_KGpipAutoSklearn):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_KGpipAutoSklearn):.2f} ({statistics.stdev(mc_KGpipAutoSklearn):.2f})')

    print(f'AL: \n\tbinary classification (mean/stdev): {statistics.mean(bc_AL):.2f} ({statistics.stdev(bc_AL):.2f})'
                 f'\n\tmulti-class classification (mean/stdev): {statistics.mean(mc_AL):.2f} ({statistics.stdev(mc_AL):.2f})')


    data = [
        ('Binary Classification', [
            bc_categories, bc_FLAML, bc_KGpipAutoSklearn, bc_AutoSklearn, bc_KGpipFLAML, bc_AL
        ]),
        ('Multi-class Classification', [
            mc_categories, mc_FLAML, mc_KGpipAutoSklearn, mc_AutoSklearn, mc_KGpipFLAML, mc_AL
        ]),
        # ('Regression', [
        #     r_categories, r_FLAML, r_KGpipAutoSklearn, r_AutoSklearn, r_KGpipFLAML, r_AL
        # ])

    ]
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 14}
    #
    # matplotlib.rc('font', **font)
    # plt.style.use('ggplot')
    N = 9
    theta = radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(figsize=(10, 7), nrows=1, ncols=2, subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.subplots_adjust(wspace=0.3, hspace=0., top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm' , 'k']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        # categories, FLAML, KGpipAutoSklearn, AutoSklearn, KGpipFLAML = case_data
        theta = radar_factory(len(case_data[0]), frame='polygon')
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data[1:], colors):
            ax.plot(theta, d, color=color)
            # ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(case_data[0])
        # label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
        # lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)

    # add legend relative to top-left plot
    labels = ('FLAML', 'KGpipAutoSklearn', 'AutoSklearn', 'KGpipFLAML', 'AL')
    legend = axs[0].legend(labels, loc=(0.0, 1.2),
                           labelspacing=0.1, fontsize='large', mode='extend', borderaxespad=0, ncol=8)

    # fig.text(0.5, 0.965, '1 Hour time limit',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    plt.show()
    fig.savefig(sheet_name.replace(' ', '_') + '.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)



def draw_variable_graphs_fig(sysname):
    # draw_variable_graphs_fig(sheet_name= '30m_r1_5graphs')
    # draw_variable_graphs_fig(sheet_name = '30m_r1_3graphs1')
    # draw_variable_graphs_fig(sheet_name='avg. 30m')


    g3_r_categories, _, g3_r_KGpipAutoSklearn, _, g3_r_KGpipFLAML, \
    g3_bc_categories, _, g3_bc_KGpipAutoSklearn, _, g3_bc_KGpipFLAML, \
    g3_mc_categories, _, g3_mc_KGpipAutoSklearn, _, g3_mc_KGpipFLAML = get_sheet_data('30m_r1_3graphs_extra')

    g5_r_categories, _, g5_r_KGpipAutoSklearn, _, g5_r_KGpipFLAML, \
    g5_bc_categories, _, g5_bc_KGpipAutoSklearn, _, g5_bc_KGpipFLAML, \
    g5_mc_categories, _, g5_mc_KGpipAutoSklearn, _, g5_mc_KGpipFLAML = get_sheet_data('30m_r1_5graphs_extra')

    g7_r_categories, _, g7_r_KGpipAutoSklearn, _, g7_r_KGpipFLAML, \
    g7_bc_categories, _, g7_bc_KGpipAutoSklearn, _, g7_bc_KGpipFLAML, \
    g7_mc_categories, _, g7_mc_KGpipAutoSklearn, _, g7_mc_KGpipFLAML = get_sheet_data('30m_r1_extra')

    if sysname == 'KGpipFLAML':
        data = [
            ('Binary Classification', [
                g5_bc_categories, g3_bc_KGpipFLAML, g5_bc_KGpipFLAML, g7_bc_KGpipFLAML
            ]),
            ('Multi-class Classification', [
                g5_mc_categories, g3_mc_KGpipFLAML, g5_mc_KGpipFLAML, g7_mc_KGpipFLAML
            ]),
            ('Regression', [  # categories are fixed
                g5_r_categories, g3_r_KGpipFLAML, g5_r_KGpipFLAML, g7_r_KGpipFLAML
            ]),
        ]
        labels = ('KGpipFLAML (3 graphs)', 'KGpipFLAML (5 graphs)', 'KGpipFLAML (7 graphs)')
    elif sysname == 'KGpipAutoSklearn':
        data = [
            ('Binary Classification', [
                g5_bc_categories, g3_bc_KGpipAutoSklearn, g5_bc_KGpipAutoSklearn, g7_bc_KGpipAutoSklearn
            ]),
            ('Multi-class Classification', [
                g5_mc_categories, g3_mc_KGpipAutoSklearn, g5_mc_KGpipAutoSklearn, g7_mc_KGpipAutoSklearn
            ]),
            ('Regression', [  # categories are fixed
                g5_r_categories, g3_r_KGpipAutoSklearn, g5_r_KGpipAutoSklearn, g7_r_KGpipAutoSklearn
            ]),
        ]
        labels = ('KGpipAutoSklearn (3 graphs)', 'KGpipAutoSklearn (5 graphs)', 'KGpipAutoSklearn (7 graphs)')

    # plt.style.use('ggplot')
    N = 9
    theta = radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(figsize=(16, 6), nrows=1, ncols=3, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85, bottom=0.05)

    colors = ['k', 'r', 'm']  # , 'y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        categories = case_data[0]
        theta = radar_factory(len(categories), frame='polygon')
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data[1:], colors):
            ax.plot(theta, d, color=color)
            # ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(categories)
        # label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
        # lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)

    # add legend relative to top-left plot
    # legend = axs[0].legend(labels, loc=(0.2, 1.2), labelspacing=0.1, fontsize='large', mode='extend', borderaxespad=0, ncol=8)
    legend = axs[0].legend(labels, loc=(0.45, 1.2), labelspacing=0.1, fontsize='large', mode='extend', borderaxespad=0, ncol=8)


    # fig.text(0.5, 0.965, '1 Hour time limit',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    plt.show()
    fig.savefig(sysname + '_var_graphs_30m.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)



if __name__ == '__main__':
    # draw_all_coulmns(sheet_name = 'avg. 1h_updated_extra')
    # draw_all_coulmns(sheet_name= 'avg. 30m')

    # draw_variable_graphs_fig(sysname= 'KGpipFLAML')
    # draw_variable_graphs_fig(sysname= 'KGpipAutoSklearn')

    draw_all_coulmns_with_AL(sheet_name = 'avg. 1h_updated_AL_only')
