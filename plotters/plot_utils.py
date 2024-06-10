import time
import plotly.io as pio
import os
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np

from environments.aircraftenv import printPurple
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.font_manager as font_manager

fontpath = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf'
prop = font_manager.FontProperties(fname=fontpath)
mplt.rcParams['font.family'] = prop.get_name()
# %matplotlib inline
plt.rcParams['figure.figsize'] = [15, 12]

plt.rcParams['lines.markersize'] = 4


def filter_outliers(data, t):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filter_idx = ((data > lower_bound) & (data < upper_bound))
    return t[filter_idx], data[filter_idx]


def plot(data, name, fault, switch_time=None, filter_action: bool = False, fig_name: str = None):
    """Evaluation Results on the fault

    Args:
        data (array timex20): [ref_values, actions, x_lst, rewards]
        name (str): name of the figure:
        fault (str): fault name
    Returns:
        fig (figure): figure object
        rewards (array): rewards #TODO: replace with rewards fig plot
    """

    print(data.shape)
    n = data.shape[1]
    n_ref = 2 if n < 20 else 3
    ref_values = data[:, :n_ref]  # ref_values [theta, phi, beta] unit: rad
    actions = data[:, n_ref:n_ref+3]  # actions [de, da, dr] unit: rad
    # x_lst [p, q, r, V, alpha, beta, phi, theta, psi, h] unit: rad, rad/s, m, m/s
    x_lst = data[:, n_ref+3:n_ref+13]
    rewards = data[:, -2]

    # ************* PARSING THE DATA *************
    # TODO: Convert from read to deg
    _t = data[:, -1]
    if n_ref == 3:
        theta_ref, phi_ref, psi_ref = ref_values[:, 0], ref_values[:, 1], \
            ref_values[:, 2]
    else:
        theta_ref, phi_ref, psi_ref = ref_values[:, 0], ref_values[:, 1], np.zeros(
            len(ref_values[:, 0]))

    p, q, r, alpha, beta, phi, theta, psi = x_lst[:, 0], x_lst[:, 1], \
        x_lst[:, 2], x_lst[:, 4], x_lst[:, 5], x_lst[:, 6], x_lst[:, 7], \
        x_lst[:, 8]

    V, h = x_lst[:, 3], x_lst[:, 9]

    de, da, dr = actions[:, 0], actions[:, 1], actions[:, 2]
    # ******* convert the rad values to degrees:

    theta_ref = np.rad2deg(theta_ref)
    phi_ref = np.rad2deg(phi_ref)
    psi_ref = np.rad2deg(psi_ref)
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    psi = np.rad2deg(psi)
    p = np.rad2deg(p)
    q = np.rad2deg(q)
    r = np.rad2deg(r)
    alpha = np.rad2deg(alpha)
    beta = np.rad2deg(beta)
    de = np.rad2deg(de)
    da = np.rad2deg(da)
    dr = np.rad2deg(dr)

    # filter for outliers in action if asked to:
    if filter_action:
        t_de, de_filt = filter_outliers(de, _t)
        t_da, da_filt = filter_outliers(da, _t)
        t_dr, dr_filt = filter_outliers(dr, _t)

    # *****************************************************
    # ********* Checking Faults **************
    # TODO: do something: saturation limits to plot
    # check each fault:
    # plot the saturation limits accordingly
    # *********  Plotting  ****************
    fig, axis = plt.subplots(6, 2)
    labels = ['Tracked State', 'Reference',
              'Actuator Deflection', 'Tracked State Rate', 'Other States']
    tracked_state_color = 'magenta'  # magenta
    ref_state_color = 'black'
    state_rate_color = 'blue'  # cyan
    action_color = 'green'
    other_color = 'purple'
    fontsize = 14

    l1 = axis[0, 0].plot(_t, theta, color=tracked_state_color, label=labels[0])

    l2 = axis[0, 0].plot(_t, theta_ref, color=ref_state_color,
                         linestyle='--', label=labels[1])
    axis[0, 0].set_ylabel(r'$\theta \:\: [{deg}]$', fontsize=fontsize)
    axis[0, 0].grid()

    # filter the elevator deflection de for outliers so that the plot is visible

    if filter_action:
        l3 = axis[0, 1].plot(
            t_de, de_filt, color=action_color, label=labels[2])
    else:
        l3 = axis[0, 1].plot(_t, de, color=action_color, label=labels[2])
    axis[0, 1].set_ylabel(r'$\delta_e \:\: [{deg}]$', fontsize=fontsize)
    axis[0, 1].grid()

    axis[1, 0].plot(_t, phi, color=tracked_state_color)
    axis[1, 0].plot(_t, phi_ref, linestyle='--', color=ref_state_color)
    axis[1, 0].set_ylabel(r'$\phi \:\: [{deg}]$', fontsize=fontsize)
    axis[1, 0].grid()

    if filter_action:
        axis[1, 1].plot(t_da, da_filt, color=action_color)
    else:
        axis[1, 1].plot(_t, da, color=action_color)
    axis[1, 1].set_ylabel(r'$\delta_a \:\: [{deg}]$', fontsize=fontsize)
    axis[1, 1].grid()

    # TODO: replace psi with beta
    axis[2, 0].plot(_t, beta, color=tracked_state_color)
    axis[2, 0].plot(_t, psi_ref, linestyle='--', color=ref_state_color)
    axis[2, 0].set_ylabel(r'$\beta \:\: [{deg}]$', fontsize=fontsize)
    axis[2, 0].grid()

    if filter_action:
        axis[2, 1].plot(t_dr, dr_filt, color=action_color)
    else:
        axis[2, 1].plot(_t, dr, color=action_color)
    axis[2, 1].set_ylabel(r'$\delta_r \:\: [{deg}]$', fontsize=fontsize)
    axis[2, 1].grid()

    l4 = axis[3, 0].plot(_t, q, color=state_rate_color, label=labels[3])
    axis[3, 0].set_ylabel(r'$q \:\: [{deg/s}]$', fontsize=fontsize)
    axis[3, 0].grid()

    l5 = axis[3, 1].plot(_t, alpha, color=other_color, label=labels[4])
    axis[3, 1].set_ylabel(r'$\alpha \:\: [{deg}]$', fontsize=fontsize)
    axis[3, 1].grid()

    axis[4, 0].plot(_t, p, color=state_rate_color)
    axis[4, 0].set_ylabel(r'$p \:\: [{deg/s}]$', fontsize=fontsize)
    axis[4, 0].grid()

    axis[4, 1].plot(_t, r, color=state_rate_color)
    axis[4, 1].set_ylabel(r'$r \:\: [{deg/s}]$', fontsize=fontsize)
    axis[4, 1].grid()

    axis[5, 0].plot(_t, V, color=other_color)
    axis[5, 0].set_ylabel(r'$V \:\: [m/s]$', fontsize=fontsize)
    axis[5, 0].grid()
    axis[5, 0].set_xlabel('Time [S]', fontsize=fontsize)
    axis[5, 1].plot(_t, h, color=other_color)
    axis[5, 1].set_ylabel(r'$h \:\: [m]$', fontsize=fontsize)
    axis[5, 1].grid()
    axis[5, 1].set_xlabel('Time [S]', fontsize=fontsize)

    # legend:
    leg_lines = []
    leg_labels = []
    for ax in fig.axes:
        axLine, axLegend = ax.get_legend_handles_labels()
        leg_lines.extend(axLine)
        leg_labels.extend(axLegend)

    # add a vertical line to all axis at the switch time
    if switch_time is not None:
        for ax in axis.flatten():
            ax.axvline(x=switch_time, color='grey',
                       linestyle='--', linewidth=3.5)

    fig.suptitle(name, fontsize=fontsize)
    fig.legend(leg_lines, leg_labels, loc='upper center',
               ncol=5, mode='expand', bbox_to_anchor=(0.11, 0.68, 0.8, 0.25), fontsize=13)
    # fig.legend([l1, l2, l3, l4, l5], labels=labels, loc='upper center',
    #            ncol=5, mode='expand', bbox_to_anchor=(0.11, 0.68, 0.8, 0.25), fontsize=13)
    # plt.legend()
    # plt.show()
    # ***************************************
    if fig_name is not None:
        # figures_filtered_actions; figures_different_input, #figures_stability, figures; figures_filtered_actions_withoutBeta
        fig_path = Path(os.getcwd()) / \
            Path('figures_withoutSuptitle')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        figname = fig_path / Path(fig_name + '.pdf')
        # fig.write_image(str(figname))
        fig.savefig(figname, dpi=300, bbox_inches='tight', format='pdf')
    return fig, rewards



def plot_attitude_response(data=None, name=None, fault=None, nmae=None, sm=None,  switch_time=None):
    if '.pkl' in name:
        name, _ = name.split('.')

    print(name)
    # print(data.shape)
    ref_values = data[:, :3]  # ref_values [theta, phi, beta] unit: rad
    actions = data[:, 3:6]  # actions [de, da, dr] unit: rad
    # x_lst [p, q, r, V, alpha, beta, phi, theta, psi, h] unit: rad, rad/s, m, m/s
    x_lst = data[:, 6:16]
    rewards = data[:, -2]

    # ************* PARSING THE DATA *************
    # TODO: Convert from read to deg
    _t = data[:, -1]
    theta_ref, phi_ref, psi_ref = ref_values[:, 0], ref_values[:, 1], \
        ref_values[:, 2]

    p, q, r, V, alpha, beta, phi, theta, psi, h = x_lst[:, 0], x_lst[:, 1], \
        x_lst[:, 2], x_lst[:, 3], x_lst[:, 4], x_lst[:, 5], x_lst[:, 6], x_lst[:, 7], \
        x_lst[:, 8], x_lst[:, 9]

    de, da, dr = actions[:, 0], actions[:, 1], actions[:, 2]
    # ******* convert the rad values to degrees:

    theta_ref = np.rad2deg(theta_ref)
    phi_ref = np.rad2deg(phi_ref)
    psi_ref = np.rad2deg(psi_ref)
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    psi = np.rad2deg(psi)
    p = np.rad2deg(p)
    q = np.rad2deg(q)
    r = np.rad2deg(r)
    alpha = np.rad2deg(alpha)
    beta = np.rad2deg(beta)
    # print(beta)
    de = np.rad2deg(de)
    da = np.rad2deg(da)
    dr = np.rad2deg(dr)

    subplots_idx = {0: [1, 2], 1: [1, 1], 3: [2, 2], 4: [2, 1], 5:   [4, 2], 6: [
        3, 2], 7: [3, 1], 8: [7, 1], 9: [5, 1], 10: [7, 2], 11: [7, 2]}
    fig = make_subplots(
        rows=6, cols=2, vertical_spacing=0.2/6, horizontal_spacing=0.17/2
    )

    # ******* theta *****************
    max_theta = np.round(np.max(np.abs(theta))+2, 0)
    interv = (max_theta)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=theta_ref, line=dict(color='#EF553B', dash='dashdot')
        ), row=1, col=1
    )
    fig.append_trace(
        go.Scatter(
            x=_t, y=theta, name=r'$\theta [^\circ]$', line=dict(color='#636EFA')
        ), row=1, col=1
    )
    fig.update_yaxes(title_text=r'$\theta \:\: [{deg}]$', row=1, col=1, title_standoff=14, tickmode='array', tickvals=np.arange(
        -max_theta, max_theta+interv, interv), ticktext=[str(-max_theta), ' ', str(np.round(-max_theta+2*interv, 0)), ' ', str(np.round(-max_theta+4*interv, 0))], tickfont=dict(size=11), range=[-max_theta, max_theta], titlefont=dict(size=13))

    # ******* phi *****************
    max_phi = np.round(np.max(np.abs(phi))+2, 0)
    interv = (max_phi)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=phi_ref, line=dict(color='#EF553B', dash='dashdot')
        ), row=2, col=1
    )
    fig.append_trace(
        go.Scatter(
            x=_t, y=phi, name=r'$\phi [^\circ]$', line=dict(color='#636EFA')
        ), row=2, col=1
    )
    fig.update_yaxes(title_text=r'$\phi \:\:  [{deg}]$', row=2, col=1, title_standoff=14, tickmode='array', tickvals=np.arange(
        -max_phi, max_phi+interv, interv), ticktext=[str(-max_phi), ' ', str(np.round(-max_phi+2*interv, 0)), ' ', str(np.round(-max_phi+4*interv, 0))], tickfont=dict(size=11), range=[-max_phi, max_phi], titlefont=dict(size=13))

    # ******* psi or beta *****************
    max_beta = np.round(np.max(np.abs(beta))+0.05, 1)

    interv = (max_beta)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=psi_ref, line=dict(color='#EF553B', dash='dashdot')
        ), row=3, col=1
    )
    fig.append_trace(
        go.Scatter(
            x=_t, y=beta, name=r'$\beta [^\circ]$', line=dict(color='#636EFA')
        ), row=3, col=1
    )
    fig.update_yaxes(title_text=r'$\beta \:\: [{deg}]$', row=3, col=1, title_standoff=14, tickmode='array', tickvals=np.arange(
        -max_beta, max_beta+interv, interv), ticktext=[str(-max_beta), ' ', '0', ' ', str(max_beta)], tickfont=dict(size=11), range=[-max_beta, max_beta], titlefont=dict(size=13))

    # TODO: remove outliers for fault cases to view properly the actions and rate

    # ******** delta elevator *****************
    max_de = np.round(np.max(np.abs(de[100:]))+0.2, 2)
    interv = (max_de)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=de, name=r'$\delta_e [^\circ]$', line=dict(color='#00CC96')
        ), row=1, col=2
    )
    fig.update_yaxes(title_text=r'$\delta_e \:\: [{deg}]$', row=1, col=2, title_standoff=8, tickmode='array', tickvals=np.arange(
        -max_de, max_de+interv, interv), ticktext=[str(-max_de), ' ', str(np.round(-max_de+2*interv, 1)), ' ', str(np.round(-max_de+4*interv, 1))], tickfont=dict(size=11), range=[-max_de, max_de], titlefont=dict(size=13))

    # ******** delta aileron *****************
    max_da = np.round(np.max(np.abs(da[100:]))+0.1, 2)
    interv = (max_da)/2
    fig.append_trace(
        go.Scatter(
            x=_t, y=da, name=r'$\delta_a [^\circ]$', line=dict(color='#00CC96')
        ), row=2, col=2
    )
    fig.update_yaxes(title_text=r'$\delta_a \:\: [{deg}]$', row=2, col=2, title_standoff=8, tickmode='array', tickvals=np.arange(
        -max_da, max_da+interv, interv), ticktext=[str(-max_da), ' ', str(np.round(-max_da+2*interv, 1)), ' ', str(np.round(-max_da+4*interv, 1))], tickfont=dict(size=11), range=[-max_da, max_da], titlefont=dict(size=13))

    # ******** delta rudder *****************
    max_dr = np.round(np.max(np.abs(dr[100:]))+0.2, 1)
    interv = (max_dr)/2
    fig.append_trace(
        go.Scatter(
            x=_t, y=dr, name=r'$\delta_r [^\circ]$', line=dict(color='#00CC96')
        ), row=3, col=2
    )
    fig.update_yaxes(title_text=r'$\delta_r \:\: [{deg}]$', row=3, col=2, title_standoff=8, tickmode='array', tickvals=np.arange(
        -max_dr, max_dr+interv, interv), ticktext=[str(-max_dr), ' ', str(np.round(-max_dr+2*interv, 1)), ' ', str(np.round(-max_dr+4*interv, 1))], tickfont=dict(size=11), range=[-max_dr, max_dr], titlefont=dict(size=13))

    # ********** p [deg/s] ***************
    max_p = np.round(np.max(np.abs(p))+0.2, 0)
    interv = (max_p)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=p, name=r'$p [^\circ/s]$', line=dict(color='#AB63FA')
        ), row=4, col=1
    )
    fig.update_yaxes(title_text=r'$p \:\: [{deg/s}]$', row=4, col=1, title_standoff=13, tickmode='array', tickvals=np.arange(
        -max_p, max_p+interv, interv), ticktext=[str(-max_p), ' ', str(np.round(-max_p+2*interv, 1)), ' ', str(np.round(-max_p+4*interv, 1))],
        tickfont=dict(size=11), range=[-max_p, max_p], titlefont=dict(size=13))

    # ********* q [deg/s] *****************
    max_q = np.round(np.max(np.abs(q))+0.5, 0)
    interv = (max_q)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=q, name=r'$q [^\circ/s]$', line=dict(color='#AB63FA')
        ), row=4, col=2
    )
    fig.update_yaxes(title_text=r'$q \:\: [{deg/s}]$', row=4, col=2, title_standoff=8, tickmode='array', tickvals=np.arange(
        -max_q, max_q+interv, interv), ticktext=[str(-max_q), ' ', str(np.round(-max_q+2*interv, 1)), ' ', str(np.round(-max_q+4*interv, 1))],
        tickfont=dict(size=11), range=[-max_q, max_q], titlefont=dict(size=13))

    # ******** r [deg/s]  ***************
    max_r = np.round(np.max(np.abs(r))+0.5, 0)
    interv = (max_r)/2.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=r, name=r'$r [^\circ/s]$', line=dict(color='#AB63FA')
        ), row=5, col=1
    )
    fig.update_yaxes(title_text=r'$r \:\: [{deg/s}]$', row=5, col=1, title_standoff=13, tickmode='array', tickvals=np.arange(
        -max_r, max_r+interv, interv), ticktext=[str(-max_r), ' ', str(np.round(-max_r+2*interv, 1)), ' ', str(np.round(-max_r+4*interv, 1))],
        tickfont=dict(size=11), range=[-max_r, max_r], titlefont=dict(size=13))

    # ******** alpha [deg] ***************
    max_alpha = int(np.round(np.max(np.abs(alpha))+1.0, 0))
    min_alpha = int(np.round(np.min(np.abs(alpha))-0.5, 0))
    interv = (max_alpha-min_alpha)/4.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=alpha, name=r'$\alpha [^\circ]$', line=dict(color='#AB63FA')
        ), row=5, col=2
    )
    fig.update_yaxes(title_text=r'$\alpha \:\: [{deg}]$', row=5, col=2, title_standoff=8, tickmode='array', tickvals=np.arange(
        min_alpha, max_alpha+interv, interv), ticktext=[str(min_alpha), ' ', str(int(np.round(min_alpha+2*interv, 1))), ' ', str(int(np.round(min_alpha+4*interv, 1)))],
        tickfont=dict(size=11), range=[min_alpha, max_alpha], titlefont=dict(size=13))

    # ******** V [m/s] ***************
    max_V = np.round(np.max(np.abs(V))+5, 0)
    min_V = np.round(np.min(np.abs(V))-5, 0)
    interv = (max_V-min_V)/4.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=V, name=r'$V [m/s]$', line=dict(color='#AB63FA')
        ), row=6, col=1
    )
    fig.update_yaxes(title_text=r'$V \:\: [{m/s}]$', row=6, col=1, title_standoff=13, tickmode='array', tickvals=np.arange(
        min_V, max_V+interv, interv), ticktext=[str(int(min_V)), ' ',  str(int(np.round(min_V+2*interv, 0))), ' ', str(int(np.round(min_V+4*interv, 0)))],
        tickfont=dict(size=11), range=[min_V, max_V], titlefont=dict(size=13))

    # ******** h [m] ***************
    max_h = np.round(np.max(np.abs(h))+20, 0)
    min_h = np.round(np.min(np.abs(h))-20, 0)
    interv = (max_h-min_h)/4.0
    fig.append_trace(
        go.Scatter(
            x=_t, y=h, name=r'$h [m]$', line=dict(color='#AB63FA')
        ), row=6, col=2
    )
    fig.update_yaxes(title_text=r'$h \:\: [{m}]$', row=6, col=2, title_standoff=8, tickmode='array', tickvals=np.arange(
        min_h, max_h+interv, interv), ticktext=[str(int(min_h)), ' ', str(int(np.round(min_h+2*interv, 0))), ' ', str(int(np.round(min_h+4*interv, 0)))],
        tickfont=dict(size=11), range=[min_h, max_h], titlefont=dict(size=13))

    # *************************************************
    # if fault != 'nominal':
    #     fig.add_vline(x=5, row='all', col='all',
    #                   line=dict(color="Grey", width=1.5))
    if fault != 'nominal' and switch_time is not None:
        fig.add_vline(x=switch_time, row='all', col='all',
                      line=dict(color="Grey", width=1.5, dash='dot'))

    fig.update_layout(showlegend=False, width=800, height=480, margin=dict(
        l=10, r=2, b=5, t=0,
    ))
    fig.layout.font.family = 'Arial'

    tick_interval = 10
    end_time = _t[-1] + 0.02
    print(end_time)
    fig.update_xaxes(title_text=r'$t \:\: {[s]}$', row=6, col=1, range=[0, end_time], tickmode='array', tickvals=np.arange(
        0, end_time, tick_interval), tickfont=dict(size=11), titlefont=dict(size=13), title_standoff=11)
    fig.update_xaxes(title_text=r'$t \:\: {[s]}$', row=6, col=2, range=[0, end_time], tickmode='array', tickvals=np.arange(
        0, end_time, tick_interval), tickfont=dict(size=11), titlefont=dict(size=13), title_standoff=11)

    for row in range(6):
        for col in range(2):
            fig.update_xaxes(showticklabels=False, tickmode='array', tickvals=np.arange(
                0, end_time, tick_interval), row=row, col=col)
    fig.update_traces(mode='lines')
    fig.show()

    # TODO: saving the figure
    # fig_path = Path(os.getcwd()) / Path('figures')
    # if not os.path.exists(fig_path):
    #     os.makedirs(fig_path)
    # figname = fig_path / Path(f'{name}_{fault}__nmae{nmae}_sm{-sm}.pdf')
    # fig.write_image(str(figname))
    # time.sleep(2)

    # fig.write_image(str(figname))

    return


def plot_comparative_analysis(data, faultname, fig_name=None):

    fontpath = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    mplt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['figure.figsize'] = [15, 12]

    plt.rcParams['lines.markersize'] = 4

    fontsize = 14
    labels = ['Tracked States', 'Reference', 'Actuator Deflection']
    ref_values = data[0, :, :3]  # theta, phi, beta: radians
    actions1 = data[0, :, 3:6]  # [de, da, dr] for env 1
    actions2 = data[1, :, 3:6]  # [de, da, dr] for env 2

    # [p, q, r, V, alpha, beta, phi, theta, psi, h] unit: rad, rad/s, m, m/s
    x_lst1 = data[0, :, 6:16]
    x_lst2 = data[1, :, 6:16]
    _t = data[0, :, -1]  # time

    theta_ref, phi_ref, psi_ref = ref_values[:, 0], ref_values[:, 1], \
        ref_values[:, 2]
    theta_ref = np.rad2deg(theta_ref)
    phi_ref = np.rad2deg(phi_ref)
    psi_ref = np.rad2deg(psi_ref)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(_t, theta_ref, linestyle='--',
                   color='black', label=labels[1], linewidth=3)
    axs[0, 0].set_ylabel(r'$\theta \:\: [{deg}]$', fontsize=fontsize)
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].plot(_t, phi_ref, linestyle='--', color='black', linewidth=3)
    axs[1, 0].set_ylabel(r'$\phi \:\: [{deg}]$', fontsize=fontsize)
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[2, 0].plot(_t, psi_ref, linestyle='--', color='black',
                   linewidth=3)
    axs[2, 0].set_ylabel(r'$\beta \:\: [{deg}]$', fontsize=fontsize)
    axs[2, 0].grid()
    axs[2, 1].grid()

    tracked_state_color = 'magenta'
    action_color = 'green'
    # i = 0
    line_styles = ['-', 'dotted']
    x_lsts = [x_lst1, x_lst2]
    actions = [actions1, actions2]
    for i in range(len(line_styles)):
        p, q, r, V, alpha, beta, phi, theta, psi, h = x_lsts[i][:, 0], x_lsts[i][:, 1], \
            x_lsts[i][:, 2], x_lsts[i][:, 3], x_lsts[i][:, 4], x_lsts[i][:, 5], x_lsts[i][:, 6], x_lsts[i][:, 7], \
            x_lsts[i][:, 8], x_lsts[i][:, 9]

        de, da, dr = actions[i][:, 0], actions[i][:, 1], actions[i][:, 2]
        de = np.rad2deg(de)
        da = np.rad2deg(da)
        dr = np.rad2deg(dr)
        t_de, de_filt = filter_outliers(de, _t)
        t_da, da_filt = filter_outliers(da, _t)
        t_dr, dr_filt = filter_outliers(dr, _t)

        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
        psi = np.rad2deg(psi)
        p = np.rad2deg(p)
        q = np.rad2deg(q)
        r = np.rad2deg(r)
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)

        axs[0, 0].plot(_t, theta,
                       linestyle=line_styles[i], color=tracked_state_color, label=labels[0], linewidth=2)
        axs[0, 1].plot(t_de, de_filt, label=labels[2], color=action_color,
                       linestyle=line_styles[i], linewidth=2)
        axs[0, 1].set_ylabel(r'$\delta_e \:\: [{deg}]$', fontsize=fontsize)

        axs[1, 0].plot(_t, phi,
                       linestyle=line_styles[i],
                       color=tracked_state_color, linewidth=2)
        axs[1, 1].plot(t_da, da_filt, color=action_color,
                       linestyle=line_styles[i], linewidth=2)
        axs[1, 1].set_ylabel(r'$\delta_a \:\: [{deg}]$', fontsize=fontsize)

        axs[2, 0].plot(_t, beta,
                       linestyle=line_styles[i], color=tracked_state_color, linewidth=2)
        axs[2, 1].plot(t_dr, dr_filt, color=action_color,
                       linestyle=line_styles[i], linewidth=2)
        axs[2, 1].set_ylabel(r'$\delta_r \:\: [{deg}]$', fontsize=fontsize)
        axs[2, 0].set_xlabel('Time [S]', fontsize=fontsize)
        axs[2, 1].set_xlabel('Time [S]', fontsize=fontsize)
    leg_lines = []
    leg_labels = []
    for ax in fig.axes:
        axLine, axLegend = ax.get_legend_handles_labels()
        leg_lines.extend(axLine)
        leg_labels.extend(axLegend)

    fig.legend(leg_lines, leg_labels, loc='upper center',
               ncol=5, mode='expand', bbox_to_anchor=(0.11, 0.68, 0.8, 0.25), fontsize=13)
    fig.suptitle(faultname, fontsize=fontsize)

    if fig_name is not None:
        # figures_filtered_actions; figures_different_input, #figures_stability, figures;figures_comparative_performance
        fig_path = Path(os.getcwd()) / Path('figures_withoutSuptitle')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        figname = fig_path / Path(fig_name + '.pdf')
        # fig.write_image(str(figname))
        fig.savefig(figname, dpi=300, bbox_inches='tight', format='pdf')
    # plt.show()
    return fig

if __name__ == "__main__":
    # testing the function
    pass
