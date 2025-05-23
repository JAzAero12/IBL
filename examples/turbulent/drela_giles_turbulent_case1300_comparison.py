"""
Comparing Head's method solution for accelerating flow case.

This example shows a comparison between Head's method and case 1300 from the
1968 Stanford Olympics from Luwieg and Tillman.
"""

# pylint: disable=too-many-statements,too-many-locals

# pylint: disable=duplicate-code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.reference import StanfordOlympics1968
from ibl.head_method import HeadMethod
from ibl.drela_giles_turbulent_mod import DrelaGilesTurbulentMOD

import os

# NACA 0009 Re = 10000
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['figure.figsize'] = [13, 13]
plt.rcParams['legend.loc'] = 'lower center'
plt.rcParams['legend.borderaxespad'] = -8.6
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 3.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.default"] = "rm"
#print(os.path.dirname(os.path.abspath(__file__)))

file_name = "CASE_1300"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)


def compare_case1300() -> None:
    """Compare the Head method results to expiremental data."""
    so68 = StanfordOlympics1968("1300")
    x = so68.x()
    u_e = so68.u_e()
    du_e = so68.du_e()
    x_sm = so68.x_smooth()
    u_e_sm = so68.u_e_smooth()
    du_e_sm = so68.du_e_smooth()
    rho = 1.2

    dg_reg = DrelaGilesTurbulentMOD(nu=so68.nu_ref, U_e=[x, u_e],old_ctau_ratio=True)
    dg_reg.initial_delta_m = so68.delta_m()[0]
    dg_reg.initial_shape_d = so68.shape_d()[0]

    rtn = dg_reg.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for DG method: " + rtn.message)
        print(rtn.x_end)
        print(so68.x()[-1])
        return

    #dg_sm = DrelaGilesTurbulentMOD(nu=so68.nu_ref, U_e=[x_sm, u_e_sm],old_ctau_ratio=True)
    #dg_sm.initial_delta_m = so68.delta_m()[0]
    #dg_sm.initial_shape_d = so68.shape_d()[0]
    #rtn = dg_sm.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    #if not rtn.success:
    #    print("Could not get solution for DG method: " + rtn.message)
    #    return

    #hm_sm2 = HeadMethod(nu=so68.nu_ref, U_e=u_e[0], dU_edx=[x, du_e])
    hm_sm2 = DrelaGilesTurbulentMOD(nu=so68.nu_ref, U_e=u_e[0], dU_edx=[x, du_e],old_ctau_ratio=True)
    hm_sm2.initial_delta_m = so68.delta_m()[0]
    hm_sm2.initial_shape_d = so68.shape_d()[0]
    rtn = hm_sm2.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for DG method: " + rtn.message)
        return

    # Calculate the boundary layer parameters
    x_ref = so68.x()
    delta_d_ref = so68.delta_d()
    delta_m_ref = so68.delta_m()
    shape_d_ref = so68.shape_d()
    c_f_ref = so68.c_f()
    u_e_ref = so68.u_e()

    x = np.linspace(x_ref[0], x_ref[-1], 101)

    delta_d_dg_reg = dg_reg.delta_d(x)
    delta_m_dg_reg = dg_reg.delta_m(x)
    shape_d_dg_reg = dg_reg.shape_d(x)
    c_f_dg_reg = 2*dg_reg.tau_w(x, rho)/(rho*dg_reg.u_e(x)**2)
    v_e_dg_reg = dg_reg.v_e(x)

    #delta_d_dg_sm = dg_sm.delta_d(x)
    #delta_m_dg_sm = dg_sm.delta_m(x)
    #shape_d_dg_sm = dg_sm.shape_d(x)
    #c_f_dg_sm = 2*dg_sm.tau_w(x, rho)/(rho*dg_sm.u_e(x)**2)
    #v_e_dg_sm = dg_sm.v_e(x)

    delta_d_head_sm2 = hm_sm2.delta_d(x)
    delta_m_head_sm2 = hm_sm2.delta_m(x)
    shape_d_head_sm2 = hm_sm2.shape_d(x)
    c_f_head_sm2 = 2*hm_sm2.tau_w(x, rho)/(rho*hm_sm2.u_e(x)**2)
    v_e_head_sm2 = hm_sm2.v_e(x)

    # Plot results
    fig = plt.figure()
    #fig.set_figwidth(10)
    #fig.set_figheight(15)
    #gs = GridSpec(6, 2, figure=fig)
    gs = GridSpec(5, 2, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_shape_d = fig.add_subplot(gs[2, 0])
    axis_shape_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_u_e = fig.add_subplot(gs[4, 0])
    axis_u_e_diff = fig.add_subplot(gs[4, 1])
    #axis_du_e = fig.add_subplot(gs[5, 0])
    #axis_v_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "L & T Data"
    dg_reg_color = "red"
    dg_reg_label = "D-G"
    dg_sm_color = "green"
    #dg_sm_label = r"D-G (Smooth $u_e$)"
    head_sm2_color = "orange"
    head_sm2_label = r"D-G (d$u_e/$d$s$)"

    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(x_ref, delta_d_ref, color=ref_color, linestyle="",
                        marker="o", label=ref_label)
    head_reg_curve = ax.plot(x, delta_d_dg_reg, color="#A4D65E",
                             label=dg_reg_label)
    #head_sm_curve = ax.plot(x, delta_d_dg_sm, color="#A4D65E",
    #                        label=dg_sm_label,linestyle='--')
    head_sm2_curve = ax.plot(x, delta_d_head_sm2, color="#A4D65E",
                             label=head_sm2_label,linestyle=':')
    _ = ax.set_ylim((0, 0.008))
    _ = ax.set_ylabel(r"$\delta_d$ [m]")
    #_ = ax.set_ylabel(r"$\delta^*$ (m)")
    ax.grid(True)

    ax = axis_delta_d_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.delta_d(x_ref)/delta_d_ref),
                color="#A4D65E")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.delta_d(x_ref)/delta_d_ref),
    #            color="#A4D65E",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.delta_d(x_ref)/delta_d_ref),
                color="#A4D65E",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    _ = ax.plot(x_ref, delta_m_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, delta_m_dg_reg, color="#BB00FF")
    #_ = ax.plot(x, delta_m_dg_sm, color="#3A913F",linestyle='--')
    _ = ax.plot(x, delta_m_head_sm2, color="#BB00FF",linestyle=':')
    _ = ax.set_ylim((0, 0.003))
    _ = ax.set_ylabel(r"$\delta_m$ [m]")
    #_ = ax.set_ylabel(r"$\theta$ (m)")
    ax.grid(True)

    ax = axis_delta_m_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.delta_m(x_ref)/delta_m_ref),
                color="#BB00FF")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.delta_m(x_ref)/delta_m_ref),
    #            color="#3A913F",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.delta_m(x_ref)/delta_m_ref),
                color="#BB00FF",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    _ = ax.plot(x_ref, shape_d_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, shape_d_dg_reg, color="#612D00")
    #_ = ax.plot(x, shape_d_dg_sm, color="#A4D65E",linestyle='--')
    _ = ax.plot(x, shape_d_head_sm2, color="#612D00",linestyle=':')
    _ = ax.set_ylim(1, 3)
    _ = ax.set_ylabel(r"$H_d$")
    #_ = ax.set_ylabel(r"$H$")
    ax.grid(True)

    ax = axis_shape_d_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.shape_d(x_ref)/shape_d_ref),
                color="#612D00")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.shape_d(x_ref)/shape_d_ref),
    #            color="#A4D65E",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.shape_d(x_ref)/shape_d_ref),
                color="#612D00",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    _ = ax.plot(x_ref, c_f_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, c_f_dg_reg, color="#818181")
    #_ = ax.plot(x, c_f_dg_sm, color="#F8E08E",linestyle='--')
    _ = ax.plot(x, c_f_head_sm2, color="#818181",linestyle=':')
    _ = ax.set_ylim((0, 0.005))
    _ = ax.set_ylabel(r"$c_f$")
    ax.grid(True)

    ax = axis_c_f_diff
    temp = 2*dg_reg.tau_w(x_ref, rho)/(rho*dg_reg.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color="#818181")
    #temp = 2*dg_sm.tau_w(x_ref, rho)/(rho*dg_sm.u_e(x_ref)**2)
    #_ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
    #            color="#F8E08E",linestyle='--')
    temp = 2*hm_sm2.tau_w(x_ref, rho)/(rho*hm_sm2.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color="#818181",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_u_e
    _ = ax.plot(x_sm, u_e_sm, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, dg_reg.u_e(x), color="#9FC9CD")
    #_ = ax.plot(x, dg_sm.u_e(x), color="#154734",linestyle='--')
    _ = ax.plot(x, hm_sm2.u_e(x), color="#9FC9CD",linestyle=':')
    _ = ax.set_ylim((10, 30))
    _ = ax.set_xlabel(r"$s$ [m]")
    _ = ax.set_ylabel(r"$u_e$ [m/s]")
    ax.grid(True)

    ax = axis_u_e_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.u_e(x_ref)/u_e_ref),
                color="#9FC9CD")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.u_e(x_ref)/u_e_ref),
    #            color="#154734",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.u_e(x_ref)/u_e_ref),
                color="#9FC9CD",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    _ = ax.set_xlabel(r"$s$ [m]")
    ax.set_yscale('log')
    ax.grid(True)

    # Transpiration velocity in 5,:
    # ax = axis_du_e
    # _ = ax.plot(x_sm, du_e_sm, color=ref_color, linestyle="", marker="o")
    # _ = ax.plot(x, dg_reg.du_e(x), color="#BD8B13")
    # #_ = ax.plot(x, dg_sm.du_e(x), color="#BD8B13",linestyle='--')
    # _ = ax.plot(x, hm_sm2.du_e(x), color="#BD8B13",linestyle=':')
    # _ = ax.set_ylim((3, 6))
    # _ = ax.set_xlabel(r"$s$ [m]")
    # _ = ax.set_ylabel(r"d$u_e/$d$s$ [m/s^2]")
    # ax.grid(True)

    # ax = axis_v_e
    # _ = ax.plot(x, v_e_dg_reg, color="#789F90")
    # #_ = ax.plot(x, v_e_dg_sm, color="#789F90",linestyle='--')
    # _ = ax.plot(x, v_e_head_sm2, color="#789F90",linestyle=':')
    # _ = ax.set_ylim((0, 0.05))
    # _ = ax.set_xlabel(r"$s$ [m]")
    # _ = ax.set_ylabel(r"$v_e$ [m/s]")
    # ax.grid(True)
    # _ = fig.subplots_adjust(left=0.15, wspace=0.5, hspace=.5, top=.95)
    # _ = fig.legend(handles=[ref_curve[0], head_reg_curve[0],
    #                         head_sm2_curve[0]], labels=[ref_label, dg_reg_label,
    #                        head_sm2_label], loc="upper center",
    #                bbox_to_anchor=(.495, 0.07), ncol=4, borderaxespad=.8)
    
    _ = fig.subplots_adjust(left=0.15, wspace=0.5, hspace=.5, top=.95, bottom=0.15)
    _ = fig.legend(handles=[ref_curve[0], head_reg_curve[0],
                            head_sm2_curve[0]], labels=[ref_label, dg_reg_label,
                           head_sm2_label], loc="upper center",
                   bbox_to_anchor=(.51, 0.06), ncol=4, borderaxespad=.2)
    fig.savefig(file_name+'\\'+'DG_CASE1300_Comp.png')
    

    so68 = StanfordOlympics1968("1300")
    x = so68.x()
    u_e = so68.u_e()
    du_e = so68.du_e()
    x_sm = so68.x_smooth()
    u_e_sm = so68.u_e_smooth()
    du_e_sm = so68.du_e_smooth()
    rho = 1.2

    dg_reg = HeadMethod(nu=so68.nu_ref, U_e=[x, u_e])
    dg_reg.initial_delta_m = so68.delta_m()[0]
    dg_reg.initial_shape_d = so68.shape_d()[0]

    rtn = dg_reg.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for DG method: " + rtn.message)
        print(rtn.x_end)
        print(so68.x()[-1])
        return

    #dg_sm = HeadMethod(nu=so68.nu_ref, U_e=[x_sm, u_e_sm])
    #dg_sm.initial_delta_m = so68.delta_m()[0]
    #dg_sm.initial_shape_d = so68.shape_d()[0]
    #rtn = dg_sm.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    #if not rtn.success:
    #    print("Could not get solution for DG method: " + rtn.message)
    #    return

    hm_sm2 = HeadMethod(nu=so68.nu_ref, U_e=u_e[0], dU_edx=[x, du_e])
    hm_sm2.initial_delta_m = so68.delta_m()[0]
    hm_sm2.initial_shape_d = so68.shape_d()[0]
    rtn = hm_sm2.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for DG method: " + rtn.message)
        return

    # Calculate the boundary layer parameters
    x_ref = so68.x()
    delta_d_ref = so68.delta_d()
    delta_m_ref = so68.delta_m()
    shape_d_ref = so68.shape_d()
    c_f_ref = so68.c_f()
    u_e_ref = so68.u_e()

    x = np.linspace(x_ref[0], x_ref[-1], 101)

    delta_d_dg_reg = dg_reg.delta_d(x)
    delta_m_dg_reg = dg_reg.delta_m(x)
    shape_d_dg_reg = dg_reg.shape_d(x)
    c_f_dg_reg = 2*dg_reg.tau_w(x, rho)/(rho*dg_reg.u_e(x)**2)
    v_e_dg_reg = dg_reg.v_e(x)

    #delta_d_dg_sm = dg_sm.delta_d(x)
    #delta_m_dg_sm = dg_sm.delta_m(x)
    #shape_d_dg_sm = dg_sm.shape_d(x)
    #c_f_dg_sm = 2*dg_sm.tau_w(x, rho)/(rho*dg_sm.u_e(x)**2)
    #v_e_dg_sm = dg_sm.v_e(x)

    delta_d_head_sm2 = hm_sm2.delta_d(x)
    delta_m_head_sm2 = hm_sm2.delta_m(x)
    shape_d_head_sm2 = hm_sm2.shape_d(x)
    c_f_head_sm2 = 2*hm_sm2.tau_w(x, rho)/(rho*hm_sm2.u_e(x)**2)
    v_e_head_sm2 = hm_sm2.v_e(x)

    # Plot results
    fig = plt.figure()
    #fig.set_figwidth(10)
    #fig.set_figheight(15)
    #gs = GridSpec(6, 2, figure=fig)
    gs = GridSpec(5, 2, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_shape_d = fig.add_subplot(gs[2, 0])
    axis_shape_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_u_e = fig.add_subplot(gs[4, 0])
    axis_u_e_diff = fig.add_subplot(gs[4, 1])
    #axis_du_e = fig.add_subplot(gs[5, 0])
    #axis_v_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "L & T Data"
    dg_reg_color = "red"
    dg_reg_label = "Head"
    dg_sm_color = "green"
    dg_sm_label = r"Head(Smooth $u_e$)"
    head_sm2_color = "orange"
    head_sm2_label = r"Head(d$u_e/$d$s$)"

    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(x_ref, delta_d_ref, color=ref_color, linestyle="",
                        marker="o", label=ref_label)
    head_reg_curve = ax.plot(x, delta_d_dg_reg, color="#A4D65E",
                             label=dg_reg_label)
    #head_sm_curve = ax.plot(x, delta_d_dg_sm, color="#A4D65E",
    #                        label=dg_sm_label,linestyle='--')
    head_sm2_curve = ax.plot(x, delta_d_head_sm2, color="#A4D65E",
                             label=head_sm2_label,linestyle=':')
    _ = ax.set_ylim((0, 0.008))
    _ = ax.set_ylabel(r"$\delta_d$ [m]")
    #_ = ax.set_ylabel(r"$\delta^*$ (m)")
    ax.grid(True)

    ax = axis_delta_d_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.delta_d(x_ref)/delta_d_ref),
                color="#A4D65E")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.delta_d(x_ref)/delta_d_ref),
    #            color="#A4D65E",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.delta_d(x_ref)/delta_d_ref),
                color="#A4D65E",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    _ = ax.plot(x_ref, delta_m_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, delta_m_dg_reg, color="#BB00FF")
    #_ = ax.plot(x, delta_m_dg_sm, color="#3A913F",linestyle='--')
    _ = ax.plot(x, delta_m_head_sm2, color="#BB00FF",linestyle=':')
    _ = ax.set_ylim((0, 0.003))
    _ = ax.set_ylabel(r"$\delta_m$ [m]")
    #_ = ax.set_ylabel(r"$\theta$ (m)")
    ax.grid(True)

    ax = axis_delta_m_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.delta_m(x_ref)/delta_m_ref),
                color="#BB00FF")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.delta_m(x_ref)/delta_m_ref),
    #            color="#3A913F",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.delta_m(x_ref)/delta_m_ref),
                color="#BB00FF",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    _ = ax.plot(x_ref, shape_d_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, shape_d_dg_reg, color="#612D00")
    #_ = ax.plot(x, shape_d_dg_sm, color="#A4D65E",linestyle='--')
    _ = ax.plot(x, shape_d_head_sm2, color="#612D00",linestyle=':')
    _ = ax.set_ylim(1, 3)
    _ = ax.set_ylabel(r"$H_d$")
    #_ = ax.set_ylabel(r"$H$")
    ax.grid(True)

    ax = axis_shape_d_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.shape_d(x_ref)/shape_d_ref),
                color="#612D00")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.shape_d(x_ref)/shape_d_ref),
    #            color="#A4D65E",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.shape_d(x_ref)/shape_d_ref),
                color="#612D00",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    _ = ax.plot(x_ref, c_f_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, c_f_dg_reg, color="#818181")
    #_ = ax.plot(x, c_f_dg_sm, color="#F8E08E",linestyle='--')
    _ = ax.plot(x, c_f_head_sm2, color="#818181",linestyle=':')
    _ = ax.set_ylim((0, 0.005))
    _ = ax.set_ylabel(r"$c_f$")
    ax.grid(True)

    ax = axis_c_f_diff
    temp = 2*dg_reg.tau_w(x_ref, rho)/(rho*dg_reg.u_e(x_ref)**2)
    #_ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
    #            color="#818181")
    #temp = 2*dg_sm.tau_w(x_ref, rho)/(rho*dg_sm.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color="#818181",linestyle='--')
    temp = 2*hm_sm2.tau_w(x_ref, rho)/(rho*hm_sm2.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color="#818181",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_u_e
    _ = ax.plot(x_sm, u_e_sm, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, dg_reg.u_e(x), color="#9FC9CD")
    #_ = ax.plot(x, dg_sm.u_e(x), color="#154734",linestyle='--')
    _ = ax.plot(x, hm_sm2.u_e(x), color="#9FC9CD",linestyle=':')
    _ = ax.set_ylim((10, 30))
    _ = ax.set_xlabel(r"$s$ [m]")
    _ = ax.set_ylabel(r"$u_e$ [m/s]")
    ax.grid(True)

    ax = axis_u_e_diff
    _ = ax.plot(x_ref, np.abs(1-dg_reg.u_e(x_ref)/u_e_ref),
                color="#9FC9CD")
    #_ = ax.plot(x_ref, np.abs(1-dg_sm.u_e(x_ref)/u_e_ref),
    #            color="#154734",linestyle='--')
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.u_e(x_ref)/u_e_ref),
                color="#9FC9CD",linestyle=':')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Rel. Error")
    _ = ax.set_xlabel(r"$s$ [m]")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Transpiration velocity in 5,:
    # ax = axis_du_e
    # _ = ax.plot(x_sm, du_e_sm, color=ref_color, linestyle="", marker="o")
    # _ = ax.plot(x, dg_reg.du_e(x), color="#BD8B13")
    # #_ = ax.plot(x, dg_sm.du_e(x), color="#BD8B13",linestyle='--')
    # _ = ax.plot(x, hm_sm2.du_e(x), color="#BD8B13",linestyle=':')
    # _ = ax.set_ylim((3, 6))
    # _ = ax.set_xlabel(r"$s$ [m]")
    # _ = ax.set_ylabel(r"d$u_e/$d$s$ [m/s^2]")
    # ax.grid(True)

    # ax = axis_v_e
    # _ = ax.plot(x, v_e_dg_reg, color="#789F90")
    # #_ = ax.plot(x, v_e_dg_sm, color="#789F90",linestyle='--')
    # _ = ax.plot(x, v_e_head_sm2, color="#789F90",linestyle=':')
    # _ = ax.set_ylim((0, 0.05))
    # _ = ax.set_xlabel(r"$s$ [m]")
    # _ = ax.set_ylabel(r"$v_e$ [m/s]")
    # ax.grid(True)

    _ = fig.subplots_adjust(left=0.15, wspace=0.5, hspace=.5, top=.95, bottom=0.15)
    _ = fig.legend(handles=[ref_curve[0], head_reg_curve[0],
                            head_sm2_curve[0]], labels=[ref_label, dg_reg_label,
                           head_sm2_label], loc="upper center",
                   bbox_to_anchor=(.51, 0.06), ncol=4, borderaxespad=.2)
    fig.savefig(file_name+'\\'+'HEAD_CASE1300_Comp.png')
    #plt.show()


if __name__ == "__main__":
    compare_case1300()
