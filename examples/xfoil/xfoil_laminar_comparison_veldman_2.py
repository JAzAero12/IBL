"""
Comparing Thwaites' method solution against XFoil case.

This example shows a comparison between Thwaites' method and XFoil laminar
results for a NACA 0003 airfoil using the XFoil edge velocity profile. It shows
similar results to Figures 3.17 to 3.19 in Edland thesis.
"""

# pylint: disable=too-many-statements,too-many-locals

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.reference import XFoilReader
from ibl.initial_condition import ManualCondition
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
from scipy.signal import savgol_filter

from ibl.interaction_law import interaction_law, c_maker_func

import os

# NACA 0003 Re = 1000
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['figure.figsize'] = [13, 8]
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

file_name = "NACA0003_Re1E3_INT"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

def compare_xfoil_laminar() -> None:
    """Compare the Thwaites' method results to XFoil results."""
    # Read in XFoil data
    data_dir = Path(__file__).resolve().parent.parent.joinpath("data")
    if not data_dir.exists():
        raise IOError(f"Cannot find data directory: {data_dir}")

    inv_file = data_dir.joinpath("xfoil_0003_inviscid_dump.txt")
    visc_file = data_dir.joinpath("xfoil_0003_laminar_dump.txt")
    airfoil_name = "NACA 0003"
    alpha = 0
    c = 1  # (m)
    u_inf = 20  # (m/s)
    re = 1000
    rho_inf = 1.2
    nu_inf = u_inf*c/re
    x_trans = 1
    n_trans = 9
    xfoil_inv = XFoilReader(str(inv_file))
    xfoil_inv.name = airfoil_name
    xfoil_inv.alpha = alpha
    xfoil_inv.c = c
    xfoil_inv.u_ref = u_inf
    xfoil_visc = XFoilReader(str(visc_file))
    xfoil_visc.name = airfoil_name
    xfoil_visc.alpha = alpha
    xfoil_visc.c = c
    xfoil_visc.u_ref = u_inf
    xfoil_visc.reynolds = re
    xfoil_visc.x_trans_lower = x_trans
    xfoil_visc.x_trans_upper = x_trans
    xfoil_visc.n_trans = n_trans

    s_ref = xfoil_inv.s_upper()
    u_e_inv = xfoil_inv.u_e_upper()

    fig, temp = plt.subplots()
    temp.plot(s_ref,u_e_inv)
    temp.set_title('Original Inviscid Velocity')

    u_e_visc = xfoil_visc.u_e_upper()
    s = np.linspace(s_ref[0], s_ref[-1], 101)


    ic = ManualCondition(delta_d=xfoil_visc.delta_d_upper()[0],delta_m=xfoil_visc.delta_m_upper()[0],delta_k=xfoil_visc.delta_k_upper()[0])

    # Setup Thwaites methods
    delta_m0 = xfoil_visc.delta_m_upper()[0]
    tm_v_orig = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref, xfoil_inv.u_e_upper()],
                                      data_fits="Spline")
    tm_v_orig.initial_delta_m = delta_m0

    #tm_v_orig = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[s_ref, u_e_inv],ic=ic,cf_crit=9e-4)

    rtn = tm_v_orig.solve(x0=s_ref[0], x_end=s_ref[-1])

    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    print('Original')
    print(rtn.message)
    print(rtn.x_end)

    ## Did the calcs survive? ##
    print('Did the calculations survive (inviscid only)?')
    print(f'delta_d survival: {not np.any(np.isnan(tm_v_orig.delta_d(s_ref)))}')
    print(f'delta_m survival: {not np.any(np.isnan(tm_v_orig.delta_m(s_ref)))}')
    print(f'shape_d survival: {not np.any(np.isnan(tm_v_orig.shape_d(s_ref)))}')
    print(f'c_f survival: {not np.any(np.isnan(tm_v_orig.tau_w(s_ref,rho_inf)))}')
    print(f'v_e survival: {not np.any(np.isnan(tm_v_orig.v_e(s_ref)))}')

    s_sep_visc = np.inf #Keep going despite separation to see what happens
    if rtn.status == 1:
        s_sep_visc = rtn.x_end

    #demo of c_matrix accuracy
    n_pts = np.array([10.,20.,40.,80.,160.,320.,640.])
    c_mat_err = np.zeros_like(n_pts)
    for i,n_pt in enumerate(n_pts):
        n_pt = int(n_pt)
        s_rng = np.linspace(0,1,n_pt)
        index = int(n_pt/2)
        dv = s_rng[index]*s_rng - .5*s_rng**2
        c_m = c_maker_func(s_rng)
        test = c_m @ dv
        c_mat_err[i] = abs(1/np.pi - test[index])/abs(1/np.pi)
    
    fig, c_er = plt.subplots(constrained_layout=True,figsize=[10,6])
    c_er.plot(n_pts,c_mat_err,marker='o',color='#154734')
    c_er.set_xlabel('Number of Points')
    c_er.set_ylabel('Relative Error')
    c_er.set_yscale('log')
    c_er.set_xscale('log')
    fig.savefig(file_name+'\\'+'c_mat_err.png')


    u_e_inv_orig = xfoil_inv.u_e_upper()

    fig, hardcode = plt.subplots()
    hardcode.plot(s_ref,u_e_inv,marker='*')
    hardcode.plot(s_ref,u_inf*np.ones_like(s_ref))
    hardcode.set_title('u_e_inv profile, ')

    func_corrections,u_e_preprocess = interaction_law(s_ref,u_inf,u_e_inv,nu_inf,debug=True)
    #Use delta_d from xfoil in wake?
    chord_len = abs(s_ref[-1] - s_ref[0])
    #Create a wake approximation for plotting the full range of corrective values
    #wake_spacing = abs(s_ref[-1] - s_ref[-2])
    #wake_approx_s = np.arange(s_ref[-1]+wake_spacing, 2.*chord_len+wake_spacing, wake_spacing)
    #Cluster method
    wake_cluster = np.linspace(0,1,len(s_ref))**1.2
    wake_approx_s = s_ref[-1] + chord_len*wake_cluster
    wake_approx_s = wake_approx_s[1:]
    #delta_d_wake = xfoil_visc.delta_d_upper()[-1] * np.exp(-(wake_approx_s - s_ref[-1])/(wake_approx_s[-1] - s_ref[-1])) #same wake correlation
    #wakeue = u_e_inv[-1]*np.ones_like(delta_d_wake)
    #total_delta_d = np.concatenate((xfoil_visc.delta_d_upper(),delta_d_wake))
    #d_vec2 = total_delta_d*np.concatenate((u_e_inv,wakeue))
    #temp = c_mat @ d_vec2
    #func_corrections = temp
    #func_corrections[len(s_ref)-1] = 14*abs(func_corrections[len(s_ref)-1]) #Works for DG, main idea moving forward is to create a better delta_d*u_e profile
    #func_corrections[len(s_ref)] = 5*abs(func_corrections[len(s_ref)])

    fig, velcomp = plt.subplots()
    velcomp.plot(s_ref,func_corrections[:len(u_e_inv)],label='Veldman from LE to TE',marker='*')
    velcomp.plot(s_ref,u_e_visc-u_e_preprocess+func_corrections[:len(u_e_inv)],label='Veldman from LE to TE and Preprocessed u_e',marker='*')
    velcomp.plot(s_ref,u_e_visc-u_e_inv,label='Visc - Inv',marker='*')
    velcomp.plot([s_ref[-1],s_ref[-1]],[-5.,1.5])
    velcomp.legend()


    tm_v_corr = ThwaitesMethodNonlinear(nu=nu_inf, 
                                        U_e=[s_ref, u_e_inv+func_corrections[:len(u_e_inv)]],
                                        #U_e=[s_ref, u_e_preprocess+func_corrections[:len(u_e_inv)]], #This provides the best results, and calcs all the way through
                                        #U_e=[s_ref, u_e_preprocess],
                                     data_fits="Spline")
    #delta_m0 = float(np.sqrt(0.075*nu_inf/tm_v_corr.du_e(s[0])))  # Moran's method
    delta_m0 = xfoil_visc.delta_m_upper()[0]
    tm_v_corr.initial_delta_m = delta_m0

    #tm_v_corr = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[s_ref, u_e_preprocess+func_corrections[:len(u_e_inv)]],ic=ic,cf_crit=9e-4)

    rtn = tm_v_corr.solve(x0=s[0], x_end=s[-1])
    print('With Corrective Values')
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    else:
        print(rtn.message)
        print(rtn.x_end)
        #print(s_ref[-2]) #Second to last position of s_ref

    s_sep_inv = np.inf #Force a full plot, regardless of if the calculations 'survive'
    if rtn.status == 1:
        s_sep_inv = rtn.x_end

    # Calculate the boundary layer parameters
    delta_d_ref = xfoil_visc.delta_d_upper()
    delta_m_ref = xfoil_visc.delta_m_upper()
    shape_d_ref = xfoil_visc.shape_d_upper()
    c_f_ref = xfoil_visc.c_f_upper()

    s_ref_visc = s_ref[s_ref < s_sep_visc]
    delta_d_ref_visc = delta_d_ref[s_ref < s_sep_visc]
    delta_m_ref_visc = delta_m_ref[s_ref < s_sep_visc]
    shape_d_ref_visc = shape_d_ref[s_ref < s_sep_visc]
    c_f_ref_visc = c_f_ref[s_ref < s_sep_visc]

    s_ref_inv = s_ref[s_ref < s_sep_inv]
    delta_d_ref_inv = delta_d_ref[s_ref < s_sep_inv]
    delta_m_ref_inv = delta_m_ref[s_ref < s_sep_inv]
    shape_d_ref_inv = shape_d_ref[s_ref < s_sep_inv]
    c_f_ref_inv = c_f_ref[s_ref < s_sep_inv]

    s_visc = np.linspace(s_ref[0], min(s_ref[-1], s_sep_visc), 101)
    delta_d_visc = tm_v_orig.delta_d(s_visc)
    delta_m_visc = tm_v_orig.delta_m(s_visc)
    shape_d_visc = tm_v_orig.shape_d(s_visc)
    ue = tm_v_orig.u_e(s_visc)
    c_f_visc = 2*tm_v_orig.tau_w(s_visc, rho_inf)/(rho_inf*u_inf**2)
    v_e_visc = tm_v_orig.v_e(s_visc)
    du_e_visc = tm_v_orig.du_e(s_visc)
    d2u_e_visc = tm_v_orig.d2u_e(s_visc)

    s_inv = np.linspace(s_ref[0], min(s_ref[-1], s_sep_inv), 101)
    delta_d_inv = tm_v_corr.delta_d(s_ref_inv)
    delta_m_inv = tm_v_corr.delta_m(s_ref_inv)
    shape_d_inv = tm_v_corr.shape_d(s_ref_inv)
    ue = tm_v_corr.u_e(s_ref_inv)
    c_f_inv = 2*tm_v_corr.tau_w(s_ref_inv, rho_inf)/(rho_inf*u_inf**2)
    v_e_inv = tm_v_corr.v_e(s_ref_inv)
    du_e_inv = tm_v_corr.du_e(s_ref_inv)
    d2u_e_inv = tm_v_corr.d2u_e(s_ref_inv)

    fig, due_check = plt.subplots()
    due_check.plot(s_ref_inv,du_e_inv)
    due_check.set_ylim([-2,3])
    due_check.set_title('Veldman+ Inviscid Velocity Derivative')

    fig, ue_check = plt.subplots(constrained_layout=True)
    ue_check.plot(s_ref,u_e_visc,label=r'Viscous',color='black',linestyle='--',linewidth=4.)
    ue_check.plot(s_ref,u_e_inv,label=r'Inviscid',color='#9FC9CD')
    ue_check.plot(s_ref,u_e_preprocess,label=r'Preproc Inv.',color='#9FC9CD',linestyle='-.')
    ue_check.plot(s_ref,u_e_inv+func_corrections[:len(u_e_inv)],label=r'Inv. + Int. Law',color='#9FC9CD',linestyle=':')
    ue_check.plot(s_ref,u_e_preprocess+func_corrections[:len(u_e_inv)],label=r'Preproc Inv. + Int. Law',color='#9FC9CD',linestyle='--')
    ue_check.set_ylim([18,21.5])
    ue_check.set_ylabel(r'$u_e$ [m/s]')
    ue_check.set_xlabel(r's [m]')
    ue_check.legend(borderaxespad=-10)
    fig.tight_layout()
    fig.savefig(file_name+'\\'+'u_e_profs.png')

    ## Did the calcs survive? ##
    print('Did the calculations survive (inviscid + corrections)?')
    print(f'delta_d survival: {not np.any(np.isnan(tm_v_corr.delta_d(s_ref)))}')
    print(f'delta_m survival: {not np.any(np.isnan(tm_v_corr.delta_m(s_ref)))}')
    print(f'shape_d survival: {not np.any(np.isnan(tm_v_corr.shape_d(s_ref)))}')
    print(f'c_f survival: {not np.any(np.isnan(c_f_inv))}')
    print(f'v_e survival: {not np.any(np.isnan(v_e_inv))}')


    tm_v_corr2 = ThwaitesMethodNonlinear(nu=nu_inf, 
                                        U_e=[s_ref, u_e_preprocess+func_corrections[:len(u_e_inv)]], #This provides the best results, and calcs all the way through
                                     data_fits="Spline")
    #delta_m0 = float(np.sqrt(0.075*nu_inf/tm_v_corr.du_e(s[0])))  # Moran's method
    delta_m0 = xfoil_visc.delta_m_upper()[0]
    tm_v_corr2.initial_delta_m = delta_m0

    #tm_v_corr = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[s_ref, u_e_preprocess+func_corrections[:len(u_e_inv)]],ic=ic,cf_crit=9e-4)

    rtn = tm_v_corr2.solve(x0=s[0], x_end=s[-1])
    print('With Corrective Values')
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    else:
        print(rtn.message)
        print(rtn.x_end)
        #print(s_ref[-2]) #Second to last position of s_ref

    s_sep_inv = np.inf #Force a full plot, regardless of if the calculations 'survive'
    if rtn.status == 1:
        s_sep_inv = rtn.x_end

    s_ref_inv2 = s_ref[s_ref < s_sep_inv]

    s_inv = np.linspace(s_ref[0], min(s_ref[-1], s_sep_inv), 101)
    delta_d_inv2 = tm_v_corr2.delta_d(s_ref_inv2)
    delta_m_inv2 = tm_v_corr2.delta_m(s_ref_inv2)
    shape_d_inv2 = tm_v_corr2.shape_d(s_ref_inv2)
    c_f_inv2 = 2*tm_v_corr2.tau_w(s_ref_inv2, rho_inf)/(rho_inf*u_inf**2)
    v_e_inv2 = tm_v_corr2.v_e(s_ref_inv2)
    du_e_inv2 = tm_v_corr2.du_e(s_ref_inv2)
    d2u_e_inv2 = tm_v_corr2.d2u_e(s_ref_inv2)
    delta_d_ref_inv2 = delta_d_ref[s_ref < s_sep_inv]
    delta_m_ref_inv2 = delta_m_ref[s_ref < s_sep_inv]
    shape_d_ref_inv2 = shape_d_ref[s_ref < s_sep_inv]
    c_f_ref_inv2 = c_f_ref[s_ref < s_sep_inv]

    ## Did the calcs survive? ##
    print('Did the calculations survive (preprocessed inviscid + corrections)?')
    print(f'delta_d survival: {not np.any(np.isnan(tm_v_corr2.delta_d(s_ref)))}')
    print(f'delta_m survival: {not np.any(np.isnan(tm_v_corr2.delta_m(s_ref)))}')
    print(f'shape_d survival: {not np.any(np.isnan(tm_v_corr2.shape_d(s_ref)))}')
    print(f'c_f survival: {not np.any(np.isnan(c_f_inv2))}')
    print(f'v_e survival: {not np.any(np.isnan(v_e_inv2))}')

    # Plot results,
    # pylint: disable=duplicate-code

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24
    plt.rcParams['figure.figsize'] = [15, 15]
    plt.rcParams['legend.loc'] = 'lower center'
    plt.rcParams['legend.borderaxespad'] = -10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["lines.linewidth"] = 3.

    fig = plt.figure()
    #fig.set_figwidth(10)
    #fig.set_figheight(30)
    gs = GridSpec(4, 2, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_shape_d = fig.add_subplot(gs[2, 0])
    axis_shape_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    #axis_u_e = fig.add_subplot(gs[4, 0])
    #axis_du_e = fig.add_subplot(gs[4, 1])
    #axis_d2u_e = fig.add_subplot(gs[5, 0])
    #axis_v_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "XFOIL"
    thwaites_v_orig_color = "#154734"
    thwaites_v_orig_label = r"Inviscid"
    thwaites_v_corr_color = "#BD8B13"
    thwaites_v_corr_label = r"Inv. + Int. Law"
    thwaites_v_corr_label2 = r"Preproc Inv. + Int. Law"
    # pylint: enable=duplicate-code


    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(s_ref/c, delta_d_ref/c, color=ref_color,
                        label=ref_label,linestyle='--',linewidth=4.)
    thwaites_visc_curve = ax.plot(s_visc/c, delta_d_visc/c,
                                  color='#A4D65E',
                                  label=thwaites_v_orig_label)
    thwaites_inv_curve = ax.plot(s_ref_inv/c, delta_d_inv/c,
                                 color='#A4D65E',
                                 label=thwaites_v_corr_label,linestyle=':')
    thwaites_inv_curve = ax.plot(s_ref_inv2/c, delta_d_inv2/c,
                                 color='#A4D65E',
                                 label=thwaites_v_corr_label2,linestyle='--')
    _ = ax.set_ylim(0, 0.1)
    _ = ax.set_ylabel(r"$\delta_d/c$")
    ax.grid(True)

    ax = axis_delta_d_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-tm_v_orig.delta_d(s_ref_visc)/delta_d_ref_visc),
                color='#A4D65E')
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-tm_v_corr.delta_d(s_ref_inv)/delta_d_ref_inv),
                color='#A4D65E',linestyle=':')
    _ = ax.plot(s_ref_inv2/c,
                np.abs(1-tm_v_corr2.delta_d(s_ref_inv2)/delta_d_ref_inv2),
                color='#A4D65E',linestyle='--')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    _ = ax.plot(s_ref/c, delta_m_ref/c, color=ref_color,linestyle='--',linewidth=4.)
    _ = ax.plot(s_visc/c, delta_m_visc/c, color='#BB00FF')
    _ = ax.plot(s_ref_inv/c, delta_m_inv/c, color='#BB00FF',linestyle=':')
    _ = ax.plot(s_ref_inv2/c, delta_m_inv2/c, color='#BB00FF',linestyle='--')
    _ = ax.set_ylim(0, 0.025)
    _ = ax.set_ylabel(r"$\delta_m/c$")
    ax.grid(True)

    ax = axis_delta_m_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-tm_v_orig.delta_m(s_ref_visc)/delta_m_ref_visc),
                color='#BB00FF')
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-tm_v_corr.delta_m(s_ref_inv)/delta_m_ref_inv),
                color='#BB00FF',linestyle=':')
    _ = ax.plot(s_ref_inv2/c,
                np.abs(1-tm_v_corr2.delta_m(s_ref_inv2)/delta_m_ref_inv2),
                color='#BB00FF',linestyle='--')
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_ylim((1e-5,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    _ = ax.plot(s_ref/c, shape_d_ref, color=ref_color,linestyle='--',linewidth=4.)
    _ = ax.plot(s_visc/c, shape_d_visc, color='#612D00')
    _ = ax.plot(s_ref_inv/c, shape_d_inv, color='#612D00',linestyle=':')
    _ = ax.plot(s_ref_inv2/c, shape_d_inv2, color='#612D00',linestyle='--')
    _ = ax.set_ylim(2.2, 4)
    _ = ax.set_ylabel(r"$H_d$")
    ax.grid(True)

    ax = axis_shape_d_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-tm_v_orig.shape_d(s_ref_visc)/shape_d_ref_visc),
                color='#612D00')
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-tm_v_corr.shape_d(s_ref_inv)/shape_d_ref_inv),
                color='#612D00',linestyle=':')
    _ = ax.plot(s_ref_inv2/c,
                np.abs(1-tm_v_corr2.shape_d(s_ref_inv2)/shape_d_ref_inv2),
                color='#612D00',linestyle='--')
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((5e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    _ = ax.plot(s_ref/c, c_f_ref, color=ref_color,linestyle='--',linewidth=4.)
    _ = ax.plot(s_visc/c, c_f_visc, color='#818181')
    _ = ax.plot(s_ref_inv/c, c_f_inv, color='#818181',linestyle=':')
    _ = ax.plot(s_ref_inv2/c, c_f_inv2, color='#818181',linestyle='--')
    _ = ax.set_ylabel(r"$c_f$")
    _ = ax.set_xlabel(r"$s/c$")
    ax.grid(True)

    ax = axis_c_f_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-2*tm_v_orig.tau_w(s_ref_visc,
                                         rho_inf)/(rho_inf
                                                   *u_inf**2)/c_f_ref_visc),
                color='#818181',linestyle=':')
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-2*tm_v_corr.tau_w(s_ref_inv,
                                        rho_inf)/(rho_inf
                                                  *u_inf**2)/c_f_ref_inv),
                color='#818181')
    _ = ax.plot(s_ref_inv2/c,
                np.abs(1-2*tm_v_corr2.tau_w(s_ref_inv2,
                                        rho_inf)/(rho_inf
                                                  *u_inf**2)/c_f_ref_inv2),
                color='#818181',linestyle='--')
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_xlabel(r"$s/c$")
    _ = ax.set_ylim((5e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # # Edge velocity in 4,:
    # ax = axis_u_e
    # _ = ax.plot(s_ref/c, u_e_visc/u_inf, color='#154734')
    # _ = ax.plot(s_ref/c, u_e_inv/u_inf, color='#154734')
    # _ = ax.plot(s_ref/c, u_e_inv/u_inf, color='#154734',linestyle='--')
    # _ = ax.set_ylim(0, 1.1)
    # _ = ax.set_xlabel(r"$s/c$")
    # _ = ax.set_ylabel(r"$u_e/u_{inf}$")
    # ax.grid(True)
    
    # ax = axis_du_e
    # _ = ax.plot(s_visc/c, du_e_visc, color='#BD8B13')
    # _ = ax.plot(s_inv/c, du_e_inv, color='#BD8B13')
    # _ = ax.set_ylim(-2, 0.5)
    # _ = ax.set_xlabel(r"$s/c$")
    # _ = ax.set_ylabel(r"d$u_e/$d$x$ (1/s)")
    # ax.grid(True)


    _ = fig.subplots_adjust(left=0.15, wspace=0.5, hspace=.5, top=.95)
    _ = fig.legend(labels=[ref_label, thwaites_v_orig_label, thwaites_v_corr_label, thwaites_v_corr_label2], loc="upper center",
                   bbox_to_anchor=(.51, 0.06), ncol=4, borderaxespad=.8,columnspacing=1)
    #fig.tight_layout()
    fig.savefig(file_name+'\\'+'Int_Law_Comp.png')

    plt.show()

    ## Next, if given a profile with actual flow separation, does it overcorrect? (It shouldn't)
    
    #Using the same NACA 0003 airfoil, same Re of 1000, but an alpha of 6 degrees -> Flow separation at roughly x = .17648

    #Viscous velocity profile
    visc_s_ref = np.array([0.00289,0.00494,0.00633,0.00838,0.01231,0.02117,0.03357,
                            0.04642,0.05936,0.07233,0.08532,0.09832,0.11134,0.12436,
                            0.13739,0.15042,0.16345,0.17648,0.18952,0.20256,0.2156,
                            0.22864,0.24169,0.25473,0.26778,0.28082,0.29387,0.30692,
                            0.31997,0.33302,0.34607,0.35912,0.37217,0.38522,0.39827,
                            0.41132,0.42437,0.43743,0.45048,0.46353,0.47658,0.48964,
                            0.50269,0.51575,0.5288,0.54185,0.55491,0.56796,0.58102,
                            0.59407,0.60713,0.62018,0.63324,0.64629,0.65935,0.67241,
                            0.68546,0.69852,0.71157,0.72463,0.73769,0.75074,0.7638,
                            0.77685,0.78991,0.80297,0.81602,0.82908,0.84213,0.85519,
                            0.86825,0.8813,0.89436,0.90741,0.92047,0.93353,0.94658,
                            0.95964,0.97269,0.98574,0.99861,1.00742])

    visc_u_e = np.array([0.35295,1.18273,1.89317,1.88987,1.7211,1.58927,1.50113,
                            1.44701,1.409,1.37998,1.35662,1.33716,1.32051,1.30599,
                            1.29313,1.28162,1.27119,1.26168,1.25293,1.24485,1.23733,
                            1.23032,1.22374,1.21755,1.2117,1.20617,1.20092,1.19593,
                            1.19117,1.18662,1.18226,1.17809,1.17408,1.17023,1.16652,
                            1.16294,1.15949,1.15616,1.15293,1.14981,1.14679,1.14385,
                            1.141,1.13824,1.13555,1.13294,1.13039,1.12791,1.1255,
                            1.12314,1.12084,1.1186,1.11641,1.11427,1.11218,1.11013,
                            1.10813,1.10617,1.10426,1.10238,1.10054,1.09874,1.09697,
                            1.09524,1.09354,1.09188,1.09024,1.08863,1.08706,1.08551,
                            1.08399,1.08249,1.08102,1.07958,1.07816,1.07676,1.07538,
                            1.07403,1.0727,1.07139,1.07012,1.06927])
    
    visc_u_e = u_inf*visc_u_e

    #start a little after stag pt?
    visc_s_ref = visc_s_ref[2:]
    visc_u_e = visc_u_e[2:]

    #Initial conditions for this run
    delta_m0 = 0.0008336
    delta_d0 = 0.0018586
    delta_k0 = 0.0013514
    shape_d0 = 2.2295
    ic = ManualCondition(delta_d=delta_d0,delta_m=delta_m0,delta_k=delta_k0) #Pulled from XFOIL

    #tm_should_sep = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[visc_s_ref,visc_u_e],ic=ic)
    tm_should_sep = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[visc_s_ref,visc_u_e],data_fits='Spline')
    tm_should_sep.initial_delta_m = delta_m0

    #Should separate
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('NACA 0003 at alpha of 6 degrees')
    rtn = tm_should_sep.solve(visc_s_ref[0],visc_s_ref[-1])
    print("SHOULD BE SEPARATING, VISC NO CORRECTIONS")
    print('Starting location is:')
    print(visc_s_ref[0])

    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    else:
        print(rtn.status)
        print(rtn.x_end)

    func_corrections,_,_,_,visc_u_e_preproc = interaction_law(visc_s_ref,u_inf,visc_u_e,nu_inf)
    #visc_u_e = visc_u_e + func_corrections[:len(visc_u_e)]
    #visc_u_e = visc_u_e_preproc + func_corrections[:len(visc_u_e)] #as concluded by prior work, the best 'chance'

    fig, sep_visc_flo = plt.subplots()
    sep_visc_flo.plot(visc_s_ref,visc_u_e,label='Original viscous u_e')
    sep_visc_flo.plot(visc_s_ref,visc_u_e_preproc,label='Preprocessed viscous u_e')
    sep_visc_flo.plot(visc_s_ref,visc_u_e + func_corrections[:len(visc_u_e)],label='Original viscous u_e + corrections')
    sep_visc_flo.legend()
    sep_visc_flo.set_title('Viscous u_e results alpha = 6')
    #Storing initial conditions for PyBL


    #tm_should_sep = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[visc_s_ref,visc_u_e_preproc + func_corrections[:len(visc_u_e)]],ic=ic)
    tm_should_sep = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[visc_s_ref,visc_u_e_preproc + func_corrections[:len(visc_u_e)]],data_fits='Spline')
    tm_should_sep.initial_delta_m = delta_m0

    #Should separate
    rtn = tm_should_sep.solve(visc_s_ref[0],visc_s_ref[-1])
    print("SHOULD BE SEPARATING, VISC")
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    else:
        print(rtn.status)
        print(rtn.x_end)

    #Inviscid Velocity Profile
    inv_s_ref = np.array([0.,0.00881,0.02168,0.03473,0.04778,0.06084,0.07389,
                        0.08695,0.1,0.11306,0.12612,0.13917,0.15223,0.16528,
                        0.17834,0.1914,0.20445,0.21751,0.23056,0.24362,0.25668,
                        0.26973,0.28279,0.29585,0.3089,0.32196,0.33501,0.34807,
                        0.36112,0.37418,0.38724,0.40029,0.41335,0.4264,0.43946,
                        0.45251,0.46556,0.47862,0.49167,0.50473,0.51778,0.53083,
                        0.54389,0.55694,0.56999,0.58305,0.5961,0.60915,0.6222,
                        0.63525,0.6483,0.66135,0.6744,0.68745,0.7005,0.71355,
                        0.72659,0.73964,0.75269,0.76573,0.77877,0.79182,0.80486,
                        0.8179,0.83093,0.84397,0.857,0.87003,0.88306,0.89608,
                        0.90909,0.9221,0.93509,0.94806,0.961,0.97385,0.98625,
                        0.99511,0.99904,1.00109])
    inv_s_ref = max(inv_s_ref) - np.flip(inv_s_ref)

    inv_u_e = np.array([0.9199,0.97237,0.98607,0.99592,1.00339,1.00966,1.01514,
                        1.02007,1.02461,1.02884,1.03283,1.03662,1.04027,1.04378,
                        1.0472,1.05052,1.05378,1.05698,1.06014,1.06325,1.06635,
                        1.06942,1.07248,1.07553,1.07859,1.08165,1.08472,1.0878,
                        1.09091,1.09404,1.0972,1.10039,1.10362,1.1069,1.11022,
                        1.1136,1.11703,1.12053,1.12409,1.12773,1.13144,1.13524,
                        1.13914,1.14313,1.14723,1.15145,1.15579,1.16028,1.16491,
                        1.1697,1.17467,1.17983,1.18521,1.19082,1.19669,1.20284,
                        1.20932,1.21616,1.22341,1.23112,1.23935,1.2482,1.25775,
                        1.26814,1.2795,1.29206,1.30606,1.32185,1.33992,1.36093,
                        1.38589,1.41633,1.45477,1.50569,1.57792,1.69258,1.90659,
                        2.37818,3.02018,5.1105])
    inv_u_e = u_inf*inv_u_e
    inv_u_e = np.flip(inv_u_e)

    #start a little after stag pt?
    inv_s_ref = inv_s_ref[2:]
    inv_u_e = inv_u_e[2:]

    #tm_should_sep_inv = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[inv_s_ref,inv_u_e],ic=ic)
    tm_should_sep_inv = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[inv_s_ref,inv_u_e],data_fits='Spline')
    tm_should_sep_inv.initial_delta_m = delta_m0

    #Should separate
    rtn = tm_should_sep_inv.solve(inv_s_ref[0],inv_s_ref[-1])
    print("SHOULD BE SEPARATING INV NO CORRECTIONS")
    print('Starting location is:')
    print(inv_s_ref[0])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    else:
        print(rtn.status)
        print(rtn.x_end)


    func_corrections,_,_,_,inv_u_e_preproc = interaction_law(inv_s_ref,u_inf,inv_u_e,nu_inf,debug=True)
    #inv_u_e = inv_u_e + func_corrections[:len(inv_u_e)]

    fig, sep_inv_flo = plt.subplots()
    sep_inv_flo.plot(inv_s_ref,inv_u_e,label='Original inviscid u_e')
    sep_inv_flo.plot(inv_s_ref,inv_u_e_preproc,label='Preprocessed inviscid u_e')
    sep_inv_flo.plot(inv_s_ref,inv_u_e + func_corrections[:len(inv_u_e)],label='Original inviscid u_e + corrections')
    sep_inv_flo.legend()
    sep_inv_flo.set_title('Inviscid u_e results alpha = 6')

    #tm_should_sep_inv = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[inv_s_ref,inv_u_e_preproc + func_corrections[:len(inv_u_e)]],ic=ic)
    tm_should_sep_inv = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[inv_s_ref,inv_u_e_preproc + func_corrections[:len(inv_u_e)]],data_fits='Spline')
    tm_should_sep_inv.initial_delta_m = delta_m0

    #Should separate
    rtn = tm_should_sep_inv.solve(inv_s_ref[0],inv_s_ref[-1])
    print("SHOULD BE SEPARATING INV")
    print('Starting location is:')
    print(inv_s_ref[0])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    else:
        print(rtn.status)
        print(rtn.x_end)


    fig, temp = plt.subplots()
    temp.plot(s_ref,u_e_visc,label='Viscous u_e')
    temp.plot(s_ref,u_e_inv+func_corrections[:len(u_e_inv)],label='Untreated + Corrected')
    temp.plot(s_ref,u_e_preprocess+func_corrections[:len(u_e_inv)],label='Treated + Corrected')
    temp.plot(s_ref,u_e_preprocess,label='Treated')
    temp.plot(s_ref,u_e_inv_orig,label='Untreated')
    temp.set_title('Inviscid Velocity Results')
    temp.legend()

    # Plot results,
    # pylint: disable=duplicate-code
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(6, 2, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_shape_d = fig.add_subplot(gs[2, 0])
    axis_shape_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_u_e = fig.add_subplot(gs[4, 0])
    axis_du_e = fig.add_subplot(gs[4, 1])
    #axis_d2u_e = fig.add_subplot(gs[5, 0])
    #axis_v_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "XFoil"
    thwaites_v_orig_color = "#154734"
    thwaites_v_orig_label = "Thwaites (Inviscid $U_e$)"
    thwaites_v_corr_color = "#BD8B13"
    thwaites_v_corr_label = "Thwaites (Inviscid $U_e$ with Corrections)"
    # pylint: enable=duplicate-code


    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(s_ref/c, delta_d_ref/c, color=ref_color,
                        label=ref_label)
    thwaites_visc_curve = ax.plot(s_visc/c, delta_d_visc/c,
                                  color=thwaites_v_orig_color,
                                  label=thwaites_v_orig_label)
    thwaites_inv_curve = ax.plot(s_inv/c, delta_d_inv/c,
                                 color=thwaites_v_corr_color,
                                 label=thwaites_v_corr_label)
    _ = ax.set_ylim(0, 0.06)
    #_ = ax.set_ylabel(r"$\delta_d/c$")
    _ = ax.set_ylabel(r"$\delta^*/c$")
    ax.grid(True)

    ax = axis_delta_d_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-tm_v_orig.delta_d(s_ref_visc)/delta_d_ref_visc),
                color=thwaites_v_orig_color)
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-tm_v_corr.delta_d(s_ref_inv)/delta_d_ref_inv),
                color=thwaites_v_corr_color)
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    _ = ax.plot(s_ref/c, delta_m_ref/c, color=ref_color)
    _ = ax.plot(s_visc/c, delta_m_visc/c, color=thwaites_v_orig_color)
    _ = ax.plot(s_inv/c, delta_m_inv/c, color=thwaites_v_corr_color)
    _ = ax.set_ylim(0, 0.025)
    #_ = ax.set_ylabel(r"$\delta_m/c$")
    _ = ax.set_ylabel(r"$\theta/c$")
    ax.grid(True)

    ax = axis_delta_m_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-tm_v_orig.delta_m(s_ref_visc)/delta_m_ref_visc),
                color=thwaites_v_orig_color)
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-tm_v_corr.delta_m(s_ref_inv)/delta_m_ref_inv),
                color=thwaites_v_corr_color)
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_ylim((1e-5,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    _ = ax.plot(s_ref/c, shape_d_ref, color=ref_color)
    _ = ax.plot(s_visc/c, shape_d_visc, color=thwaites_v_orig_color)
    _ = ax.plot(s_inv/c, shape_d_inv, color=thwaites_v_corr_color)
    _ = ax.set_ylim(2.2, 3)
    #_ = ax.set_ylabel(r"$H_d$")
    _ = ax.set_ylabel(r"$H$")
    ax.grid(True)

    ax = axis_shape_d_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-tm_v_orig.shape_d(s_ref_visc)/shape_d_ref_visc),
                color=thwaites_v_orig_color)
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-tm_v_corr.shape_d(s_ref_inv)/shape_d_ref_inv),
                color=thwaites_v_corr_color)
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    _ = ax.plot(s_ref/c, c_f_ref, color=ref_color)
    _ = ax.plot(s_visc/c, c_f_visc, color=thwaites_v_orig_color)
    _ = ax.plot(s_inv/c, c_f_inv, color=thwaites_v_corr_color)
    _ = ax.set_ylabel(r"$c_f$")
    ax.grid(True)

    ax = axis_c_f_diff
    _ = ax.plot(s_ref_visc/c,
                np.abs(1-2*tm_v_orig.tau_w(s_ref_visc,
                                         rho_inf)/(rho_inf
                                                   *u_inf**2)/c_f_ref_visc),
                color=thwaites_v_orig_color)
    _ = ax.plot(s_ref_inv/c,
                np.abs(1-2*tm_v_corr.tau_w(s_ref_inv,
                                        rho_inf)/(rho_inf
                                                  *u_inf**2)/c_f_ref_inv),
                color=thwaites_v_corr_color)
    #_ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_ylim((1e-4,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_u_e
    _ = ax.plot(s_ref/c, u_e_visc/u_inf, color=thwaites_v_orig_color)
    _ = ax.plot(s_ref/c, u_e_inv/u_inf, color=thwaites_v_corr_color)
    _ = ax.set_ylim(0, 1.1)
    _ = ax.set_ylabel(r"$U_e/U_\infty$")
    ax.grid(True)
    
    ax = axis_du_e
    _ = ax.plot(s_visc/c, du_e_visc, color=thwaites_v_orig_color)
    _ = ax.plot(s_inv/c, du_e_inv, color=thwaites_v_corr_color)
    _ = ax.set_ylim(-2, 0.5)
    _ = ax.set_xlabel(r"$x/c$")
    _ = ax.set_ylabel(r"d$U_e/$d$x$ (1/s)")
    ax.grid(True)
    #
    ## Transpiration velocity in 5,:
    #ax = axis_d2u_e
    #_ = ax.plot(s_visc/c, d2u_e_visc, color=thwaites_visc_color)
    #_ = ax.plot(s_inv/c, d2u_e_inv, color=thwaites_inv_color)
    #_ = ax.set_ylim(-5, 5)
    #_ = ax.set_xlabel(r"$x/c$")
    #_ = ax.set_ylabel(r"d$^2U_e/$d$x^2$ (1/(m$\cdot$s)")
    #ax.grid(True)
    #
    #ax = axis_v_e
    #_ = ax.plot(s_visc/c, v_e_visc/u_inf, color=thwaites_visc_color)
    #_ = ax.plot(s_inv/c, v_e_inv/u_inf, color=thwaites_inv_color)
    #_ = ax.set_ylim(0, 0.1)
    #_ = ax.set_xlabel(r"$x/c$")
    #_ = ax.set_ylabel(r"$V_e/U_e$")
    #ax.grid(True)
    #
    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    _ = fig.legend(handles=[ref_curve[0], thwaites_visc_curve[0],
                            thwaites_inv_curve[0]],
                   labels=[ref_label, thwaites_v_orig_label, thwaites_v_corr_label],
                   loc="lower center", bbox_to_anchor=(0.45, 0.1), ncol=4,
                   borderaxespad=0.1,columnspacing=0)
    plt.show()
    pass


if __name__ == "__main__":
    compare_xfoil_laminar()
