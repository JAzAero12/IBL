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
    u_e_visc = xfoil_visc.u_e_upper()
    s = np.linspace(s_ref[0], s_ref[-1], 101)

    # Applying Veldman's corrections to see if implementation idea works
    # Applied to u_e_inv, other values the same
    # Imported file doesnt have a wake, inviscid analysis has no wake

    ## Testing Hillbert Integral Function ##
    #D1 = test_hillbert(21)
    #D2 = test_hillbert(43)
    #D3 = test_hillbert(87)
    #D4 = test_hillbert(175)
    #print('[step size, error (real - numerical)]')
    #print(D1)
    #print(D2)
    #print(D3)
    #print(D4)
    #print('[step size reduction, error reduction]')
    #print(D1/D2)
    #print(D2/D3)
    #print(D3/D4)
    #print(D1/D4)

    # Cebeci interaction implementation
    ReX = (u_inf * s_ref)/nu_inf
    LEN = len(u_e_inv)
    dStar_FP = np.zeros(LEN)
    dStar_FP[1:] = .046875 * s_ref[1:] / ReX[1:]**.2
    print(dStar_FP[-1])
    Di = u_e_inv * dStar_FP
    veld_corrections = np.zeros(LEN)

    #print(s_ref)

    ## This is wake data from X-Foil, NACA 0003, RE = 1000 ##
    wakePOS = np.array([2.00356,2.00963,2.01675,2.02514,2.03499,2.04658,
                        2.0602,2.07623,2.09506,2.11721,2.14325,2.17387,
                        2.20988,2.25221,2.30199,2.36052,2.42934,2.51025,
                        2.60539,2.71726,2.8488,3.00346])
    wakePOS = wakePOS - 1

    wakeUE = np.array([1.04405,1.0431,1.04246,1.04172,1.04086,1.03987,
                       1.03874,1.03743,1.03595,1.03427,1.03237,1.03024,
                       1.02789,1.02532,1.02254,1.0196,1.01656,1.01353,
                       1.01066,1.00824,1.00678,1.01563])

    wakeUE = u_inf*wakeUE

    #d_star_wake = dStar_FP[-1]*np.exp(-(wakePOS - s_ref[-1])/(wakePOS[-1] - s_ref[-1]))
    #DI_W_xfoil = wakeUE * d_star_wake

    #s_inv_wake = np.concatenate((s_ref,wakePOS))
    #Di_Wake = np.concatenate((Di,DI_W_xfoil))

    # Can we pretend to have a wake?
    H = s_ref[-1] - s_ref[-2]
    #JankWakePos = np.linspace(s_ref[-1],2*s_ref[-1],21) # Position
    JankWakePos = np.arange(s_ref[-1],2*s_ref[-1]+H,H) # Position
    JankWakeU   = u_inf*np.ones(len(JankWakePos)) # Velocity
    JankWakePos = JankWakePos[1:]
    JankWakeU   = JankWakeU[1:]

    d_star_wake = dStar_FP[-1]/np.exp((JankWakePos - s_ref[-1])/(JankWakePos[-1] - s_ref[-1]))
    DI_W = u_inf * d_star_wake
    s_inv_wake = np.concatenate((s_ref,JankWakePos))
    Di_Wake = np.concatenate((Di,DI_W))


    C = np.zeros((LEN,LEN))
    E = np.zeros((LEN,LEN))
    A = 70
    #print(s_ref[A])
    veld_corrections_2 = Hillbert(s_inv_wake,Di_Wake)
    #veld_corrections_2 = Hillbert(s_ref,Di)
    ## First Correction, scale by 1/pi ##
    #veld_corrections_2 = (1/np.pi) * veld_corrections_2
    ## Second Correction, combine inviscid ue with corrected ue thru a linear transition
    #veld_corrections_2[A] = .2*veld_corrections_2[A]
    #veld_corrections_2[A+3] = .8*veld_corrections_2[A+3]
    #veld_corrections_2[A+1] = .4*veld_corrections_2[A+1]
    #veld_corrections_2[A+2] = .6*veld_corrections_2[A+2]
    #print(veld_corrections_2[A:])
    ue_corr = np.zeros(LEN)

    Cij = Hillbert_C(s_inv_wake)
    for i in range(LEN):
        gi = np.dot(Cij[i,:],Di_Wake)
        ue_new = u_e_inv[i] + gi
        ue_corr[i] = ue_new
        Di_Wake[i] = ue_new * dStar_FP[i]
    #print(ue_corr)
    #veld_corrections_2 = np.dot(Cij,Di_Wake)
    #print(veld_corrections_2)
    #veld_corrections_2 = (1/(s_ref[1]-s_ref[0])) * veld_corrections_2

    ## Constantly updating Di as progressing thru s_ref
    #Cij = Hillbert_C(s_inv_wake) # Just the Cij matrix

    # Creating the E matrix
    for i in range(A, LEN):
        for j in range(A+1,LEN-1): # Ei0 = EiL = 0 pg 75
            if j == i:
                E[i,j] = (((s_ref[i+1]-s_ref[i])/(s_ref[i+1]-s_ref[i-1]))*np.log(abs((s_ref[i]-s_ref[i-1])/(s_ref[i]-s_ref[i+1])))+2)/(s_ref[i]-s_ref[i-1])
            elif j == i+1:
                E[i,j] = (((s_ref[i]-s_ref[i-1])/(s_ref[i+1]-s_ref[i-1]))*np.log(abs((s_ref[i]-s_ref[i-1])/(s_ref[i]-s_ref[i+1])))-2)/(s_ref[i+1]-s_ref[i])
            else:
                E[i,j] = np.log(abs((s_ref[i]-s_ref[j-1])/(s_ref[i]-s_ref[j])))/(s_ref[j]-s_ref[j-1])

    # Creating the C matrix
    for i in range(A,LEN):
        for j in range(A,LEN):
            if j == (LEN-1):
                C[i,j] = (1/np.pi)*(E[i,j])
            else:
                C[i,j] = (1/np.pi)*(E[i,j] - E[i,j+1])

    # The vector of corrections should be C * Di
    #veld_corrections = C @ Di
    #veld_corrections = Hillbert(s_ref,Di)
    applied_corrections = np.zeros(LEN)
    applied_corrections[A:] = veld_corrections_2[A:LEN] # Corrections are applied to a limited range
    ue_test = u_e_inv + applied_corrections
    #ue_test2[-1]=u_e_visc[-1]
    #print(ue_test2/u_inf)
    # The scary part
    # Splitting the original and corrected versions for easier comparison
    print('Before')
    print(u_e_inv[A:])
    #ue_veld = u_e_inv
    ue_veld = ue_test
    #ue_veld[A:] = ue_test2[A:]
    print('After')
    print(u_e_inv[A:])
    #ue_veld[A:] = ue_test2[A:] # The whole domain is used, but actual corrections are confined
    ## Third Correction, last few terms were forced to be smaller
    #u_e_inv[-9:] = u_e_inv[-10]
    #print(u_e_inv - ue_corr)
    #print(u_e_visc - u_e_inv)
    #u_e_inv[-1] = u_e_visc[-1] # Even changing just the last point makes the calcs survive
    #u_e_inv[75:] = ue_corr[75:]

    # Setup Thwaites methods
    delta_m0 = xfoil_visc.delta_m_upper()[0]
    tm_visc = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref, u_e_visc],
                                      data_fits="Spline")
    tm_visc.initial_delta_m = delta_m0
    rtn = tm_visc.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    s_sep_visc = np.inf
    if rtn.status == -1:
        s_sep_visc = rtn.x_end

    # Veldman corrections applied
    tm_veld = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref,ue_veld],
                                     data_fits="Spline")

    delta_m0 = float(np.sqrt(0.075*nu_inf/tm_veld.du_e(s[0])))  # Moran's method
    tm_veld.initial_delta_m = delta_m0
    rtn = tm_veld.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    s_sep_veld = np.inf
    if rtn.status == -1:
        s_sep_veld = rtn.x_end

    tm_inv = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref, u_e_inv],
                                     data_fits="Spline")
    delta_m0 = float(np.sqrt(0.075*nu_inf/tm_inv.du_e(s[0])))  # Moran's method
    tm_inv.initial_delta_m = delta_m0
    rtn = tm_inv.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    s_sep_inv = np.inf
    if rtn.status == -1:
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

    # Veldman versions
    s_ref_veld = s_ref[s_ref < s_sep_veld]
    delta_d_ref_veld = delta_d_ref[s_ref < s_sep_veld]
    delta_m_ref_veld = delta_m_ref[s_ref < s_sep_veld]
    shape_d_ref_veld = shape_d_ref[s_ref < s_sep_veld]
    c_f_ref_veld = c_f_ref[s_ref < s_sep_veld]

    s_visc = np.linspace(s_ref[0], min(s_ref[-1], s_sep_visc), 101)
    delta_d_visc = tm_visc.delta_d(s_visc)
    delta_m_visc = tm_visc.delta_m(s_visc)
    shape_d_visc = tm_visc.shape_d(s_visc)
    c_f_visc = 2*tm_visc.tau_w(s_visc, rho_inf)/(rho_inf*u_inf**2)
    v_e_visc = tm_visc.v_e(s_visc)
    du_e_visc = tm_visc.du_e(s_visc)
    d2u_e_visc = tm_visc.d2u_e(s_visc)

    s_inv = np.linspace(s_ref[0], min(s_ref[-1], s_sep_inv), 101)
    delta_d_inv = tm_inv.delta_d(s_inv)
    delta_m_inv = tm_inv.delta_m(s_inv)
    shape_d_inv = tm_inv.shape_d(s_inv)
    c_f_inv = 2*tm_inv.tau_w(s_inv, rho_inf)/(rho_inf*u_inf**2)
    v_e_inv = tm_inv.v_e(s_inv)
    du_e_inv = tm_inv.du_e(s_inv)
    d2u_e_inv = tm_inv.d2u_e(s_inv)

    # Veldman versions
    s_veld = np.linspace(s_ref[0], min(s_ref[-1], s_sep_veld), 101)
    delta_d_veld = tm_veld.delta_d(s_veld)
    delta_m_veld = tm_veld.delta_m(s_veld)
    shape_d_veld = tm_veld.shape_d(s_veld)
    c_f_veld = 2*tm_veld.tau_w(s_veld, rho_inf)/(rho_inf*u_inf**2)
    v_e_veld = tm_veld.v_e(s_veld)
    du_e_veld = tm_veld.du_e(s_veld)
    d2u_e_veld = tm_veld.d2u_e(s_veld)

    # Plot results
    # pylint: disable=duplicate-code
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(40)

    #gs = GridSpec(6, 2, figure=fig)
    gs = GridSpec(6, 1, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0])
    #axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1])
    #axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_shape_d = fig.add_subplot(gs[2])
    #axis_shape_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3])
    #axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_u_e = fig.add_subplot(gs[4])
    #axis_du_e = fig.add_subplot(gs[4, 1])
    #axis_d2u_e = fig.add_subplot(gs[5, 0])
    axis_v_e = fig.add_subplot(gs[5])

    ref_color = "#54585A"
    ref_label = "XFoil"
    #thwaites_visc_color = "#5CB8B2"
    #thwaites_visc_label = "Thwaites (Viscous $U_e$)"
    thwaites_inv_color = "#FF6A39"
    thwaites_inv_label = "Thwaites (Inviscid $U_e$)"
    thwaites_veld_color = "#5CB8B2"
    thwaites_veld_label = "Thwaites (Veldman $U_e$)"
    # pylint: enable=duplicate-code

    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(s_ref/c, delta_d_ref/c, color=ref_color,
                        label=ref_label, linestyle="--")
    thwaites_veld_curve = ax.plot(s_veld/c, delta_d_veld/c,
                                  color=thwaites_veld_color,
                                  label=thwaites_veld_label)
    thwaites_inv_curve = ax.plot(s_inv/c, delta_d_inv/c,
                                 color=thwaites_inv_color,
                                 label=thwaites_inv_label)
    _ = ax.set_ylim(0, 0.1)
    _ = ax.set_ylabel(r"$\delta^*/c$",fontsize=15,rotation=0,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    #ax = axis_delta_d_diff
    #_ = ax.plot(s_ref_visc/c,
    #            np.abs(1-tm_visc.delta_d(s_ref_visc)/delta_d_ref_visc),
    #            color=thwaites_visc_color)
    #_ = ax.plot(s_ref_inv/c,
    #            np.abs(1-tm_inv.delta_d(s_ref_inv)/delta_d_ref_inv),
    #            color=thwaites_inv_color)
    #_ = ax.set_ylabel("Relative Difference")
    #_ = ax.set_ylim((1e-3,1))
    #ax.set_yscale('log')
    #ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    _ = ax.plot(s_ref/c, delta_m_ref/c, color=ref_color, linestyle="--")
    _ = ax.plot(s_veld/c, delta_m_veld/c, color=thwaites_veld_color)
    _ = ax.plot(s_inv/c, delta_m_inv/c, color=thwaites_inv_color)
    _ = ax.set_ylim(0, 0.025)
    _ = ax.set_ylabel(r"$\theta/c$",fontsize=15,rotation=0,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    #ax = axis_delta_m_diff
    #_ = ax.plot(s_ref_visc/c,
    #            np.abs(1-tm_visc.delta_m(s_ref_visc)/delta_m_ref_visc),
    #            color=thwaites_visc_color)
    #_ = ax.plot(s_ref_inv/c,
    #            np.abs(1-tm_inv.delta_m(s_ref_inv)/delta_m_ref_inv),
    #            color=thwaites_inv_color)
    #_ = ax.set_ylabel("Relative Difference")
    #_ = ax.set_ylim((1e-5,1))
    #ax.set_yscale('log')
    #ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    _ = ax.plot(s_ref/c, shape_d_ref, color=ref_color, linestyle="--")
    _ = ax.plot(s_veld/c, shape_d_veld, color=thwaites_veld_color)
    _ = ax.plot(s_inv/c, shape_d_inv, color=thwaites_inv_color)
    _ = ax.set_ylim(2.2, 3.6)
    _ = ax.set_ylabel(r"$H$",fontsize=15,rotation=0,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    #ax = axis_shape_d_diff
    #_ = ax.plot(s_ref_visc/c,
    #            np.abs(1-tm_visc.shape_d(s_ref_visc)/shape_d_ref_visc),
    #            color=thwaites_visc_color)
    #_ = ax.plot(s_ref_inv/c,
    #            np.abs(1-tm_inv.shape_d(s_ref_inv)/shape_d_ref_inv),
    #            color=thwaites_inv_color)
    #_ = ax.set_ylabel("Relative Difference")
    #_ = ax.set_ylim((1e-3,1))
    #ax.set_yscale('log')
    #ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    _ = ax.plot(s_ref/c, c_f_ref, color=ref_color, linestyle="--")
    _ = ax.plot(s_veld/c, c_f_veld, color=thwaites_veld_color)
    _ = ax.plot(s_inv/c, c_f_inv, color=thwaites_inv_color)
    _ = ax.set_ylabel(r"$c_f$",fontsize=15,rotation=0,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    #ax = axis_c_f_diff
    #_ = ax.plot(s_ref_visc/c,
    #            np.abs(1-2*tm_visc.tau_w(s_ref_visc,
    #                                     rho_inf)/(rho_inf
    #                                               *u_inf**2)/c_f_ref_visc),
    #            color=thwaites_visc_color)
    #_ = ax.plot(s_ref_inv/c,
    #            np.abs(1-2*tm_inv.tau_w(s_ref_inv,
    #                                    rho_inf)/(rho_inf
    #                                              *u_inf**2)/c_f_ref_inv),
    #            color=thwaites_inv_color)
    #_ = ax.set_ylabel("Relative Difference")
    #_ = ax.set_ylim((1e-4,1))
    #ax.set_yscale('log')
    #ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_u_e
    _ = ax.plot(s_ref/c, ue_veld/u_inf, color=thwaites_veld_color)
    _ = ax.plot(s_ref/c, u_e_inv/u_inf, color=thwaites_inv_color)
    #_ = ax.set_ylim(0, 1.1)
    _ = ax.set_ylabel(r"$u_e/u_\infty$",fontsize=15,rotation=0,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    #ax = axis_du_e
    #_ = ax.plot(s_visc/c, du_e_visc, color=thwaites_visc_color)
    #_ = ax.plot(s_inv/c, du_e_inv, color=thwaites_inv_color)
    #_ = ax.set_ylim(-2, 0.5)
    #_ = ax.set_xlabel(r"$x/c$")
    #_ = ax.set_ylabel(r"d$U_e/$d$x$ (1/s)")
    #ax.grid(True)

    ## Transpiration velocity in 5,:
    #ax = axis_d2u_e
    #_ = ax.plot(s_visc/c, d2u_e_visc, color=thwaites_visc_color)
    #_ = ax.plot(s_inv/c, d2u_e_inv, color=thwaites_inv_color)
    #_ = ax.set_ylim(-5, 5)
    #_ = ax.set_xlabel(r"$x/c$")
    #_ = ax.set_ylabel(r"d$^2U_e/$d$x^2$ (1/(m$\cdot$s)")
    #ax.grid(True)

    ax = axis_v_e
    _ = ax.plot(s_veld/c, v_e_veld/u_inf, color=thwaites_veld_color)
    _ = ax.plot(s_inv/c, v_e_inv/u_inf, color=thwaites_inv_color)
    _ = ax.set_ylim(0, 0.1)
    _ = ax.set_xlabel(r"$x/c$")
    _ = ax.set_ylabel(r"$V_e/U_e$")
    ax.grid(True)

    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    _ = fig.legend(handles=[ref_curve[0], thwaites_veld_curve[0],
                            thwaites_inv_curve[0]],
                   labels=[ref_label, thwaites_veld_label, thwaites_inv_label],
                   loc="upper center", bbox_to_anchor=(0.45, 0.03), ncol=4,
                   borderaxespad=0.1)
    plt.show()

# Create Hillbert Integral Function

def test_hillbert(n=11): # Runs a simple test of the Hillbert function
    s_ref = np.linspace(0,1,n) # Default 11 points from 0 to 1
    RanInd = 5 # Some random index
    zi = s_ref[RanInd] # Some random location within s_ref
    D_vec = zi*s_ref - .5*s_ref**2
    Di_sp = D_vec[RanInd] # The specific Di value
    # uncomment for specifics
    #print('s_ref')
    #print(s_ref)
    #print('Point')
    #print(zi)
    #print('D values')
    #print(D_vec)
    #print('Di')
    #print(Di_sp)
    Hi_real = 1/np.pi # With the way the test function is declared, the result should be 
    # 1/pi*(zL - z0), which is just 1/pi for this domain
    # Hi = Cii*Di + sum(Cij*Dj)[0->i-1] + sum(Cij*Dj)[i+1->End]
    print('What we expect')
    print(Hi_real)
    Test = Hillbert(s_ref,D_vec)
    print('Result ?')
    print(Test[RanInd])
    return np.array([s_ref[1]-s_ref[0],Hi_real - Test[RanInd]])

def Hillbert(s_ref,Di,A = 0): # Performs the summation form of the Hillbert Integral
    #zi is the specific location to run the integral
    #s_ref is body streamwise position locations
    #A is the index to start from, default is 0 -> the very first point
    # Output of this function should be a vector of C * Di
    LEN = len(s_ref)
    C = np.zeros((LEN,LEN))
    E = np.zeros((LEN,LEN))
    #print(LEN)
    # Creating the E matrix
    for i in range(A, LEN):
        for j in range(A+1,LEN-1): # Ei0 = EiL = 0 pg 75
            if j == i:
                E[i,j] = (((s_ref[i+1]-s_ref[i])/(s_ref[i+1]-s_ref[i-1]))*np.log(abs((s_ref[i]-s_ref[i-1])/(s_ref[i]-s_ref[i+1])))+2)/(s_ref[i]-s_ref[i-1])
            elif j == i+1:
                E[i,j] = (((s_ref[i]-s_ref[i-1])/(s_ref[i+1]-s_ref[i-1]))*np.log(abs((s_ref[i]-s_ref[i-1])/(s_ref[i]-s_ref[i+1])))-2)/(s_ref[i+1]-s_ref[i])
            else:
                E[i,j] = np.log(abs((s_ref[i]-s_ref[j-1])/(s_ref[i]-s_ref[j])))/(s_ref[j]-s_ref[j-1])

    # Creating the C matrix
    for i in range(A,LEN):
        for j in range(A,LEN):
            if j == (LEN-1):
                C[i,j] = (1/np.pi)*(E[i,j])
            else:
                C[i,j] = (1/np.pi)*(E[i,j] - E[i,j+1])

    #for i in range(LEN): # Most likely redundant calculation
        #C[i,0] = 0

    return C @ Di

def Hillbert_C(s_ref,A=0):  # This only creates the Cij matrix
    #zi is the specific location to run the integral
    #s_ref is body streamwise position locations
    #A is the index to start from, default is 0 -> the very first point

    LEN = len(s_ref)
    C = np.zeros((LEN,LEN))
    E = np.zeros((LEN,LEN))
    #print(LEN)
    # Creating the E matrix
    for i in range(A, LEN):
        for j in range(A+1,LEN-1): # Ei0 = EiL = 0 pg 75
            if j == i:
                E[i,j] = (((s_ref[i+1]-s_ref[i])/(s_ref[i+1]-s_ref[i-1]))*np.log(abs((s_ref[i]-s_ref[i-1])/(s_ref[i]-s_ref[i+1])))+2)/(s_ref[i]-s_ref[i-1])
            elif j == i+1:
                E[i,j] = (((s_ref[i]-s_ref[i-1])/(s_ref[i+1]-s_ref[i-1]))*np.log(abs((s_ref[i]-s_ref[i-1])/(s_ref[i]-s_ref[i+1])))-2)/(s_ref[i+1]-s_ref[i])
            else:
                E[i,j] = np.log(abs((s_ref[i]-s_ref[j-1])/(s_ref[i]-s_ref[j])))/(s_ref[j]-s_ref[j-1])

    # Creating the C matrix
    for i in range(A,LEN):
        for j in range(A,LEN):
            if j == (LEN-1):
                C[i,j] = (1/np.pi)*(E[i,j])
            else:
                C[i,j] = (1/np.pi)*(E[i,j] - E[i,j+1])
    
    for i in range(LEN):
        C[i,0] = 0
    
    return C


if __name__ == "__main__":
    compare_xfoil_laminar()
