"""
Comparing Drela-Giles Laminar method solutions for flat plate case.

This example shows a comparison between various forms of Thwaites and the
Blasius solution to laminar flat plate boundary layer flows. 
"""
#import time
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.analytic import Blasius
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
#from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.initial_condition import ManualCondition
from ibl.typing import InputParam
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams['legend.loc'] = 'lower center'
plt.rcParams['legend.borderaxespad'] = -10
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 3.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.default"] = "rm"

#print(os.path.dirname(os.path.abspath(__file__)))

file_name = "Blasius_Comparison"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

def compare_blasius_solution() -> None:
    """Compare the various solutions to the Blasius solution."""
    # pylint: disable=too-many-locals, too-many-statements
    # Set flow parameters
    #start = time.time()
    u_inf = 10
    nu_inf = 1.45e-5
    rho_inf = 1.2
    c = 2
    npts = 101
    x = np.linspace(1e-6, c, npts)

    # Set up the velocity functions
    def u_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        return u_inf*np.ones_like(x)

    def du_e_fun(x: InputParam) -> npt.NDArray: #TODO no-go for ICs of F-S
        x = np.asarray(x)
        return np.zeros_like(x)
    
    def du_e_fun_dg(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        return .01 * np.exp(- (x / .01))

    def d2u_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        return np.zeros_like(x)

    # setup plot functions
    fig = plt.figure()
    #fig.set_figwidth(10)
    #fig.set_figheight(15)
    #gs = GridSpec(5, 2, figure=fig)
    gs = GridSpec(4, 2, figure=fig)
    axis_delta_d = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axis_delta_m = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axis_shape_d = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    axis_c_f = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    #axis_v_e = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    # extract the Blasius (exact) solution
    bs = Blasius(u_ref=u_inf, nu_ref=nu_inf)
    delta_d_exact = bs.delta_d(x)
    delta_m_exact = bs.delta_m(x)
    #delta_m_exact = bs.delta_m(x)
    
    c_f_exact = bs.tau_w(x, rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d_exact = bs.shape_d(x)
    v_e_exact = bs.v_e(x)

    exact_color = "black"
    curve_handles = [axis_delta_d[0].plot(x/c, delta_d_exact/c,
                                          color=exact_color,linewidth=4.,linestyle='--')[0]]
    _ = axis_delta_m[0].plot(x/c, delta_m_exact/c, color=exact_color,linewidth=4.,linestyle='--')
    _ = axis_shape_d[0].plot(x/c, shape_d_exact, color=exact_color,linewidth=4.,linestyle='--')
    _ = axis_c_f[0].plot(x/c, c_f_exact, color=exact_color,linewidth=4.)
    #_ = axis_v_e[0].plot(x/c, v_e_exact/u_inf, color=exact_color,linewidth=4.,linestyle='--')

    # create the various models
    colors = ["#A4D65E","#BB00FF"]
    #color = "green"
    labels = ["Blasius", "D-G, Orig", "D-G, Modern"]

    start = 0
    model = DrelaGilesLaminarMOD(nu = nu_inf, U_e= u_e_fun, dU_edx= du_e_fun, ic = ManualCondition(delta_d=delta_d_exact[start], delta_m=delta_m_exact[start],delta_k=0),show_prog=False,src=False) #src = False fits better
    rtn = model.solve(x0=x[start], x_end=x[-1])
    if not rtn.success:
        print("Could not get solution for method: " + rtn.message)
    delta_d = model.delta_d(x)
    delta_m = model.delta_m(x)
    c_f = model.tau_w(x, rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d = model.shape_d(x)
    v_e = model.v_e(x)

    model2 = DrelaGilesLaminarMOD(nu = nu_inf, U_e= u_e_fun, dU_edx= du_e_fun, ic = ManualCondition(delta_d=delta_d_exact[start], delta_m=delta_m_exact[start],delta_k=0),show_prog=False,src=True)
    rtn2 = model2.solve(x0=x[start], x_end=x[-1])
    if not rtn2.success:
        print("Could not get solution for method: " + rtn2.message)
    delta_d_src = model2.delta_d(x)
    delta_m_src = model2.delta_m(x)
    c_f_src = model2.tau_w(x, rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d_src = model2.shape_d(x)
    v_e_src = model2.v_e(x)

    curve_handles.append(axis_delta_d[0].plot(x/c, delta_d/c,
                                              color=colors[0])[0])
    _ = axis_delta_d[1].plot(x/c, np.abs(1-delta_d/delta_d_exact),
                             color=colors[0])
    _ = axis_delta_m[0].plot(x/c, delta_m/c, color="#BB00FF")
    _ = axis_delta_m[1].plot(x/c, np.abs(1-delta_m/delta_m_exact),
                             color="#BB00FF")
    _ = axis_shape_d[0].plot(x/c, shape_d, color='#612D00')
    _ = axis_shape_d[1].plot(x/c, np.abs(1-shape_d/shape_d_exact),
                             color='#612D00')
    _ = axis_c_f[0].plot(x/c, c_f, color="#818181")
    _ = axis_c_f[1].plot(x/c, np.abs(1-c_f/c_f_exact), color="#818181")
    #_ = axis_v_e[0].plot(x/c, v_e/u_inf, color=colors[0])
    #_ = axis_v_e[1].plot(x/c, np.abs(1-v_e/v_e_exact), color=colors[0])

    curve_handles.append(axis_delta_d[0].plot(x/c, delta_d_src/c,
                                              color=colors[0],linestyle='--')[0])
    _ = axis_delta_d[1].plot(x/c, np.abs(1-delta_d_src/delta_d_exact),
                             color=colors[0],linestyle='--')
    _ = axis_delta_m[0].plot(x/c, delta_m_src/c, color="#BB00FF",linestyle='--')
    _ = axis_delta_m[1].plot(x/c, np.abs(1-delta_m_src/delta_m_exact),
                             color="#BB00FF",linestyle='--')
    _ = axis_shape_d[0].plot(x/c, shape_d_src, color='#612D00',linestyle='--')
    _ = axis_shape_d[1].plot(x/c, np.abs(1-shape_d_src/shape_d_exact),
                             color='#612D00',linestyle='--')
    _ = axis_c_f[0].plot(x/c, c_f_src, color="#818181",linestyle='--')
    _ = axis_c_f[1].plot(x/c, np.abs(1-c_f_src/c_f_exact), color="#818181",linestyle='--')
    #_ = axis_v_e[0].plot(x/c, v_e_src/u_inf, color=colors[0],linestyle='--')
    #_ = axis_v_e[1].plot(x/c, np.abs(1-v_e_src/v_e_exact), color=colors[0],linestyle='--')

    # Displacement thickness in 0,:
    _ = axis_delta_d[0].set_ylim((0, 0.0015))
    temp = r"$\delta_d/c$"
    _ = axis_delta_d[0].set_ylabel(temp)
    #_ = axis_delta_d[0].set_ylabel(r"$\delta^*/c$")
    axis_delta_d[0].grid(True)

    _ = axis_delta_d[1].set_ylabel("Relative Error")
    #_ = axis_delta_d[1].set_ylim((1e-4,1))
    axis_delta_d[1].set_yscale('log')
    axis_delta_d[1].grid(True)

    # Momentum thickness in 1,:
    _ = axis_delta_m[0].set_ylim(0, 0.0006)
    _ = axis_delta_m[0].set_ylabel(r"$\delta_m/c$")
    #_ = axis_delta_m[0].set_ylabel(r"$\theta/c$")
    axis_delta_m[0].grid(True)

    _ = axis_delta_m[1].set_ylabel("Relative Error")
    #_ = axis_delta_m[1].set_ylim((1e-4,1))
    axis_delta_m[1].set_yscale('log')
    axis_delta_m[1].grid(True)

    # Displacement shape factor in 2,:
    _ = axis_shape_d[0].set_ylim((2.5, 2.7))
    _ = axis_shape_d[0].set_ylabel(r"$H_d$")
    #_ = axis_shape_d[0].set_ylabel(r"$H$")
    axis_shape_d[0].grid(True)

    _ = axis_shape_d[1].set_ylabel("Relative Error")
    #_ = axis_shape_d[1].set_ylim((1e-4,1))
    axis_shape_d[1].set_yscale('log')
    axis_shape_d[1].grid(True)

    # Skin friction coefficient in 3,:
    _ = axis_c_f[0].set_ylim((0, 0.01))
    _ = axis_c_f[0].set_ylabel(r"$c_f$")
    axis_c_f[0].grid(True)

    _ = axis_c_f[1].set_ylabel("Relative Error")
    #_ = axis_c_f[1].set_ylim((1e-4,1))
    axis_c_f[1].set_yscale('log')
    axis_c_f[1].grid(True)

    # Transpiration velocity in 4,:
    #_ = axis_v_e[0].set_ylim((0, 0.01))
    #_ = axis_v_e[0].set_xlabel(r"$s/c$")
    #_ = axis_v_e[0].set_ylabel(r"$v_e/u_{\infty}$")
    #axis_v_e[0].grid(True)

    #_ = axis_v_e[1].set_xlabel(r"$s/c$")
    #_ = axis_v_e[1].set_ylabel("Relative Error")
    #_ = axis_v_e[1].set_ylim((1e-4,1))
    #axis_v_e[1].set_yscale('log')
    #axis_v_e[1].grid(True)

    _ = fig.subplots_adjust(left=0.15, wspace=0.5, hspace=.5, top=.95)
    _ = fig.legend(handles=curve_handles, labels=labels, loc="upper center",
                   bbox_to_anchor=(.51, 0.06), ncol=3, borderaxespad=.8)
    #fig.tight_layout()
    fig.savefig(file_name+'\\'+'Blasius_Comp.png')
    plt.show()

if __name__ == "__main__":
    compare_blasius_solution()
