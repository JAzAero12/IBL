"""
Comparing Drela-Giles Laminar method solutions for wedge flow case.

This example shows a comparison between various forms of Thwaites and the
Falkner-Skan solution to laminar flat plate boundary layer wedge flows at angle
of pi/4.
"""

# pylint: disable=duplicate-code
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.analytic import FalknerSkan
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
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

file_name = "Falkner_Skan_Comparison"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

#TODO look into why shape_d does that
def compare_stagnation_solution() -> None:
    """Compare the various solutions to the Falkner-Skan solution."""
    # pylint: disable=too-many-locals, too-many-statements
    # Set flow parameters
    u_inf = 10
    m = 1/3
    nu_inf = 1.45e-5
    rho_inf = 1.2
    c = 2
    npts = 201
    x = np.linspace(1e-6, c, npts)
    #x = np.linspace(1, c, npts)

    # Set up the velocity functions
    def u_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        return u_inf*x**m

    def du_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        if m == 0:
            return np.zeros_like(x)
        return m*u_inf*x**(m-1)

    def d2u_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        if m in (0, 1):
            return np.zeros_like(x)
        return m*(m-1)*u_inf*x**(m-2)

    # setup plot functions
    fig = plt.figure()
    #fig.set_figwidth(10)
    #fig.set_figheight(15)
    gs = GridSpec(5, 2, figure=fig)
    axis_delta_d = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axis_delta_m = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axis_shape_d = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    axis_c_f = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    axis_v_e = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    # extract the Falkner-Skan (exact) solution
    fs = FalknerSkan(beta=0.5, u_ref=u_inf, nu_ref=nu_inf)
    delta_d_exact = fs.delta_d(x)
    delta_m_exact = fs.delta_m(x)
    c_f_exact = fs.tau_w(x, rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d_exact = fs.shape_d(x)
    delta_k_exact = fs.delta_k(x)
    v_e_exact = fs.v_e(x)

    exact_color = "black"
    curve_handles = [axis_delta_d[0].plot(x/c, delta_d_exact/c,
                                          color=exact_color,linewidth=4.,linestyle='--')[0]]
    _ = axis_delta_m[0].plot(x/c, delta_m_exact/c, color=exact_color,linewidth=4.,linestyle='--')
    _ = axis_shape_d[0].plot(x/c, shape_d_exact, color=exact_color,linewidth=4.,linestyle='--')
    _ = axis_c_f[0].plot(x/c, c_f_exact, color=exact_color,linewidth=4.,linestyle='--')
    _ = axis_v_e[0].plot(x/c, v_e_exact/u_inf, color=exact_color,linewidth=4.,linestyle='--')


    labels = ["Falkner-Skan", "D-G, Old", "D-G, Modern"]
    model = DrelaGilesLaminarMOD(nu=nu_inf,U_e=u_e_fun,dU_edx=du_e_fun,ic = ManualCondition(delta_d=delta_d_exact[0], delta_m=delta_m_exact[0], delta_k=delta_k_exact[0]),show_prog=False,src=False) #src = True fits better
    tmp = 0
    rtn = model.solve(x0=x[tmp], x_end=x[-1])
    if not rtn.success:
        print("Could not get solution for Drela-Giles method: " + rtn.message)
        print(x[tmp])
        print(rtn.x_end)
        return

    #color = '#154734'
    color = '#A4D65E'
    delta_d = model.delta_d(x[tmp:])
    delta_m = model.delta_m(x[tmp:])
    c_f = model.tau_w(x[tmp:], rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d = model.shape_d(x[tmp:])
    v_e = model.v_e(x[tmp:])

    model2 = DrelaGilesLaminarMOD(nu=nu_inf,U_e=u_e_fun,dU_edx=du_e_fun,ic = ManualCondition(delta_d=delta_d_exact[0], delta_m=delta_m_exact[0], delta_k=delta_k_exact[0]),show_prog=False,src=True) #src = True fits better
    rtn2 = model2.solve(x0=x[tmp], x_end=x[-1])
    if not rtn2.success:
        print("Could not get solution for Drela-Giles method: " + rtn2.message)
        print(x[tmp])
        print(rtn2.x_end)
        return

    color2 = '#A4D65E'
    delta_d_src = model2.delta_d(x[tmp:])
    delta_m_src = model2.delta_m(x[tmp:])
    c_f_src = model2.tau_w(x[tmp:], rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d_src = model2.shape_d(x[tmp:])
    v_e_src = model2.v_e(x[tmp:])


    curve_handles.append(axis_delta_d[0].plot(x[tmp:]/c, delta_d/c,
                                              color=color)[0])
    _ = axis_delta_d[1].plot(x[tmp:]/c, np.abs(1-delta_d/delta_d_exact[tmp:]),
                             color=color)
    _ = axis_delta_m[0].plot(x[tmp:]/c, delta_m/c, color='#154734')
    _ = axis_delta_m[1].plot(x[tmp:]/c, np.abs(1-delta_m/delta_m_exact[tmp:]),
                             color='#154734')
    _ = axis_shape_d[0].plot(x[tmp:]/c, shape_d, color=color)
    _ = axis_shape_d[1].plot(x[tmp:]/c, np.abs(1-shape_d/shape_d_exact[tmp:]),
                             color=color)
    _ = axis_c_f[0].plot(x[tmp:]/c, c_f, color="#F8E08E")
    _ = axis_c_f[1].plot(x[tmp:]/c, np.abs(1-c_f/c_f_exact[tmp:]), color="#F8E08E")
    _ = axis_v_e[0].plot(x[tmp:]/c, v_e/u_inf, color=color)
    _ = axis_v_e[1].plot(x[tmp:]/c, np.abs(1-v_e/v_e_exact[tmp:]), color=color)

    curve_handles.append(axis_delta_d[0].plot(x[tmp:]/c, delta_d_src/c,
                                              color=color2,linestyle='--')[0])
    _ = axis_delta_d[1].plot(x[tmp:]/c, np.abs(1-delta_d_src/delta_d_exact[tmp:]),
                             color=color2,linestyle='--')
    _ = axis_delta_m[0].plot(x[tmp:]/c, delta_m_src/c, color='#154734',linestyle='--')
    _ = axis_delta_m[1].plot(x[tmp:]/c, np.abs(1-delta_m_src/delta_m_exact[tmp:]),
                             color='#154734',linestyle='--')
    _ = axis_shape_d[0].plot(x[tmp:]/c, shape_d_src, color=color2,linestyle='--')
    _ = axis_shape_d[1].plot(x[tmp:]/c, np.abs(1-shape_d_src/shape_d_exact[tmp:]),
                             color=color2,linestyle='--')
    _ = axis_c_f[0].plot(x[tmp:]/c, c_f_src, color="#F8E08E",linestyle='--')
    _ = axis_c_f[1].plot(x[tmp:]/c, np.abs(1-c_f_src/c_f_exact[tmp:]), color="#F8E08E",linestyle='--')
    _ = axis_v_e[0].plot(x[tmp:]/c, v_e_src/u_inf, color=color2,linestyle='--')
    _ = axis_v_e[1].plot(x[tmp:]/c, np.abs(1-v_e_src/v_e_exact[tmp:]), color=color2,linestyle='--')

    # Displacement thickness in 0,:
    _ = axis_delta_d[0].set_ylim((0, 0.0008))
    _ = axis_delta_d[0].set_ylabel(r"$\delta_d/c$")
    #_ = axis_delta_d[0].set_ylabel(r"$\delta^*/c$")
    axis_delta_d[0].grid(True)

    _ = axis_delta_d[1].set_ylabel("Relative Error")
    _ = axis_delta_d[1].set_ylim((1e-4,1))
    axis_delta_d[1].set_yscale('log')
    axis_delta_d[1].grid(True)

    # Momentum thickness in 1,:
    _ = axis_delta_m[0].set_ylim((0, 0.0004))
    _ = axis_delta_m[0].set_ylabel(r"$\delta_m/c$")
    #_ = axis_delta_m[0].set_ylabel(r"$\theta/c$")
    axis_delta_m[0].grid(True)

    _ = axis_delta_m[1].set_ylabel("Relative Error")
    _ = axis_delta_m[1].set_ylim((1e-4,1))
    axis_delta_m[1].set_yscale('log')
    axis_delta_m[1].grid(True)

    # Displacement shape factor in 2,:
    _ = axis_shape_d[0].set_ylim((2.2, 2.5))
    _ = axis_shape_d[0].set_ylabel(r"$H_d$")
    #_ = axis_shape_d[0].set_ylabel(r"$H$")
    axis_shape_d[0].grid(True)

    _ = axis_shape_d[1].set_ylabel("Relative Error")
    _ = axis_shape_d[1].set_ylim((1e-4,1))
    axis_shape_d[1].set_yscale('log')
    axis_shape_d[1].grid(True)

    # Skin friction coefficient in 3,:
    _ = axis_c_f[0].set_ylim((0.001, 0.002))
    _ = axis_c_f[0].set_ylabel(r"$c_f$")
    axis_c_f[0].grid(True)

    _ = axis_c_f[1].set_ylabel("Relative Error")
    _ = axis_c_f[1].set_ylim((1e-4,1))
    axis_c_f[1].set_yscale('log')
    axis_c_f[1].grid(True)

    # Transpiration velocity in 4,:
    _ = axis_v_e[0].set_ylim((0, 0.01))
    _ = axis_v_e[0].set_xlabel(r"$s/c$")
    _ = axis_v_e[0].set_ylabel(r"$v_e/u_{inf}$")
    axis_v_e[0].grid(True)

    _ = axis_v_e[1].set_xlabel(r"$s/c$")
    _ = axis_v_e[1].set_ylabel("Relative Error")
    _ = axis_v_e[1].set_ylim((1e-4,1))
    axis_v_e[1].set_yscale('log')
    axis_v_e[1].grid(True)

    _ = fig.subplots_adjust(left=0.15, wspace=0.5, hspace=.5, top=.95)
    _ = fig.legend(handles=curve_handles, labels=labels, loc="upper center",
                   bbox_to_anchor=(.51, 0.06), ncol=3, borderaxespad=.8)
    #fig.tight_layout()
    fig.savefig(file_name+'\\'+'Falkner_Skan_Comp.png')
    plt.show()
    pass


if __name__ == "__main__":
    compare_stagnation_solution()
