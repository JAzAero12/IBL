# pylint: disable=too-many-statements,too-many-locals

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.head_method import HeadMethod
from ibl.interaction_law import interaction_law
from ibl.initial_condition import ManualCondition
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
from ibl.drela_giles_turbulent_mod import DrelaGilesTurbulentMOD
from ibl.transition_coupler import transition_coupler
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['legend.loc'] = 'lower center'
plt.rcParams['legend.borderaxespad'] = -6.
plt.rcParams["axes.grid"] = True

transition_loc = 0.5
re_inf = 1000000

chord = 1.
u_inf = 20.
nu_inf = chord*u_inf/re_inf
rho_inf = 1.225

s_ref = np.array([0.00083,0.00258,0.00442,0.00639,0.00851,0.01081,0.01333,
                    0.01612,0.01926,0.02283,0.02694,0.03175,0.03743,0.04421,
                    0.05229,0.06178,0.07263,0.08462,0.09745,0.11089,0.12473,
                    0.13886,0.1532,0.1677,0.18233,0.19707,0.21189,0.22679,
                    0.24176,0.25678,0.27185,0.28696,0.30212,0.31731,0.33254,
                    0.34779,0.36307,0.37838,0.39371,0.40907,0.42444,0.43983,
                    0.45524,0.47067,0.48611,0.50157,0.51703,0.53251,0.54801,
                    0.56351,0.57902,0.59454,0.61007,0.6256,0.64114,0.65669,
                    0.67224,0.6878,0.70336,0.71893,0.7345,0.75007,0.76564,
                    0.78121,0.79679,0.81236,0.82793,0.8435,0.85908,0.87464,
                    0.89021,0.90576,0.9213,0.9368,0.95221,0.96736,0.9819,
                    0.9952,1.00653,1.01451])

u_e_per_u_inf = np.array([0.09563,0.28758,0.46594,0.61837,0.74137,0.8374,0.9115,
                    0.96868,1.01312,1.04796,1.07555,1.09753,1.11505,1.12891,
                    1.13964,1.14763,1.15324,1.15687,1.15894,1.15983,1.15983,
                    1.15917,1.158,1.15644,1.15457,1.15245,1.15012,1.14762,
                    1.14499,1.14224,1.1394,1.13648,1.13351,1.13049,1.12743,
                    1.12435,1.12126,1.11817,1.1151,1.11206,1.10906,1.10615,
                    1.10335,1.10075,1.09849,1.09692,1.09516,1.08546,1.07588,
                    1.07055,1.06765,1.0654,1.06315,1.06073,1.05814,1.05539,
                    1.05251,1.04952,1.04643,1.04323,1.03993,1.03652,1.03299,
                    1.02933,1.02553,1.02154,1.01736,1.01294,1.00823,1.00318,
                    0.99771,0.99172,0.98507,0.97757,0.96899,0.959,0.9473,
                    0.93393,0.91917,0.90844])

u_e_visc = u_e_per_u_inf*u_inf
fig, velplot = plt.subplots()
velplot.plot(s_ref,u_e_visc,color='#154734')
velplot.set_title(r'u_e')

T_air= 288.15 
R_air = 287 
gamma = 1.4
m_e_visc = u_e_visc/np.sqrt(gamma*R_air*T_air)

delta_m_XF = np.array([0.0000272,0.000028,0.0000298,0.0000326,0.0000364,0.0000409,0.000046,
                    0.0000518,0.0000582,0.0000653,0.0000731,0.0000817,0.0000914,0.0001022,
                    0.0001143,0.0001274,0.0001414,0.0001558,0.0001703,0.0001846,0.0001985,
                    0.0002122,0.0002255,0.0002385,0.0002512,0.0002637,0.0002759,0.000288,
                    0.0002999,0.0003116,0.0003232,0.0003347,0.0003461,0.0003573,0.0003685,
                    0.0003796,0.0003906,0.0004015,0.0004123,0.000423,0.0004335,0.0004439,
                    0.000454,0.0004638,0.0004729,0.0004808,0.0004894,0.0005165,0.0005481,
                    0.0005783,0.0006098,0.0006441,0.0006806,0.0007186,0.0007576,0.0007971,
                    0.0008371,0.0008775,0.0009181,0.000959,0.0010003,0.0010421,0.0010843,
                    0.0011273,0.001171,0.0012156,0.0012614,0.0013087,0.0013577,0.0014089,
                    0.0014629,0.0015204,0.0015825,0.0016507,0.0017273,0.0018149,0.001917,
                    0.0020351,0.0021692,0.0022727])

delta_d_XF = np.array([0.0000607,0.0000625,0.000067,0.0000739,0.0000832,0.0000944,0.0001073,
                    0.000122,0.0001383,0.0001565,0.0001767,0.0001992,0.0002245,0.0002528,
                    0.0002845,0.0003192,0.0003562,0.0003945,0.0004332,0.0004716,0.0005095,
                    0.0005467,0.0005833,0.0006193,0.000655,0.0006903,0.0007253,0.0007602,
                    0.0007949,0.0008296,0.0008643,0.000899,0.0009337,0.0009685,0.0010034,
                    0.0010384,0.0010735,0.0011086,0.0011436,0.0011783,0.0012126,0.001246,
                    0.0012777,0.0013063,0.0013292,0.0013395,0.0013139,0.001196,0.0010658,
                    0.0009886,0.0009651,0.0009762,0.0010075,0.0010503,0.0010998,0.0011534,
                    0.0012096,0.0012675,0.0013265,0.0013865,0.0014474,0.0015091,0.0015717,
                    0.0016354,0.0017004,0.001767,0.0018356,0.0019066,0.0019807,0.0020588,
                    0.0021419,0.0022317,0.0023301,0.0024406,0.0025676,0.0027179,0.0029003,
                    0.0031217,0.0033887,0.003601])

delta_k_XF = np.array([0.0000441,0.0000453,0.0000482,0.0000527,0.0000586,0.0000657,0.0000739,
                    0.000083,0.0000931,0.0001042,0.0001165,0.0001301,0.0001453,0.0001624,
                    0.0001813,0.0002019,0.0002238,0.0002464,0.0002691,0.0002914,0.0003133,
                    0.0003346,0.0003554,0.0003756,0.0003954,0.0004148,0.0004338,0.0004525,
                    0.0004708,0.000489,0.0005068,0.0005245,0.000542,0.0005593,0.0005764,
                    0.0005933,0.0006101,0.0006268,0.0006433,0.0006596,0.0006757,0.0006916,
                    0.0007072,0.0007223,0.0007367,0.0007497,0.0007547,0.0008145,0.0008955,
                    0.000975,0.0010495,0.0011216,0.0011926,0.0012632,0.0013336,0.0014038,
                    0.0014739,0.0015442,0.0016146,0.0016855,0.0017568,0.0018288,0.0019017,
                    0.0019756,0.0020508,0.0021276,0.0022062,0.0022872,0.0023711,0.0024584,
                    0.0025502,0.0026475,0.002752,0.002866,0.0029926,0.0031359,0.0033002,
                    0.0034866,0.0036936,0.0038528])

shape_d_XF = np.array([2.2295,2.2342,2.2487,2.2655,2.2877,2.3097,2.3329,
                        2.3553,2.3771,2.3981,2.4181,2.4372,2.4556,2.4729,
                        2.4895,2.5049,2.5191,2.5321,2.544,2.5552,2.5659,
                        2.5764,2.5867,2.597,2.6074,2.6179,2.6286,2.6396,
                        2.6508,2.6622,2.6739,2.6859,2.6981,2.7105,2.7231,
                        2.7358,2.7486,2.7613,2.7738,2.7859,2.7972,2.8069,
                        2.814,2.8166,2.8105,2.7862,2.6847,2.3154,1.9443,
                        1.7095,1.5825,1.5157,1.4803,1.4615,1.4517,1.4469,
                        1.4449,1.4445,1.4449,1.4458,1.4469,1.4482,1.4495,
                        1.4508,1.4521,1.4536,1.4551,1.4569,1.4589,1.4613,
                        1.4642,1.4678,1.4724,1.4785,1.4865,1.4976,1.5129,
                        1.5339,1.5622,1.5844])

shape_k_XF = np.array([1.6211,1.6203,1.618,1.6154,1.6121,1.6089,1.6056,
                        1.6025,1.5996,1.5969,1.5944,1.5921,1.5899,1.5879,
                        1.586,1.5843,1.5828,1.5814,1.5802,1.5791,1.578,
                        1.577,1.576,1.575,1.574,1.573,1.572,1.5711,
                        1.5701,1.5691,1.5681,1.5671,1.5661,1.5651,
                        1.5641,1.5631,1.5622,1.5612,1.5603,1.5595,
                        1.5587,1.558,1.5575,1.5574,1.5578,1.5595,1.5421,
                        1.5768,1.6336,1.6859,1.721,1.7414,1.7524,1.7579,
                        1.7603,1.761,1.7607,1.7598,1.7587,1.7575,1.7562,
                        1.755,1.7538,1.7526,1.7514,1.7502,1.749,1.7477,
                        1.7464,1.7449,1.7433,1.7413,1.739,1.7362,1.7326,
                        1.7278,1.7215,1.7133,1.7027,1.6953])


c_f_XF = np.array([0.0025219,0.0073329,0.0109315,0.0129493,0.0135173,0.0131851,0.0123411,
                0.0112942,0.0101964,0.0091312,0.0081347,0.0072203,0.006387,0.0056379,
                0.0049709,0.0043888,0.0038919,0.0034755,0.0031296,0.0028415,0.0025993,
                0.0023928,0.0022145,0.0020584,0.0019202,0.0017966,0.0016852,0.0015839,
                0.0014914,0.0014063,0.0013277,0.0012549,0.0011872,0.0011241,0.0010652,0.0010102,
                0.0009588,0.0009109,0.0008664,0.0008253,0.0007881,0.0007551,0.0007275,0.0007074,
                0.0006995,0.0007159,0.0008285,0.0013412,0.0023959,0.0034378,0.0041444,
                0.0045275,0.0046959,0.0047398,0.0047152,0.0046544,0.0045756,0.0044887,
                0.0043991,0.0043098,0.0042221,0.0041366,0.0040533,0.0039719,0.0038919,
                0.0038128,0.0037338,0.0036542,0.0035729,0.003489,0.003401,0.0033072,0.0032055,
                0.0030928,0.0029652,0.0028177,0.0026452,0.0024479,0.0022285,0.0020727])

c_D_XF = np.array([0.0000693,0.0006081,0.0014835,0.0023602,0.0030031,0.0033661,0.0034949,
                0.0034645,0.0033351,0.0031496,0.0029351,0.0027087,0.00248,0.0022567,
                0.0020442,0.0018481,0.001673,0.0015208,0.0013909,0.0012804,0.0011861,
                0.001105,0.0010344,0.0009724,0.0009174,0.0008682,0.0008239,0.0007836,
                0.000747,0.0007133,0.0006824,0.0006538,0.0006272,0.0006026,0.0005796,
                0.0005582,0.0005382,0.0005194,0.0005019,0.0004855,0.0004703,0.0004561,
                0.0004432,0.0004316,0.0004219,0.0004156,0.0010821,0.0021424,0.0026478,
                0.0026685,0.002593,0.0025123,0.0024344,0.0023604,0.0022911,0.0022269,
                0.0021678,0.0021134,0.0020634,0.002017,0.0019737,0.0019329,0.0018943,
                0.0018573,0.0018215,0.0017865,0.0017521,0.0017178,0.0016832,0.0016481,
                0.0016119,0.0015743,0.0015345,0.001492,0.0014457,0.0013949,0.0013391,
                0.0012799,0.00122,0.0011803])

re_delta_m_XF = np.array([2.602,8.04,13.884,20.18,26.958,34.209,41.938,
                        50.167,58.952,68.383,78.583,89.708,101.939,115.427,
                        130.239,146.245,163.082,180.264,197.351,214.066,230.28,
                        245.959,261.119,275.798,290.037,303.876,317.355,330.506,
                        343.36,355.942,368.274,380.376,392.265,403.954,415.455,
                        426.777,437.926,448.905,459.715,470.348,480.79,491.013,
                        500.964,510.54,519.523,527.353,535.985,560.682,589.736,
                        619.101,651.084,686.2,723.561,762.244,801.617,841.304,
                        881.101,920.911,960.699,1000.471,1040.26,1080.117,1120.109,
                        1160.318,1200.842,1241.799,1283.332,1325.617,1368.873,1413.385,
                        1459.524,1507.79,1558.871,1613.726,1673.688,1740.489,1815.988,
                        1900.604,1993.906,2064.618])

n_tild_c_tau = np.array([0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,
                    0,0.0087552,0.0701571,0.1870706,0.3297438,0.4744924,0.6209138,
                    0.7693549,0.9201447,1.0735926,1.2299853,1.3895837,1.5526198,1.7192913,
                    1.889754,2.0641102,2.2423916,2.4245322,2.6103261,2.7993594,2.9908958,
                    3.1836761,3.3755435,3.5626715,3.7377159,0.0257034,0.0416422,0.0490607,
                    0.0496522,0.0480195,0.0461635,0.0445978,0.0433715,0.0424294,0.0417075,
                    0.0411532,0.0407262,0.0403965,0.0401417,0.0399452,0.0397946,0.0396806,
                    0.0395966,0.0395377,0.0395006,0.0394832,0.0394847,0.0395051,0.0395456,
                    0.0396087,0.0396982,0.0398203,0.0399841,0.0402028,0.0404953,0.0408847,
                    0.0413841,0.0419969,0.0424731])


c_tau_eq_XF = np.array([0.0788606,0.0710830,0.0604549,0.0513836,0.0453720,0.0418340,
                    0.0398678,0.0388201,0.0382957,0.0380672,0.0380048,0.0380341,0.0381120,
                    0.0382140,0.0383264,0.0384418,0.0385572,0.0386717,0.0387864,0.0389033,
                    0.0390256,0.0391573,0.0393034,0.0394704,0.0396668,0.0399039,0.0401974,
                    0.0405704,0.0410567,0.0417059,0.0425853,0.0437494,0.0452514,0.0464124,
                    ])

ic = ManualCondition(delta_d=delta_d_XF[0],delta_m=delta_m_XF[0],delta_k=delta_k_XF[0])

dg_laminar = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[s_ref,u_e_visc],ic=ic,src=True)

turbstart_idx = 46 #appears to be where XFOIL transitions
turbstart = s_ref[turbstart_idx]
turbulent_model = DrelaGilesTurbulentMOD(nu = nu_inf, U_e= [s_ref,u_e_visc],show_prog=False,src=True)
turbulent_model.initial_delta_m = delta_m_XF[turbstart_idx]
turbulent_model.initial_shape_d = delta_d_XF[turbstart_idx]/delta_m_XF[turbstart_idx]
#turbulent_model.initial_shape_k = delta_k_XF[turbstart_idx]/delta_m_XF[turbstart_idx]
rtn_turb = turbulent_model.solve(x0=turbstart,x_end=s_ref[-1])
print(rtn_turb.message)
print(rtn_turb.x_end)
c_tau = turbulent_model.tau_w(s_ref[turbstart_idx:],rho_inf)
c_tau_XF = n_tild_c_tau[turbstart_idx:]
redelm = u_e_visc[turbstart_idx:]*turbulent_model.delta_m(s_ref[turbstart_idx:])/nu_inf
c_tau_eq = turbulent_model._c_tau_eq(turbulent_model.shape_d(s_ref[turbstart_idx:]),redelm,m_e_visc[turbstart_idx:],True)

fig, c_taus = plt.subplots()
#c_taus.plot(s_ref[turbstart_idx:],c_tau**.5,label='PyBL, ctau sqrt')
c_taus.plot(s_ref[turbstart_idx:],c_tau_eq,label=r'$c_{\tau EQ}$ PyBL',marker='o',markersize=4,color='#D0DF00')
c_taus.plot(s_ref[turbstart_idx:],c_tau_XF**2,label=r'$c_\tau$',marker='o',markersize=4,color='#FF6A39',linestyle='--')
c_taus.plot(s_ref[turbstart_idx:],c_tau_eq_XF**2,label=r'$c_{\tau EQ}$ XFOIL',marker='o',markersize=4,color='#D0DF00',linestyle='--')
c_taus.plot(s_ref[turbstart_idx:],turbulent_model.c_tau(s_ref[turbstart_idx:]),label=r'$c_{\tau}$ PyBL',marker='o',markersize=4,color='#FF6A39')
c_taus.set_title(r'$c_{\tau}$')
c_taus.legend()

print('XFOIL c_tau ratio')
print(c_tau_XF[0]/c_tau_eq_XF[0])

pass
transition_loc = s_ref[46]
full_model = transition_coupler(solution_range=s_ref,laminar_model=dg_laminar,turbulent_class=DrelaGilesTurbulentMOD,nu=nu_inf,U_e=[s_ref,u_e_visc],transition_loc=turbstart)

delta_d = full_model.delta_d(s_ref)
delta_m = full_model.delta_m(s_ref)
delta_k = full_model.delta_k(s_ref)
shape_d = full_model.shape_d(s_ref)
shape_k = full_model.shape_k(s_ref)
c_f     = full_model.tau_w(s_ref,rho_inf)/(0.5*rho_inf*u_e_visc**2)
c_D     = full_model.dissipation(s_ref,rho_inf)/(0.5*rho_inf*u_e_visc**3)

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,delta_d_XF,label=r'$\delta_d$ XFOIL',linestyle='--',marker='o',markersize=4,color='#A4D65E')
deltas.plot(s_ref,delta_m_XF,label=r'$\delta_m$ XFOIL',linestyle='--',marker='o',markersize=4,color='#3A913F')
deltas.plot(s_ref,delta_k_XF,label=r'$\delta_k$ XFOIL',linestyle='--',marker='o',markersize=4,color='#F2C75C')
deltas.plot(s_ref,delta_d,label=r'$\delta_d$ PyBL, Drela-Giles',marker='s',markersize=4,color='#A4D65E')
deltas.plot(s_ref,delta_m,label=r'$\delta_m$ PyBL, Drela-Giles',marker='s',markersize=4,color='#3A913F')
deltas.plot(s_ref,delta_k,label=r'$\delta_k$ PyBL, Drela-Giles',marker='s',markersize=4,color='#F2C75C')
deltas.plot([transition_loc,transition_loc],[min(delta_m_XF),max(delta_d_XF)],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses')
deltas.legend(ncol=2,borderaxespad=-8.5)
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Thicknesses [m]')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(delta_d-delta_d_XF)/delta_d_XF,label=r'$\delta_d$ PyBL, Drela-Giles',marker='s',markersize=4,color='#A4D65E')
deltas.plot(s_ref,np.abs(delta_m-delta_m_XF)/delta_m_XF,label=r'$\delta_m$ PyBL, Drela-Giles',marker='s',markersize=4,color='#3A913F')
deltas.plot(s_ref,np.abs(delta_k-delta_k_XF)/delta_k_XF,label=r'$\delta_k$ PyBL, Drela-Giles',marker='s',markersize=4,color='#F2C75C')
deltas.plot([transition_loc,transition_loc],[1e-6,1e-1],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend(borderaxespad=-8.5)

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,shape_d_XF,label=r'$H_d$ XFOIL',linestyle='--',marker='o',markersize=4,color='#A4D65E')
deltas.plot(s_ref,shape_k_XF,label=r'$H_k$ XFOIL',linestyle='--',marker='o',markersize=4,color='#F2C75C')
deltas.plot(s_ref,shape_d,label=r'$H_d$ PyBL, Drela-Giles',marker='s',markersize=4,color='#A4D65E')
deltas.plot(s_ref,shape_k,label=r'$H_k$ PyBL, Drela-Giles',marker='s',markersize=4,color='#F2C75C')
deltas.plot([transition_loc,transition_loc],[min(shape_d_XF),max(shape_d_XF)],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses')
deltas.legend(ncol=2,borderaxespad=-7.3)
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Shape Factors')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(shape_d-shape_d_XF)/shape_d_XF,label=r'$H_d$ PyBL, Drela-Giles',marker='s',markersize=4,color='#A4D65E')
deltas.plot(s_ref,np.abs(shape_k-shape_k_XF)/shape_k_XF,label=r'$H_k$ PyBL, Drela-Giles',marker='s',markersize=4,color='#F2C75C')
deltas.plot([transition_loc,transition_loc],[1e-6,1e-1],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend(ncol=2)

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,c_f_XF,label=r'$c_f$ XFOIL',linestyle='--',marker='o',markersize=4,color='#F8E08E')
deltas.plot(s_ref,c_D_XF,label=r'$c_D$ XFOIL',linestyle='--',marker='o',markersize=4,color='#5CB8B2')
deltas.plot(s_ref,c_f,label=r'$c_f$ PyBL, Drela-Giles',marker='s',markersize=4,color='#F8E08E')
deltas.plot(s_ref,c_D,label=r'$c_D$ PyBL, Drela-Giles',marker='s',markersize=4,color='#5CB8B2')
deltas.plot([transition_loc,transition_loc],[0,max(c_f)],color='black',linestyle='--')
deltas.set_ylim([0,0.02])
#deltas.set_title('Boundary Layer Thicknesses')
deltas.legend(ncol=2,borderaxespad=-7.3)
deltas.set_ylabel('Coefficients')
deltas.set_xlabel('s [m]')
#deltas.set_ylabel('Shape Factors')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(c_f-c_f_XF)/c_f_XF,label=r'$c_f$ PyBL, Drela-Giles',marker='s',markersize=4,color='#F8E08E')
deltas.plot(s_ref,np.abs(c_D-c_D_XF)/c_D_XF,label=r'$c_D$ PyBL, Drela-Giles',marker='s',markersize=4,color='#5CB8B2')
deltas.plot([transition_loc,transition_loc],[1e-3,1e2],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend(ncol=2)

fig, deltas = plt.subplots(constrained_layout=True)
shape_km_XF = DrelaGilesLaminarMOD._shape_km(shape_d_XF,m_e_visc)
tmp1 = DrelaGilesLaminarMOD._c_f_dg(shape_km_XF,re_delta_m_XF,True)
tmp2 = DrelaGilesLaminarMOD._c_D(shape_km_XF,shape_k_XF,re_delta_m_XF,True)
fctmp = DrelaGilesTurbulentMOD._fc(m_e_visc,True,1.4)
tmp1_1 = DrelaGilesTurbulentMOD._c_f_dg(shape_km_XF,re_delta_m_XF,fctmp,True)
u_stmp = DrelaGilesTurbulentMOD._u_s(shape_km_XF,re_delta_m_XF,m_e_visc,True)
tmp2_1 = DrelaGilesTurbulentMOD._c_D(tmp1_1,u_stmp,n_tild_c_tau**2,True)
c_f = np.append(tmp1[:turbstart_idx],tmp1_1[turbstart_idx:])
c_D = np.append(tmp2[:turbstart_idx],tmp2_1[turbstart_idx:])
deltas.plot(s_ref,c_f_XF,label=r'$c_f$ XFOIL',linestyle='--',marker='o',markersize=4,color='#F8E08E')
deltas.plot(s_ref,c_D_XF,label=r'$c_D$ XFOIL',linestyle='--',marker='o',markersize=4,color='#5CB8B2')
deltas.plot(s_ref,c_f,label=r'$c_f$ XFOIL, PyBL Static Method',marker='s',markersize=4,color='#F8E08E')
deltas.plot(s_ref,c_D,label=r'$c_D$ XFOIL, PyBL Static Method',marker='s',markersize=4,color='#5CB8B2')
deltas.plot([transition_loc,transition_loc],[0,max(c_f)],color='black',linestyle='--')
deltas.set_ylim([0,0.02])
#deltas.set_title('Boundary Layer Thicknesses')
deltas.legend(ncol=2,borderaxespad=-7.3)
deltas.set_ylabel('Coefficients')
deltas.set_xlabel('s [m]')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(c_f-c_f_XF)/c_f_XF,label=r'$c_f$ XFOIL, PyBL Static Method',marker='s',markersize=4,color='#F8E08E')
deltas.plot(s_ref,np.abs(c_D-c_D_XF)/c_D_XF,label=r'$c_D$ XFOIL, PyBL Static Method',marker='s',markersize=4,color='#5CB8B2')
deltas.plot([transition_loc,transition_loc],[1e-3,1e2],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend(ncol=2)

# ddeld_dx_pybl = []
# ddeld_dx_fd = []
# for idx,(s,delm,deld) in enumerate(zip(s_ref[:46],delta_m_XF[:46],delta_d_XF[:46])):
#     f = np.array([delm,deld,0.])
#     f_p = DrelaGilesLaminarMOD._ode_impl(dg_laminar,s,f)
#     FD_deld = (delta_d_XF[idx+1] - deld)/(s_ref[idx+1] - s)
#     ddeld_dx_pybl.append(f_p[1])
#     ddeld_dx_fd.append(FD_deld)

# fig, dddx = plt.subplots()
# dddx.plot(s_ref[:46],ddeld_dx_pybl,label='PyBL ddelta_d_dx',marker='o',markersize=4,color='#A4D65E')
# dddx.plot(s_ref[:46],ddeld_dx_fd,label='XFOIL ddelta_d_dx',linestyle='--',marker='o',markersize=4,color='#A4D65E')
# dddx.legend()

# fig, dddx = plt.subplots()
# dddx.plot(s_ref[:46],abs(np.array(ddeld_dx_pybl) - np.array(ddeld_dx_fd))/abs(np.array(ddeld_dx_fd)),marker='o',markersize=4,color='#A4D65E')
# dddx.set_title('ddelta_d_dx relative error')

#Thwaites and Heads Method, with the preprocessing

def Head_Preproc(lam_end_del_d,lam_end_del_m,lam_end_del_k,solution_range_laminar_end,nu,U_e_end,dU_edx_end,d2U_edx2_end):
    Re_dm_tmp = lam_end_del_m*U_e_end/nu
    Hd_tr = 1.4754/(np.log(Re_dm_tmp)) + 0.9698
    return lam_end_del_d,lam_end_del_m*Hd_tr,lam_end_del_k

tm_method = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[s_ref,u_e_visc],data_fits="Spline")
tm_method.initial_delta_m = delta_m[0]

full_model2 = transition_coupler(solution_range=s_ref,laminar_model=tm_method,turbulent_class=HeadMethod,nu=nu_inf,U_e=[s_ref,u_e_visc],transition_loc=turbstart,turb_ic_preprocessor=Head_Preproc)

delta_d = full_model2.delta_d(s_ref)
delta_m = full_model2.delta_m(s_ref)
delta_k = full_model2.delta_k(s_ref)
shape_d = full_model2.shape_d(s_ref)
shape_k = full_model2.shape_k(s_ref)
c_f     = full_model2.tau_w(s_ref,rho_inf)/(0.5*rho_inf*u_e_visc**2)
c_D     = full_model2.dissipation(s_ref,rho_inf)/(0.5*rho_inf*u_e_visc**3)

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,delta_d_XF,label=r'$\delta_d$ XFOIL',linestyle='--',marker='o',markersize=4,color='#F8E08E')
deltas.plot(s_ref,delta_m_XF,label=r'$\delta_m$ XFOIL',linestyle='--',marker='o',markersize=4,color='#3A913F')
#deltas.plot(s_ref,delta_k_XF,label=r'$\delta_k$ XF',linestyle='--',marker='o',markersize=4,color='#F2C75C')
deltas.plot(s_ref,delta_d,label=r"$\delta_d$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#F8E08E')
deltas.plot(s_ref,delta_m,label=r"$\delta_m$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#3A913F')
#deltas.plot(s_ref,delta_k,label=r'$\delta_k$ PyBL',marker='o',markersize=4,color='#F2C75C')
deltas.plot([transition_loc,transition_loc],[min(delta_m_XF),max(delta_d_XF)],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Thwaites and Heads')
deltas.legend(ncol=2,borderaxespad=-7.3)
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Thicknesses [m]')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(delta_d-delta_d_XF)/delta_d_XF,label=r"$\delta_d$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#F8E08E')
deltas.plot(s_ref,np.abs(delta_m-delta_m_XF)/delta_m_XF,label=r"$\delta_m$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#3A913F')
#deltas.plot(s_ref,np.abs(delta_k-delta_k_XF)/delta_k_XF,label=r'$\delta_k$ PyBL',marker='o',markersize=4,color='#F2C75C')
deltas.plot([transition_loc,transition_loc],[1e-15,1e-1],color='black',linestyle='--')
deltas.set_ylim([5e-6,1])
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors, Thwaites and Heads')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend(ncol=2)

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,shape_d_XF,label=r'$H_d$ XFOIL',linestyle='--',marker='o',markersize=4,color='#A4D65E')
deltas.plot(s_ref,shape_d,label=r"$H_d$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#A4D65E')
deltas.plot([transition_loc,transition_loc],[min(shape_d_XF),max(shape_d_XF)],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses')
deltas.legend(ncol=2)
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Shape Factors')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(shape_d-shape_d_XF)/shape_d_XF,label=r"$H_d$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#A4D65E')
deltas.plot([transition_loc,transition_loc],[1e-6,1e-1],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend()

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,c_f_XF,label=r'$c_f$ XFOIL',linestyle='--',marker='o',markersize=4,color='#F8E08E')
deltas.plot(s_ref,c_f,label=r"$c_f$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#F8E08E')
deltas.plot([transition_loc,transition_loc],[0,max(c_f)],color='black',linestyle='--')
deltas.set_ylim([0,0.02])
#deltas.set_title('Boundary Layer Thicknesses')
deltas.legend(ncol=2)
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Coefficients')

fig, deltas = plt.subplots(constrained_layout=True)
deltas.plot(s_ref,np.abs(c_f-c_f_XF)/c_f_XF,label=r"$c_f$ PyBL, Thwaites' & Head's",marker='s',markersize=4,color='#F8E08E')
deltas.plot([transition_loc,transition_loc],[1e-3,1e2],color='black',linestyle='--')
#deltas.set_title('Boundary Layer Thicknesses, Relative Errors')
deltas.set_yscale('log')
deltas.set_xlabel('s [m]')
deltas.set_ylabel('Relative Difference')
deltas.legend()

plt.show()
pass