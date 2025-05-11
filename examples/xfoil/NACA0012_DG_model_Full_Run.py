from ibl.drela_giles_turbulent_mod import DrelaGilesTurbulentMOD
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
from ibl.transition_coupler import transition_coupler
from ibl.initial_condition import ManualCondition
from ibl.interaction_law import interaction_law
from scipy.interpolate import CubicSpline

import numpy as np
import matplotlib.pyplot as plt

import os

# NACA 0012 Re = 1000000
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['figure.figsize'] = [13, 8]
plt.rcParams['legend.loc'] = 'lower center'
plt.rcParams['legend.borderaxespad'] = -8.6
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 3.

#print(os.path.dirname(os.path.abspath(__file__)))

file_name = "NACA0012_Re1E6_TR"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

re_inf = 1.e6
u_inf = 20
chord = 1.
nu_inf = chord*u_inf/re_inf
rho_inf = 1.2

xfoil_xtr = .687 #Location of switch for XFOIL

s_ref_inv = np.array([
                    0.,0.00840,0.01982,0.03304,0.04772,0.06337,0.07959,
                    0.0961,0.11275,0.12949,0.14626,0.16305,0.17984,0.19665,
                    0.21346,0.23027,0.24707,0.26388,0.28068,0.29748,0.31427,
                    0.33105,0.34783,0.3646,0.38135,0.39809,0.41482,0.43154,
                    0.44823,0.46491,0.48157,0.4982,0.51481,0.5314,0.54796,
                    0.56448,0.58098,0.59744,0.61386,0.63024,0.64657,0.66286,
                    0.6791,0.69527,0.71139,0.72743,0.7434,0.75928,0.77506,
                    0.79074,0.80628,0.82167,0.83688,0.85186,0.86655,0.88087,
                    0.8947,0.90789,0.92028,0.93169,0.942,0.95118,0.95926,
                    0.96635,0.97255,0.97802,0.98286,0.98718,0.99107,0.9946,
                    0.99781,1.00078,1.00352,1.00607,1.00847,1.01072,1.01286,
                    1.0149,1.01685,1.01872
                    ])
s_ref_inv = np.flip(abs(s_ref_inv - s_ref_inv[-1]))

u_e_inv = np.array([
                    0.76706,0.8738,0.90224,0.92502,0.94221,0.95628,0.96809,
                    0.97826,0.98719,0.99517,1.00243,1.00911,1.01532,
                    1.02116,1.0267,1.03199,1.03707,1.04199,1.04676,1.05143,
                    1.056,1.06049,1.06493,1.06932,1.07367,1.078,1.08231,
                    1.08661,1.09089,1.09518,1.09947,1.10375,1.10804,1.11233,
                    1.11662,1.12091,1.12519,1.12946,1.13371,1.13795,1.14215,
                    1.14632,1.15043,1.15448,1.15846,1.16234,1.1661,1.16972,
                    1.17318,1.17644,1.17945,1.18218,1.18455,1.1865,1.18792,
                    1.18869,1.18868,1.1877,1.18556,1.18209,1.1771,1.17046,
                    1.16202,1.1516,1.13898,1.12381,1.10565,1.08391,1.0578,
                    1.02628,0.98805,0.94146,0.88451,0.81489,0.7302,0.62847,
                    0.50901,0.37348,0.2263,0.07488
                    ])
u_e_inv = u_inf*u_e_inv
u_e_inv = np.flip(u_e_inv)

s_ref = np.array([
                    0.00091,0.00277,0.00472,0.00676,0.0089,0.01116,0.01355,
                    0.0161,0.01885,0.02181,0.02503,0.02855,0.03244,0.03676,
                    0.0416,0.04707,0.05328,0.06036,0.06844,0.07762,0.08793,
                    0.09935,0.11173,0.12492,0.13875,0.15307,0.16776,0.18274,
                    0.19795,0.21334,0.22888,0.24456,0.26034,0.27622,0.29219,
                    0.30823,0.32435,0.34052,0.35676,0.37305,0.38938,0.40576,
                    0.42218,0.43864,0.45514,0.47166,0.48822,0.50481,0.52142,
                    0.53805,0.55471,0.57139,0.58809,0.6048,0.62153,0.63827,
                    0.65503,0.67179,0.68857,0.70535,0.72214,0.73894,0.75574,
                    0.77255,0.78936,0.80616,0.82297,0.83978,0.85658,0.87336,
                    0.89013,0.90687,0.92353,0.94003,0.95625,0.9719,0.98658,
                    0.9998,1.01122,1.01962
                    ])

u_e_visc = np.array([
                    0.0746,0.22546,0.37217,0.50732,0.62657,0.72821,0.81295,
                    0.88269,0.93981,0.98659,1.02502,1.05674,1.08305,1.10497,
                    1.1233,1.13862,1.15139,1.16193,1.17048,1.17722,1.18229,
                    1.18583,1.18803,1.18906,1.18912,1.18838,1.187,1.18509,
                    1.18276,1.18007,1.1771,1.17388,1.17047,1.1669,1.16319,
                    1.15937,1.15545,1.15147,1.14743,1.14335,1.13923,1.1351,
                    1.13096,1.12681,1.12267,1.11855,1.11445,1.11038,1.10636,
                    1.10238,1.09847,1.09462,1.09087,1.08723,1.08371,1.08036,
                    1.07721,1.07437,1.07206,1.07035,1.05452,1.04147,1.03455,
                    1.03037,1.02669,1.02277,1.0184,1.01356,1.00822,1.00232,
                    0.9958,0.98855,0.9804,0.97116,0.96054,0.94825,0.93401,
                    0.91784,0.89949,0.88559
                    ])

u_e_visc = u_inf * u_e_visc
m_e_visc = u_e_visc/np.sqrt(1.4*287*288.15)

delta_m = np.array([
                        0.0000321,0.0000325,0.0000337,0.0000355,0.0000381,0.0000412,0.0000449,
                        0.000049,0.0000537,0.0000587,0.0000642,0.0000701,0.0000764,0.0000833,
                        0.0000907,0.0000988,0.0001075,0.000117,0.0001274,0.0001385,0.0001505,
                        0.000163,0.000176,0.0001893,0.0002026,0.0002159,0.0002292,0.0002423,
                        0.0002553,0.0002682,0.000281,0.0002937,0.0003063,0.0003188,0.0003313,
                        0.0003437,0.0003561,0.0003685,0.0003808,0.0003931,0.0004054,0.0004177,
                        0.00043,0.0004422,0.0004545,0.0004667,0.0004789,0.0004911,0.0005033,
                        0.0005154,0.0005275,0.0005395,0.0005513,0.000563,0.0005745,0.0005856,
                        0.0005963,0.0006063,0.0006149,0.0006221,0.0006723,0.0007189,0.0007556,
                        0.0007915,0.0008307,0.0008736,0.0009197,0.0009686,0.00102,0.0010742,
                        0.0011315,0.0011927,0.0012587,0.0013312,0.0014121,0.001504,0.0016102,
                        0.0017332,0.0018788,0.0019981
                        ])

delta_d = np.array([
                        0.0000715,0.0000726,0.0000755,0.0000799,0.0000861,0.0000939,0.000103,
                        0.0001135,0.0001251,0.0001379,0.0001519,0.000167,0.0001834,0.0002011,
                        0.0002204,0.0002414,0.0002643,0.0002893,0.0003167,0.0003463,0.0003781,
                        0.0004118,0.0004469,0.0004829,0.0005195,0.0005564,0.0005935,0.0006307,
                        0.000668,0.0007055,0.0007432,0.0007812,0.0008195,0.0008582,0.0008974,
                        0.0009371,0.0009773,0.0010182,0.0010598,0.0011021,0.0011452,0.0011893,
                        0.0012343,0.0012803,0.0013274,0.0013756,0.0014251,0.0014758,0.0015277,
                        0.0015808,0.0016349,0.0016897,0.0017446,0.0017987,0.0018505,0.0018977,
                        0.0019364,0.0019603,0.001957,0.0018983,0.0016757,0.0014552,0.0013233,
                        0.0012708,0.0012686,0.0012964,0.0013426,0.0014014,0.0014694,0.0015453,
                        0.0016289,0.0017208,0.0018228,0.0019376,0.0020696,0.0022251,0.0024126,
                        0.0026412,0.0029305,0.0031768
                        ])

shape_d = np.array([2.2295,2.2319,2.2398,2.2497,2.2638,2.2791,2.2962,
                        2.3136,2.3313,2.3487,2.3658,2.3824,2.3986,2.4142,
                        2.4294,2.4441,2.4585,2.4725,2.4862,2.4997,2.5128,
                        2.5258,2.5386,2.5513,2.564,2.5768,2.5898,2.603,
                        2.6167,2.6307,2.6452,2.6602,2.6757,2.6919,2.7087,
                        2.7261,2.7443,2.7632,2.7829,2.8034,2.8249,2.8472,
                        2.8706,2.8951,2.9207,2.9475,2.9755,3.0048,3.0353,
                        3.0669,3.0993,3.1321,3.1643,3.1947,3.2211,3.2404,
                        3.2472,3.2332,3.1825,3.0513,2.4925,2.0241,1.7513,
                        1.6056,1.5271,1.4838,1.4598,1.4469,1.4406,1.4386,
                        1.4396,1.4428,1.4481,1.4555,1.4657,1.4794,1.4983,
                        1.5239,1.5598,1.59])

shape_km = DrelaGilesLaminarMOD._shape_km(shape_d,m_e_visc)

shape_k = np.array([1.6211,1.6207,1.6194,1.6179,1.6157,1.6134,1.6108,
                        1.6083,1.6058,1.6034,1.6011,1.5989,1.5968,1.5949,
                        1.593,1.5912,1.5896,1.5879,1.5864,1.5849,1.5835,
                        1.5821,1.5808,1.5795,1.5782,1.5769,1.5757,1.5744,
                        1.5731,1.5719,1.5706,1.5693,1.5679,1.5666,1.5653,
                        1.5639,1.5625,1.5611,1.5597,1.5583,1.5568,1.5554,
                        1.5539,1.5525,1.551,1.5495,1.5481,1.5466,1.5452,
                        1.5438,1.5425,1.5412,1.5401,1.539,1.5382,1.5376,
                        1.5374,1.5378,1.5394,1.5153,1.5489,1.6097,1.6671,
                        1.7069,1.7314,1.7457,1.7535,1.7575,1.759,1.7588,
                        1.7575,1.7554,1.7525,1.7487,1.7439,1.7378,1.7298,
                        1.7196,1.7061,1.6952])

delta_k = np.array([
                        0.000052,0.0000527,0.0000546,0.0000575,0.0000615,0.0000664,0.0000723,
                        0.0000789,0.0000862,0.0000941,0.0001028,0.0001121,0.0001221,0.0001329,
                        0.0001445,0.0001572,0.0001709,0.0001858,0.0002021,0.0002196,0.0002383,
                        0.0002579,0.0002782,0.0002989,0.0003197,0.0003405,0.0003611,0.0003815,
                        0.0004016,0.0004216,0.0004413,0.0004608,0.0004802,0.0004995,0.0005186,
                        0.0005376,0.0005564,0.0005752,0.0005939,0.0006126,0.0006312,0.0006497,
                        0.0006681,0.0006865,0.0007049,0.0007232,0.0007414,0.0007596,0.0007777,
                        0.0007957,0.0008137,0.0008315,0.0008491,0.0008665,0.0008837,0.0009005,
                        0.0009168,0.0009324,0.0009466,0.0009427,0.0010413,0.0011573,0.0012597,
                        0.001351,0.0014383,0.0015251,0.0016128,0.0017022,0.0017942,0.0018893,
                        0.0019887,0.0020936,0.0022059,0.0023278,0.0024625,0.0026137,0.0027854,
                        0.0029803,0.0032054,0.0033872
                        ])

h_k_spline = CubicSpline(s_ref,shape_k)
spline_h_k_der = h_k_spline.derivative(1)

re_delta_m = np.array([2.394,7.336,12.539,18.028,23.842,29.989,36.472,
                        43.287,50.436,57.93,65.795,74.066,82.796,92.05,
                        101.906,112.453,123.783,135.978,149.087,163.097,177.906,
                        193.327,209.12,225.046,240.914,256.592,272.008,287.13,
                        301.952,316.482,330.736,344.732,358.489,372.027,385.361,
                        398.507,411.48,424.292,436.953,449.473,461.862,474.126,
                        486.272,498.305,510.228,522.044,533.753,545.353,556.84,
                        568.207,579.441,590.524,601.431,612.13,622.573,632.692,
                        642.373,651.399,659.23,665.884,708.93,748.753,781.743,
                        815.545,852.905,893.538,936.663,981.707,1028.39,1076.681,
                        1126.763,1179.024,1234.075,1292.789,1356.342,1426.179,1503.964,
                        1590.781,1689.969,1769.455])

c_f = np.array([0.0016681,0.0049555,0.0078167,0.0099679,0.0112792,0.0118614,0.0118742,
                    0.0115156,0.0109338,0.0102391,0.0095015,0.0087634,0.0080489,0.0073702,
                    0.0067329,0.0061376,0.0055845,0.0050733,0.0046025,0.0041727,0.0037842,
                    0.0034366,0.0031287,0.0028576,0.0026192,0.0024091,0.002223,0.0020571,
                    0.0019081,0.0017734,0.0016508,0.0015385,0.0014352,0.0013397,0.0012511,
                    0.0011684,0.0010912,0.0010188,0.0009507,0.0008865,0.0008258,0.0007684,
                    0.000714,0.0006623,0.0006131,0.0005663,0.0005217,0.0004793,0.0004391,
                    0.000401,0.0003652,0.000332,0.0003017,0.0002749,0.0002525,0.0002358,
                    0.0002271,0.0002304,0.0002551,0.0003329,0.0008494,0.0018021,0.0027615,
                    0.0034415,0.0038369,0.0040307,0.0040972,0.0040855,0.0040255,0.0039347,
                    0.0038227,0.0036944,0.0035513,0.0033924,0.003215,0.003015,0.0027882,
                    0.0025351,0.0022501,0.0020437])

c_D = np.array([0.0000358,0.0003217,0.0008421,0.0014738,0.0020806,0.0025718,0.0029117,
                    0.0031085,0.0031882,0.0031808,0.0031127,0.0030044,0.0028706,0.0027219,
                    0.0025653,0.0024054,0.0022458,0.002089,0.0019371,0.0017922,0.0016562,
                    0.0015306,0.0014165,0.0013141,0.0012226,0.0011412,0.0010685,0.0010034,
                    0.0009448,0.0008919,0.0008438,0.0007999,0.0007596,0.0007225,0.0006883,
                    0.0006567,0.0006273,0.0005999,0.0005744,0.0005506,0.0005282,0.0005073,
                    0.0004877,0.0004692,0.0004519,0.0004356,0.0004202,0.0004057,0.0003921,
                    0.0003793,0.0003672,0.0003559,0.0003453,0.0003354,0.0003263,0.0003179,
                    0.0003103,0.0003037,0.0002988,0.0013807,0.0024707,0.0027386,0.0025811,
                    0.0024114,0.002284,0.0021819,0.0020928,0.0020113,0.0019348,0.0018619,
                    0.0017915,0.0017224,0.0016537,0.0015841,0.0015124,0.0014378,0.0013601,
                    0.0012812,0.0012019,0.0011491])

n_tild_ctau_sqrt = np.array([
                        0.,0.,0.,0.,0.,0.,0.,
                        0.,0.,0.,0.,0.,0.,0.,
                        0.,0.,0.,0.,0.,0.,0.,
                        0.,0.,0.,0.,0.,0.0312519,0.1319303,
                        0.2767269,0.4319412,0.5916001,0.7561668,0.9261413,1.1020464,1.2844165,
                        1.4737939,1.6707253,1.8757598,2.0894481,2.3123418,2.5449932,2.7879548,
                        3.041778,3.3070103,3.5841916,3.873845,4.1764644,4.4924932,4.8222922,
                        5.1660921,5.5239238,5.8955192,6.2801711,6.6765346,7.082343,7.4939834,
                        7.9058097,8.3089275,8.6884371,0.0317976,0.0475731,0.0535137,0.0530551,
                        0.0507659,0.0484503,0.0465363,0.0450389,0.0438867,0.0430072,0.0423438,
                        0.0418557,0.0415152,0.0413059,0.0412211,0.0412639,0.0414465,0.0417877,
                        0.0422993,0.0430085,0.0436024
                        ])

c_tau_eq_sqrt = np.array([0.0850055,0.0752041,0.0632494,0.0534362,0.0468358,0.0427699,0.0403659,
                            0.0389932,0.0382575,0.0379223,0.0378482,0.0379546,0.0381988,0.0385643,
                            0.0390563,0.0397009,0.0405458,0.0416626,0.0431181,0.0450550,0.0466121,
                            0.0472325,0.0522985,0.0474197,0.0433185,0.0397521,0.0365318,0.0335369,
                            0.0306938,0.0279631,0.0253291,0.0227917,0.0203626,0.0180611,0.0159093,
                            0.0139276,0.0121314,0.0105292,0.0091217,0.0079025,0.0068591,0.0059754,
                            0.0052326,0.0046110])

fig, viscprof = plt.subplots(constrained_layout=True)
viscprof.plot(s_ref,u_e_visc,color='#154734',marker='o',markersize=4)
viscprof.set_ylabel(r'$u_e$ [m/s]')
viscprof.set_xlabel(r's [m]')

print("Laminar Model Only, Viscous u_e ~~~~~~~~~~~~~~~~~~~")
ic = ManualCondition(delta_d=delta_d[0],delta_m=delta_m[0],delta_k=delta_k[0])
dg_laminar = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[s_ref,u_e_visc],ic=ic,src=True)
du_e_visc = dg_laminar.du_e(s_ref)

rtn = dg_laminar.solve(x0=s_ref[0],x_end=s_ref[-1])
print("Laminar")
print(rtn.message)
print(rtn.x_end)

# fig, lam_res = plt.subplots()
# lam_res.plot(s_ref_visc,XF_delta_m,label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
# lam_res.plot(s_ref_visc,XF_delta_d,label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
# lam_res.plot(s_ref_visc,XF_delta_k,label=r'$\delta_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
# lam_res.plot(s_ref_visc[s_ref_visc<=rtn.x_end],dg_laminar.delta_m(s_ref_visc[s_ref_visc<=rtn.x_end]),label=r'$\delta_m$ PyBL',color='#3A913F')
# lam_res.plot(s_ref_visc[s_ref_visc<=rtn.x_end],dg_laminar.delta_d(s_ref_visc[s_ref_visc<=rtn.x_end]),label=r'$\delta_d$ PyBL',color='#A4D65E')
# lam_res.plot(s_ref_visc[s_ref_visc<=rtn.x_end],dg_laminar.delta_k(s_ref_visc[s_ref_visc<=rtn.x_end]),label=r'$\delta_k$ PyBL',color='#F2C75C')
# lam_res.set_title('Laminar only Boundary Layer Thicknesses')
# lam_res.legend()

transition_start_idx = 59
ic = ManualCondition(delta_d=delta_d[transition_start_idx],delta_m=delta_m[transition_start_idx],delta_k=delta_k[transition_start_idx])

print("Turbulent Model Only, Viscous u_e ~~~~~~~~~~~~~~~~~~~")
print('c_tau ratio of XFOIL')
print(n_tild_ctau_sqrt[transition_start_idx]/c_tau_eq_sqrt[0])
dg_turbulent = DrelaGilesTurbulentMOD(nu=nu_inf,U_e=[s_ref,u_e_visc],ic=ic,show_prog=False,c_tau_init=n_tild_ctau_sqrt[transition_start_idx]**2)

rtn = dg_turbulent.solve(x0=s_ref[transition_start_idx],x_end=s_ref[-1])
print("Turbulent")
print(rtn.message)
print(rtn.x_end)

# fig, turb_res = plt.subplots()
# turb_res.plot(s_ref_visc,XF_delta_m,label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
# turb_res.plot(s_ref_visc,XF_delta_d,label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
# turb_res.plot(s_ref_visc,XF_delta_k,label=r'$\delta_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
# turb_res.plot(s_ref_visc[transition_start_idx:],dg_turbulent.delta_m(s_ref_visc[transition_start_idx:]),label=r'$\delta_m$ PyBL',color='#3A913F')
# turb_res.plot(s_ref_visc[transition_start_idx:],dg_turbulent.delta_d(s_ref_visc[transition_start_idx:]),label=r'$\delta_d$ PyBL',color='#A4D65E')
# turb_res.plot(s_ref_visc[transition_start_idx:],dg_turbulent.delta_k(s_ref_visc[transition_start_idx:]),label=r'$\delta_k$ PyBL',color='#F2C75C')
# turb_res.set_title('Turbulent only Boundary Layer Thicknesses')
# turb_res.legend()

print("Full Model, Viscous u_e ~~~~~~~~~~~~~~~~~~~")
ic = ManualCondition(delta_d=delta_d[0],delta_m=delta_m[0],delta_k=delta_k[0])
dg_laminar = DrelaGilesLaminarMOD(nu=nu_inf,U_e=[s_ref,u_e_visc],ic=ic,n_tilde_crit=8.8,src=True)
rtnL = dg_laminar.solve(s_ref[0],s_ref[-1])
full_model = transition_coupler(s_ref,dg_laminar,DrelaGilesTurbulentMOD,nu=nu_inf,U_e=[s_ref,u_e_visc])



fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,full_model.delta_d(s_ref),label=r'D-G, Lam + Turb',color='#A4D65E')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(full_model.delta_d(s_ref)-delta_d)/delta_d,label=r'D-G, Lam + Turb',color='#A4D65E')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,full_model.delta_m(s_ref),label=r'D-G, Lam + Turb',color='#3A913F')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(full_model.delta_m(s_ref)-delta_m)/delta_m,label=r'D-G, Lam + Turb',color='#3A913F')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,full_model.delta_k(s_ref),label=r'D-G, Lam + Turb',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(full_model.delta_k(s_ref)-delta_k)/delta_k,label=r'D-G, Lam + Turb',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_visc.png')


fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,full_model.shape_d(s_ref),label=r'D-G, Lam + Turb',color='#A4D65E')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(full_model.shape_d(s_ref)-shape_d)/shape_d,label=r'D-G, Lam + Turb',color='#A4D65E')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_d$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_d_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,full_model.shape_k(s_ref),label=r'D-G, Lam + Turb',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(full_model.shape_k(s_ref)-shape_k)/shape_k,label=r'D-G, Lam + Turb',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_k$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_k_visc.png')


cf_dg = full_model.tau_w(s_ref,rho_inf)/(.5*rho_inf*u_inf**2)
cD_dg = full_model.dissipation(s_ref,rho_inf)/(.5*rho_inf*u_inf**3)

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_f,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,cf_dg,label=r'D-G, Lam + Turb',color='#F8E08E')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,abs(c_f-cf_dg)/abs(c_f),label=r'D-G, Lam + Turb',color='#F8E08E')
dels.set_ylabel(r'$c_f$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_f_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_D,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,cD_dg,label=r'D-G, Lam + Turb',color='#5CB8B2')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,abs(c_D-cD_dg)/abs(c_D),label=r'D-G, Lam + Turb',color='#5CB8B2')
dels.set_ylabel(r'$c_D$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_D_visc.png')

fig, n_tild = plt.subplots(constrained_layout=True)
n_tild.plot(s_ref[:transition_start_idx],n_tild_ctau_sqrt[:transition_start_idx],label='XFOIL',linestyle='--',color='black')
n_tild.plot(s_ref[s_ref<=rtnL.x_end],dg_laminar.n_tilde(s_ref[s_ref<=rtnL.x_end]),label='D-G',color='#FF6A39')
n_tild.set_xlabel('s [m]')
n_tild.set_ylabel(r'$\tilde{n}$')
n_tild.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'n_tild_visc.png')

#Inviscid

fig, invprof = plt.subplots(constrained_layout=True)
invprof.plot(s_ref,u_e_visc,label=r'Viscous',color='black',linestyle='--',linewidth=4.)
invprof.plot(s_ref_inv,u_e_inv,label=r'Inviscid',color='#154734')

function_corrections,preproc_ue = interaction_law(s_ref_inv,u_inf,u_e_inv,nu_inf,False,55)

invprof.plot(s_ref_inv,u_e_inv+function_corrections[:len(u_e_inv)],label=r'Inv. + Int. Law',color='#154734',linestyle=':')
invprof.plot(s_ref_inv,preproc_ue+function_corrections[:len(u_e_inv)],label=r'Preproc Inv. + Int. Law',color='#154734',linestyle='--')
invprof.legend(borderaxespad=-9.5)
invprof.set_ylim([15,24])
invprof.set_ylabel(r'$u_e$ [m/s]')
invprof.set_xlabel(r's [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'u_e_profs.png')

# ic = ManualCondition(delta_d=delta_d[transition_start_idx],delta_m=delta_m[transition_start_idx],delta_k=delta_k[transition_start_idx])
# dg_turbulent = DrelaGilesTurbulentMOD(nu=nu_inf,
#                                       U_e=[s_ref,u_e_visc],
#                                       ic=ic,show_prog=False,cf_crit=1e-4,c_tau_init=n_tild_ctau_sqrt[transition_start_idx]**2)

# print("Turbulent Model Only, Viscous u_e ~~~~~~~~~~~~~~~~~~~")
# rtn = dg_turbulent.solve(x0=s_ref_inv[transition_start_idx],x_end=s_ref_inv[-1])
# print("Turbulent")
# print(rtn.message)
# print(rtn.x_end)
# temp_s_ref = s_ref_inv[transition_start_idx:]
# calcend = rtn.x_end
# endidx = len(s_ref[s_ref<=calcend])
# if rtn.message == 'Completed':
#     endidx += 1
# fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
# dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],delta_d[transition_start_idx:endidx],label=r'XFOIL',linestyle='--',color='black',linewidth=4)
# dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_d(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'D-G, Turb',color='#A4D65E')
# dels_err.plot(temp_s_ref[0],[-.1],linestyle='--',label='XFOIL',color='black')
# dels_err.plot(temp_s_ref[temp_s_ref<=rtn.x_end],np.abs(dg_turbulent.delta_d(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_d[transition_start_idx:endidx])/delta_d[transition_start_idx:endidx],label=r'D-G, Turb',color='#A4D65E')
# dels_err.legend(ncol=3,borderaxespad=-5.5)
# dels.set_ylabel(r'$\delta_d$ [m]')
# dels_err.set_yscale('log')
# dels_err.set_ylabel('Relative Difference')
# dels_err.set_xlabel('s [m]')
# fig.tight_layout()
# fig.savefig(file_name+'\\'+'delta_d_visc_turb.png')

# fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
# dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],delta_m[transition_start_idx:endidx],label=r'XFOIL',linestyle='--',color='black',linewidth=4)
# dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_m(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'D-G, Turb',color='#3A913F')
# dels_err.plot(temp_s_ref[0],[-.1],linestyle='--',label='XFOIL',color='black')
# dels_err.plot(temp_s_ref[temp_s_ref<=rtn.x_end],np.abs(dg_turbulent.delta_m(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_m[transition_start_idx:endidx])/delta_m[transition_start_idx:endidx],label=r'D-G, Turb',color='#3A913F')
# dels_err.legend(ncol=3,borderaxespad=-5.5)
# dels.set_ylabel(r'$\delta_m$ [m]')
# dels_err.set_yscale('log')
# dels_err.set_ylabel('Relative Difference')
# dels_err.set_xlabel('s [m]')
# fig.tight_layout()
# fig.savefig(file_name+'\\'+'delta_m_visc_turb.png')

# fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
# dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],delta_k[transition_start_idx:endidx],label=r'XFOIL',linestyle='--',color='black',linewidth=4)
# dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_k(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'D-G, Turb',color='#F2C75C')
# dels_err.plot(temp_s_ref[0],[-.1],linestyle='--',label='XFOIL',color='black')
# dels_err.plot(temp_s_ref[temp_s_ref<=rtn.x_end],np.abs(dg_turbulent.delta_k(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_k[transition_start_idx:endidx])/delta_k[transition_start_idx:endidx],label=r'D-G, Turb',color='#F2C75C')
# dels_err.legend(ncol=3,borderaxespad=-5.5)
# dels.set_ylabel(r'$\delta_k$ [m]')
# dels_err.set_yscale('log')
# dels_err.set_ylabel('Relative Difference')
# dels_err.set_xlabel('s [m]')
# fig.tight_layout()
# fig.savefig(file_name+'\\'+'delta_k_visc_turb.png')


#u_e = [s_ref_inv,u_e_inv]
#u_e = [s_ref_inv,u_e_inv+function_corrections[:len(u_e_inv)]]
u_e = [s_ref_inv,preproc_ue+function_corrections[:len(u_e_inv)]]

ic = ManualCondition(delta_d=delta_d[transition_start_idx],delta_m=delta_m[transition_start_idx],delta_k=delta_k[transition_start_idx])
dg_turbulent = DrelaGilesTurbulentMOD(nu=nu_inf,
                                      U_e=u_e,
                                      ic=ic,show_prog=False,cf_crit=1e-4,c_tau_init=n_tild_ctau_sqrt[transition_start_idx]**2)

print("Turbulent Model Only, Inviscid u_e ~~~~~~~~~~~~~~~~~~~")
rtn = dg_turbulent.solve(x0=s_ref_inv[transition_start_idx],x_end=s_ref_inv[-1])
print("Turbulent")
print(rtn.message)
print(rtn.x_end)
temp_s_ref = s_ref_inv[transition_start_idx:]
calcend = rtn.x_end
endidx = len(s_ref[s_ref<=calcend])
if rtn.message == 'Completed':
    endidx += 1
fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],delta_d[transition_start_idx:endidx],label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_d(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'D-G, Turb',color='#A4D65E')
dels_err.plot(temp_s_ref[0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(temp_s_ref[temp_s_ref<=rtn.x_end],np.abs(dg_turbulent.delta_d(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_d[transition_start_idx:endidx])/delta_d[transition_start_idx:endidx],label=r'D-G, Turb',color='#A4D65E')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_inv_turb.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],delta_m[transition_start_idx:endidx],label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_m(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'D-G, Turb',color='#3A913F')
dels_err.plot(temp_s_ref[0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(temp_s_ref[temp_s_ref<=rtn.x_end],np.abs(dg_turbulent.delta_m(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_m[transition_start_idx:endidx])/delta_m[transition_start_idx:endidx],label=r'D-G, Turb',color='#3A913F')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_inv_turb.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],delta_k[transition_start_idx:endidx],label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_k(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'D-G, Turb',color='#F2C75C')
dels_err.plot(temp_s_ref[0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(temp_s_ref[temp_s_ref<=rtn.x_end],np.abs(dg_turbulent.delta_k(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_k[transition_start_idx:endidx])/delta_k[transition_start_idx:endidx],label=r'D-G, Turb',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_inv_turb.png')


print("Full Model, Inviscid u_e ~~~~~~~~~~~~~~~~~~~")
ic = ManualCondition(delta_d=delta_d[0],delta_m=delta_m[0],delta_k=delta_k[0])
dg_laminar = DrelaGilesLaminarMOD(nu=nu_inf,
                                  U_e=u_e,
                                  ic=ic)

full_model = transition_coupler(s_ref_inv,dg_laminar,DrelaGilesTurbulentMOD,nu=nu_inf,
                                U_e=[s_ref_inv,u_e],
                                transition_loc=s_ref_inv[transition_start_idx],cf_crit=1e-6,sep_tran=True)

calcend = full_model.turb_x_end

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[s_ref<=calcend],full_model.delta_d(s_ref[s_ref<=calcend]),label=r'D-G, Lam + Turb',color='#A4D65E')
dels.set_ylim([0,1.2*max(delta_d)])
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[s_ref<=calcend],np.abs(full_model.delta_d(s_ref[s_ref<=calcend])-delta_d[:len(s_ref[s_ref<=calcend])])/delta_d[:len(s_ref[s_ref<=calcend])],label=r'D-G, Lam + Turb',color='#A4D65E')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[s_ref<=calcend],full_model.delta_m(s_ref[s_ref<=calcend]),label=r'D-G, Lam + Turb',color='#3A913F')
dels.set_ylim([0,1.2*max(delta_m)])
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[s_ref<=calcend],np.abs(full_model.delta_m(s_ref[s_ref<=calcend])-delta_m[:len(s_ref[s_ref<=calcend])])/delta_m[:len(s_ref[s_ref<=calcend])],label=r'D-G, Lam + Turb',color='#3A913F')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[s_ref<=calcend],full_model.delta_k(s_ref[s_ref<=calcend]),label=r'D-G, Lam + Turb',color='#F2C75C')
dels.set_ylim([0,1.2*max(delta_k)])
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[s_ref<=calcend],np.abs(full_model.delta_k(s_ref[s_ref<=calcend])-delta_k[:len(s_ref[s_ref<=calcend])])/delta_k[:len(s_ref[s_ref<=calcend])],label=r'D-G, Lam + Turb',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_inv.png')

plt.show()

fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,delta_m,label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
all_res.plot(s_ref,delta_d,label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
all_res.plot(s_ref,delta_k,label=r'$\delta_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
all_res.plot(s_ref,full_model.delta_m(s_ref),label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
all_res.plot(s_ref,full_model.delta_d(s_ref),label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref,full_model.delta_k(s_ref),label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_ylabel('Thicknesses [m]')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=2,borderaxespad=-8.5)

fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,np.abs(full_model.delta_m(s_ref)-delta_m)/delta_m,label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
all_res.plot(s_ref,np.abs(full_model.delta_d(s_ref)-delta_d)/delta_d,label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref,np.abs(full_model.delta_k(s_ref)-delta_k)/delta_k,label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_ylabel('Relative Difference')
all_res.set_yscale('log')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=3)

fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,shape_d,label=r'$H_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
all_res.plot(s_ref,shape_k,label=r'$H_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
all_res.plot(s_ref,full_model.shape_d(s_ref),label=r'$H_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref,full_model.shape_k(s_ref),label=r'$H_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_ylabel('Shape Factors')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=2,borderaxespad=-7.)

fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,np.abs(full_model.shape_d(s_ref)-shape_d)/shape_d,label=r'$H_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref,np.abs(full_model.shape_k(s_ref)-shape_k)/shape_k,label=r'$H_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_ylabel('Relative Difference')
all_res.set_yscale('log')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=2)

fig, n_tilde_chek = plt.subplots(constrained_layout=True)
n_tilde_chek.plot(s_ref[:transition_start_idx],n_tild_ctau_sqrt[:transition_start_idx],marker='o',markersize=4,linestyle='--',color='#FF6A39',label='XFOIL')
n_tilde_chek.plot(s_ref[:transition_start_idx],dg_laminar.n_tilde(s_ref[:transition_start_idx]),marker='s',markersize=4,color='#FF6A39',label='PyBL, Drela-Giles')
n_tilde_chek.legend(ncol=2)
n_tilde_chek.set_ylabel(r'$\tilde{n}$')
n_tilde_chek.set_xlabel(r's [m]')

fig, ctau_chek = plt.subplots(constrained_layout=True)
ctau_chek.plot(s_ref[transition_start_idx:],n_tild_ctau_sqrt[transition_start_idx:]**2,marker='o',markersize=4,linestyle='--',color='#D0DF00',label='XFOIL')
ctau_chek.plot(s_ref[transition_start_idx:],dg_turb_test.c_tau(s_ref[transition_start_idx:]),marker='s',markersize=4,color='#D0DF00',label='PyBL, Drela-Giles')
ctau_chek.plot(s_ref[transition_start_idx:],dg_turbulent.c_tau(s_ref[transition_start_idx:]),marker='s',markersize=4,color='#D0DF00',label='PyBL, Drela-Giles, Turbulent Only',linestyle='--')
ctau_chek.legend(ncol=2)
ctau_chek.set_ylabel(r'$c_{\tau}$')
ctau_chek.set_xlabel(r's [m]')

cf_man = DrelaGilesLaminarMOD._c_f_dg(shape_km,re_delta_m,True)
cD_man = DrelaGilesLaminarMOD._c_D(shape_km,shape_k,re_delta_m,True)
fc = DrelaGilesTurbulentMOD._fc(m_e_visc,True,1.4)
cf_man2 = DrelaGilesTurbulentMOD._c_f_dg(shape_km,re_delta_m,fc,True)
u_s = DrelaGilesTurbulentMOD._u_s(shape_km,re_delta_m,m_e_visc,True)
cD_man2 = DrelaGilesTurbulentMOD._c_D(cf_man2,u_s,n_tild_ctau_sqrt**2,True)

c_f = full_model.tau_w(s_ref,rho_inf)/(.5*rho_inf*u_inf**2)
c_D = full_model.dissipation(s_ref,rho_inf)/(.5*rho_inf*u_inf**3)

tmp1 = np.abs(c_f - c_f)/c_f
tmp2 = np.abs(c_D - c_D)/c_D
tmp1_1 = np.abs(cf_man - c_f)/c_f
tmp2_1 = np.abs(cD_man - c_D)/c_D
tmp1_2 = np.abs(cf_man2 - c_f)/c_f
tmp2_2 = np.abs(cD_man2 - c_D)/c_D

fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,c_f,label=r'$c_f$ XFOIL',marker='o',color='#F8E08E',markersize=4,linestyle='--')
all_res.plot(s_ref,c_D,label=r'$c_D$ XFOIL',marker='o',color='#5CB8B2',markersize=4,linestyle='--')
all_res.plot(s_ref,c_f,label=r'$c_f$ PyBL, Drela-Giles',marker='o',color='#F8E08E',markersize=4)
all_res.plot(s_ref,c_D,label=r'$c_D$ PyBL, Drela-Giles',color='#5CB8B2',marker='s',markersize=4)
# all_res.plot(s_ref_visc,np.append(cf_man[:transition_start_idx],cf_man2[transition_start_idx:]),label=r'$c_f$ Drela-Giles Function, XFOIL Data',color='#F8E08E',marker='*',markersize=4,linestyle='--')
# all_res.plot(s_ref_visc,np.append(cD_man[:transition_start_idx],cD_man2[transition_start_idx:]),label=r'$c_D$ Drela-Giles Function, XFOIL Data',color='#5CB8B2',marker='*',markersize=4,linestyle='--')
all_res.set_ylabel('Coefficients')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=2,borderaxespad=-7.)

fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,tmp1,label=r'$c_f$ PyBL, Drela-Giles',marker='o',color='#F8E08E',markersize=4)
all_res.plot(s_ref,tmp2,label=r'$c_D$ PyBL, Drela-Giles',color='#5CB8B2',marker='s',markersize=4)
# all_res.plot(s_ref_visc,np.append(tmp1_1[:transition_start_idx],tmp1_2[transition_start_idx:]),label=r'$c_f$ Drela-Giles Function, XFOIL Data',color='#F8E08E',marker='*',markersize=4,linestyle='--')
# all_res.plot(s_ref_visc,np.append(tmp2_1[:transition_start_idx],tmp2_2[transition_start_idx:]),label=r'$c_D$ Drela-Giles Function, XFOIL Data',color='#5CB8B2',marker='*',markersize=4,linestyle='--')
all_res.set_ylabel('Relative Difference')
all_res.set_yscale('log')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=2,borderaxespad=-5.5)


# fig, dhk = plt.subplots()
# dhk.plot(s_ref_visc[:transition_start_idx],dgm_dh_k_dx[:transition_start_idx],label='DGM',marker='s',markersize=2)
# dhk.plot(s_ref_visc[:transition_start_idx],spline_dh_k_dx[:transition_start_idx],label='Spline',marker='s',markersize=2)
# dhk.set_ylim([-1,0])
# dhk.legend()
# dhk.set_title('dh_k_dx')

# fig, dhk_ratio = plt.subplots()
# dhk_ratio.plot(s_ref_visc[:transition_start_idx],spline_dh_k_dx[:transition_start_idx]/dgm_dh_k_dx[:transition_start_idx])
# dhk_ratio.set_title('dh_k_dx ratio')

# fig, dhk_ratio2 = plt.subplots()
# dhk_ratio2.plot(s_ref_visc[transition_start_idx:],spline_dh_k_dx[transition_start_idx:]/dgm_dh_k_dx2[transition_start_idx:])
# dhk_ratio2.set_title('dh_k_dx ratio, turbulent')
#plt.show()
#big question, what about the inviscid layer?

fig, invprof = plt.subplots(constrained_layout=True)
invprof.plot(s_ref,u_e_visc,label=r'Viscous $u_e$ Profile',color='black',marker='o',markersize=4)
invprof.plot(s_ref_inv,u_e_inv,label=r'Inviscid $u_e$ Profile',color='#154734',linestyle='--',marker='o',markersize=4)

function_corrections,preproc_ue = interaction_law(s_ref_inv,u_inf,u_e_inv,nu_inf,False,55)

invprof.plot(s_ref_inv,u_e_inv+function_corrections[:len(u_e_inv)],label=r'Inviscid $u_e$ Profile + Interaction Law Function Results',color='#154734',marker='*',markersize=4)
invprof.plot(s_ref_inv,preproc_ue+function_corrections[:len(u_e_inv)],label=r'Preprocessed Inviscid $u_e$ Profile + Interaction Law Function Results',color='#154734',linestyle='--',marker='*',markersize=4)
invprof.legend(borderaxespad=-9.5)
invprof.set_ylim([15,24])
invprof.set_ylabel(r'$u_e$ [m/s]')
invprof.set_xlabel(r's [m]')
fig.tight_layout()


u_e = [s_ref_inv,u_e_inv]
#u_e = [s_ref_inv,u_e_inv+function_corrections[:len(u_e_inv)]]
#u_e = [s_ref_inv,preproc_ue+function_corrections[:len(u_e_inv)]]

ic = ManualCondition(delta_d=delta_d[0],delta_m=delta_m[0],delta_k=delta_k[0])
dg_laminar = DrelaGilesLaminarMOD(nu=nu_inf,
                                  U_e=u_e,
                                  ic=ic)

rtn = dg_laminar.solve(x0=s_ref_inv[0],x_end=s_ref_inv[-1])
print("Laminar Model Only, Inviscid u_e ~~~~~~~~~~~~~~~~~~~")
print(rtn.message)
print(rtn.x_end)
transition_start_idx = 59

lam_endpos = rtn.x_end
#lam_endpos = s_ref_inv[transition_start_idx]

fig, lam_res = plt.subplots(constrained_layout=True)
lam_res.plot(s_ref,delta_m,label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
lam_res.plot(s_ref,delta_d,label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
lam_res.plot(s_ref,delta_k,label=r'$\delta_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
lam_res.plot(s_ref_inv[s_ref_inv<=lam_endpos],dg_laminar.delta_m(s_ref_inv[s_ref_inv<=lam_endpos]),label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
lam_res.plot(s_ref_inv[s_ref_inv<=lam_endpos],dg_laminar.delta_d(s_ref_inv[s_ref_inv<=lam_endpos]),label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
lam_res.plot(s_ref_inv[s_ref_inv<=lam_endpos],dg_laminar.delta_k(s_ref_inv[s_ref_inv<=lam_endpos]),label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
lam_res.set_ylabel('Thicknesses [m]')
lam_res.set_xlabel('s [m]')
lam_res.legend(ncol=2,borderaxespad=-8.5)

calcend = lam_endpos
endidx = len(s_ref[s_ref<=calcend])
fig, all_res = plt.subplots(constrained_layout=True)
tmp1 = np.abs(dg_laminar.delta_m(s_ref[s_ref<=calcend])-delta_m[:endidx])/delta_m[:endidx]
tmp2 = np.abs(dg_laminar.delta_d(s_ref[s_ref<=calcend])-delta_d[:endidx])/delta_d[:endidx]
tmp3 = np.abs(dg_laminar.delta_k(s_ref[s_ref<=calcend])-delta_k[:endidx])/delta_k[:endidx]
all_res.plot(s_ref[s_ref<=calcend],tmp1,label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
all_res.plot(s_ref[s_ref<=calcend],tmp2,label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref[s_ref<=calcend],tmp3,label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_yscale('log')
all_res.set_ylabel('Relative Difference')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=3)


ic = ManualCondition(delta_d=delta_d[transition_start_idx],delta_m=delta_m[transition_start_idx],delta_k=delta_k[transition_start_idx])
dg_turbulent = DrelaGilesTurbulentMOD(nu=nu_inf,
                                      U_e=u_e,
                                      ic=ic,show_prog=False,cf_crit=1e-4,c_tau_init=n_tild_ctau_sqrt[transition_start_idx]**2)

print("Turbulent Model Only, Inviscid u_e ~~~~~~~~~~~~~~~~~~~")
rtn = dg_turbulent.solve(x0=s_ref_inv[transition_start_idx],x_end=s_ref_inv[-1])
print("Turbulent")
print(rtn.message)
print(rtn.x_end)

fig, turb_res = plt.subplots(constrained_layout=True)
turb_res.plot(s_ref,delta_m,label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
turb_res.plot(s_ref,delta_d,label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
turb_res.plot(s_ref,delta_k,label=r'$\delta_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
temp_s_ref = s_ref_inv[transition_start_idx:]
turb_res.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_m(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
turb_res.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_d(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
turb_res.plot(temp_s_ref[temp_s_ref<=rtn.x_end],dg_turbulent.delta_k(temp_s_ref[temp_s_ref<=rtn.x_end]),label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
turb_res.set_ylim([0,max(delta_d)+.002])
turb_res.set_ylabel('Thicknesses [m]')
turb_res.set_xlabel('s [m]')
turb_res.legend(ncol=2,borderaxespad=-8.5)

calcend = rtn.x_end
endidx = len(s_ref[s_ref<=calcend])
if rtn.message == 'Completed':
    endidx += 1
fig, all_res = plt.subplots(constrained_layout=True)
tmp1 = np.abs(dg_turbulent.delta_m(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_m[transition_start_idx:endidx])/delta_m[transition_start_idx:endidx]
tmp2 = np.abs(dg_turbulent.delta_d(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_d[transition_start_idx:endidx])/delta_d[transition_start_idx:endidx]
tmp3 = np.abs(dg_turbulent.delta_k(temp_s_ref[temp_s_ref<=rtn.x_end])-delta_k[transition_start_idx:endidx])/delta_k[transition_start_idx:endidx]
all_res.plot(temp_s_ref[temp_s_ref<=rtn.x_end],tmp1,label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
all_res.plot(temp_s_ref[temp_s_ref<=rtn.x_end],tmp2,label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(temp_s_ref[temp_s_ref<=rtn.x_end],tmp3,label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_yscale('log')
all_res.set_ylabel('Relative Difference')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=3)

print("Full Model, Inviscid u_e ~~~~~~~~~~~~~~~~~~~")
ic = ManualCondition(delta_d=delta_d[0],delta_m=delta_m[0],delta_k=delta_k[0])
dg_laminar = DrelaGilesLaminarMOD(nu=nu_inf,
                                  U_e=u_e,
                                  ic=ic)

full_model = transition_coupler(s_ref_inv,dg_laminar,DrelaGilesTurbulentMOD,nu=nu_inf,
                                U_e=[s_ref_inv,u_e],
                                transition_loc=s_ref_inv[transition_start_idx],cf_crit=1e-6,sep_tran=True)

calcend = full_model.turb_x_end

endidx = len(s_ref[s_ref<=calcend])
fig, all_res = plt.subplots(constrained_layout=True)
all_res.plot(s_ref,delta_m,label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
all_res.plot(s_ref,delta_d,label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
all_res.plot(s_ref,delta_k,label=r'$\delta_k$ XFOIL',marker='o',color='#F2C75C',markersize=4,linestyle='--')
all_res.plot(s_ref[s_ref<=calcend],full_model.delta_m(s_ref[s_ref<=calcend]),label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
all_res.plot(s_ref[s_ref<=calcend],full_model.delta_d(s_ref[s_ref<=calcend]),label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref[s_ref<=calcend],full_model.delta_k(s_ref[s_ref<=calcend]),label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_ylim([0,max(delta_d)+.002])
all_res.set_ylabel('Thicknesses [m]')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=2,borderaxespad=-8.5)

fig, all_res = plt.subplots(constrained_layout=True)
tmp1 = np.abs(full_model.delta_m(s_ref[s_ref<=calcend])-delta_m[:endidx])/delta_m[:endidx]
tmp2 = np.abs(full_model.delta_d(s_ref[s_ref<=calcend])-delta_d[:endidx])/delta_d[:endidx]
tmp3 = np.abs(full_model.delta_k(s_ref[s_ref<=calcend])-delta_k[:endidx])/delta_k[:endidx]
all_res.plot(s_ref[s_ref<=calcend],tmp1,label=r'$\delta_m$ PyBL, Drela-Giles',color='#3A913F',marker='s',markersize=4)
all_res.plot(s_ref[s_ref<=calcend],tmp2,label=r'$\delta_d$ PyBL, Drela-Giles',color='#A4D65E',marker='s',markersize=4)
all_res.plot(s_ref[s_ref<=calcend],tmp3,label=r'$\delta_k$ PyBL, Drela-Giles',color='#F2C75C',marker='s',markersize=4)
all_res.set_yscale('log')
all_res.set_ylabel('Relative Difference')
all_res.set_xlabel('s [m]')
all_res.legend(ncol=3)

plt.show()
pass