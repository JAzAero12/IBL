from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.drela_giles_turbulent_mod import DrelaGilesTurbulentMOD
from ibl.head_method import HeadMethod
from ibl.transition_coupler import transition_coupler

import os

# NACA 0012 Re = 1000000
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

file_name = "NACA0012_Re1E6_VISC_INV"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

s_ref = np.array([0.00091,0.00277,0.00472,0.00676,0.0089,0.01116,0.01355,
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
                0.9998,1.01122,1.01962])

c = 1  # (m)
u_inf = 20  # (m/s)
re = 1e6
rho_inf = 1.2
nu_inf = u_inf*c/re
x_trans = 0.001

u_e = u_inf*np.array([0.07452,0.22522,0.37176,0.50683,0.6259,0.72747,0.81213,
                        0.88189,0.93924,0.98523,1.02289,1.05497,1.08151,1.10356,
                        1.12195,1.13732,1.15011,1.16067,1.16914,1.1758,1.18078,
                        1.18414,1.18593,1.18669,1.18683,1.18624,1.18505,1.1833,
                        1.18109,1.1785,1.17561,1.17245,1.1691,1.16557,1.16189,
                        1.1581,1.15421,1.15024,1.14621,1.14213,1.138,1.13385,
                        1.12967,1.12548,1.12128,1.11707,1.11286,1.10865,1.10444,
                        1.10023,1.09602,1.09181,1.0876,1.08337,1.07914,1.07489,
                        1.07061,1.0663,1.06194,1.05752,1.05303,1.04846,1.04377,
                        1.03895,1.03396,1.02878,1.02337,1.01767,1.01163,1.00517,
                        0.9982,0.99061,0.98224,0.97296,0.96257,0.951,0.93831,
                        0.92494,0.91128,0.90142])

m_e = u_e/np.sqrt(1.4*287*288.15)

u_e_v = u_e

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

delta_d = np.array([0.0000716,0.0000727,0.0000755,0.0000801,0.0000862,0.0000939,
                    0.000103,0.0001135,0.000125,0.0001355,0.0001474,0.0001633,0.0001812,
                    0.0002005,0.0002214,0.000244,0.0002684,0.0002948,0.000323,0.0003532,
                    0.0003854,0.0004187,0.0004526,0.0004902,0.0005339,0.0005818,0.0006328,0.0006856,
                    0.0007393,0.0007937,0.0008487,0.0009043,0.0009604,0.0010171,0.0010743,
                    0.0011322,0.0011906,0.0012496,0.0013093,0.0013696,0.0014305,0.0014921,
                    0.0015543,0.0016171,0.0016806,0.0017447,0.0018095,0.0018749,0.0019411,
                    0.002008,0.0020756,0.0021441,0.0022134,0.0022836,0.0023548,0.0024272,
                    0.0025008,0.0025757,0.0026522,0.0027305,0.0028108,0.0028935,0.0029788,
                    0.0030674,0.0031598,0.0032567,0.003359,0.003468,0.0035851,0.0037124,
                    0.0038525,0.0040091,0.0041871,0.004393,0.0046355,0.0049243,0.0052681,
                    0.0056682,0.0061259,0.0064905])

delta_m = np.array([0.0000321,0.0000326,0.0000337,0.0000356,0.0000381,0.0000412,0.0000449,
                    0.000049,0.0000536,0.0000597,0.0000678,0.0000774,0.0000882,0.0001002,
                    0.0001135,0.0001284,0.000145,0.0001636,0.0001844,0.0002075,0.0002328,
                    0.0002602,0.0002895,0.0003204,0.0003534,0.000388,0.0004236,0.00046,
                    0.000497,0.0005345,0.0005725,0.000611,0.0006499,0.0006893,0.0007292,
                    0.0007695,0.0008102,0.0008514,0.000893,0.0009351,0.0009776,0.0010206,
                    0.001064,0.0011078,0.0011521,0.0011968,0.0012419,0.0012876,0.0013336,
                    0.0013802,0.0014272,0.0014748,0.0015228,0.0015715,0.0016208,0.0016708,
                    0.0017214,0.0017729,0.0018254,0.0018788,0.0019333,0.0019892,0.0020466,
                    0.0021058,0.002167,0.0022307,0.0022972,0.0023672,0.0024415,0.0025209,
                    0.0026067,0.0027007,0.0028049,0.0029221,0.0030556,0.0032087,0.0033832,
                    0.0035765,0.0037858,0.003947])

delta_k = np.array([0.000052,0.0000528,0.0000546,0.0000573,0.0000612,0.0000661,0.0000719,
                    0.0000785,0.0000857,0.0000959,0.0001098,0.0001261,0.0001445,0.0001651,
                    0.0001881,0.0002141,0.0002433,0.0002762,0.0003132,0.0003547,0.0004007,
                    0.0004509,0.0005053,0.0005625,0.0006218,0.0006828,0.0007451,0.0008084,
                    0.0008727,0.0009379,0.0010039,0.0010709,0.0011387,0.0012072,0.0012766,
                    0.0013468,0.0014178,0.0014895,0.0015621,0.0016354,0.0017094,0.0017843,
                    0.0018598,0.0019362,0.0020132,0.0020911,0.0021697,0.0022491,0.0023292,
                    0.0024102,0.002492,0.0025746,0.0026582,0.0027427,0.0028283,0.0029149,
                    0.0030028,0.003092,0.0031827,0.003275,0.0033691,0.0034654,0.003564,
                    0.0036655,0.0037702,0.0038787,0.0039917,0.0041101,0.0042351,0.004368,
                    0.0045107,0.0046656,0.0048359,0.0050254,0.0052385,0.0054793,0.0057491,
                    0.0060424,0.0063532,0.0065906])

shape_d = np.array([2.2295,2.2319,2.2398,2.2503,2.263,2.2798,2.2963,
                    2.314,2.3308,2.2713,2.1741,2.1108,2.0549,2.0015,
                    1.9499,1.8998,1.8506,1.802,1.7517,1.7024,1.6552,
                    1.609,1.5636,1.5301,1.511,1.4997,1.494,1.4904,
                    1.4876,1.485,1.4825,1.4801,1.4777,1.4755,1.4734,
                    1.4714,1.4695,1.4678,1.4661,1.4647,1.4633,1.462,
                    1.4608,1.4597,1.4587,1.4578,1.457,1.4562,1.4555,
                    1.4549,1.4543,1.4538,1.4534,1.4531,1.4529,1.4528,
                    1.4527,1.4528,1.453,1.4534,1.4539,1.4546,1.4555,
                    1.4567,1.4581,1.46,1.4622,1.465,1.4684,1.4727,
                    1.4779,1.4845,1.4928,1.5034,1.517,1.5346,1.5571,
                    1.5848,1.6181,1.6444])

shape_km = DrelaGilesTurbulentMOD._shape_km(shape_d,m_e)

c_tau_sqrt = np.array([0.,0.,0.,0.0004989,0.0012249,0.0034805,0.0098505,
                    0.0232821,0.0399639,0.0505921,0.0541443,0.054876,0.0549041,0.0546055,
                    0.0540793,0.0533736,0.0525152,0.0515203,0.0503813,0.0491043,0.0477455,
                    0.046346,0.044928,0.0436602,0.0427083,0.0420378,0.0415642,0.0412314,
                    0.0409889,0.0408021,0.0406499,0.0405198,0.0404046,0.0403003,0.0402044,
                    0.0401155,0.0400326,0.0399552,0.0398828,0.0398151,0.0397518,0.0396925,
                    0.0396372,0.0395855,0.0395373,0.0394925,0.0394509,0.0394124,0.0393769,
                    0.0393444,0.039315,0.0392885,0.0392651,0.039245,0.0392281,0.0392149,
                    0.0392054,0.0392001,0.0391992,0.0392034,0.0392132,0.0392293,0.0392525,
                    0.0392839,0.0393247,0.0393765,0.039441,0.0395208,0.0396187,0.0397386,
                    0.0398852,0.0400651,0.0402865,0.0405599,0.0408976,0.0413115,0.041808,
                    0.0423762,0.0430021,0.0434824])

c_tau_eq_sqrt = np.array([0.01455400,0.02906110,0.03858130,0.04524120,0.05022360,0.05405540,0.05502500,
                    0.05452390,0.05470700,0.05470590,0.05442430,0.05390350,0.05317490,0.05225430,
                    0.05114880,0.04974580,0.04817620,0.04651140,0.04470280,0.04274360,0.04128130,
                    0.04050640,0.04009430,0.03997320,0.03994520,0.03993380,0.03991430,0.03988220,
                    0.03983950,0.03978960,0.03973580,0.03968060,0.03962570,0.03957250,0.03952150,
                    0.03947330,0.03942790,0.03938570,0.03934640,0.03931010,0.03927680,0.03924640,
                    0.03921880,0.03919400,0.03917210,0.03915310,0.03913710,0.03912420,0.03911470,
                    0.03910870,0.03910670,0.03910900,0.03911620,0.03912880,0.03914750,0.03917320,
                    0.03920700,0.03925010,0.03930380,0.03937020,0.03945120,0.03954960,0.03966870,
                    0.03981270,0.03998670,0.04019750,0.04045400,0.04076770,0.04115440,0.04163510,
                    0.04223800,0.04299870,0.04395680,0.04514460,0.04655570,0.04818430,0.04942670
                    ])

re_delta_m = np.array([2.393,7.332,12.532,18.046,23.836,29.973,36.444,
                        43.246,50.367,58.794,69.361,81.633,95.35,110.547,
                        127.377,146.05,166.809,189.894,215.586,243.966,274.905,
                        308.126,343.277,380.187,419.385,460.224,501.981,544.295,
                        586.968,629.897,673.031,716.342,759.815,803.438,847.205,
                        891.109,935.147,979.313,1023.603,1068.015,1112.544,1157.189,
                        1201.947,1246.817,1291.8,1336.897,1382.109,1427.442,1472.902,
                        1518.499,1564.243,1610.15,1656.241,1702.538,1749.072,1795.878,
                        1843.,1890.491,1938.414,1986.846,2035.879,2085.623,2136.212,
                        2187.809,2240.615,2294.874,2350.892,2409.055,2469.849,2533.904,
                        2602.031,2675.292,2755.059,2843.054,2941.265,3051.505,3174.517,
                        3308.084,3449.955,3557.899,])

shape_k = np.array([1.6211,1.6207,1.6194,1.609,1.6074,1.6052,1.603,
                    1.6008,1.5987,1.6063,1.6197,1.6294,1.6385,1.6477,
                    1.6572,1.667,1.6772,1.6879,1.6987,1.7096,1.7209,
                    1.733,1.7458,1.7557,1.7598,1.7601,1.7589,1.7574,
                    1.756,1.7547,1.7536,1.7527,1.752,1.7514,1.7508,
                    1.7503,1.7499,1.7495,1.7492,1.7489,1.7486,1.7483,
                    1.748,1.7477,1.7475,1.7472,1.747,1.7468,1.7465,
                    1.7463,1.7461,1.7458,1.7455,1.7453,1.745,1.7447,
                    1.7444,1.744,1.7436,1.7431,1.7426,1.7421,1.7414,
                    1.7407,1.7398,1.7388,1.7376,1.7363,1.7347,1.7327,
                    1.7304,1.7276,1.7241,1.7198,1.7144,1.7076,1.6993,
                    1.6894,1.6782,1.6698])

c_f = np.array([0.0016655,0.0049475,0.0078036,0.0099299,0.0112698,0.0118324,0.0118572,
                0.0114996,0.0109431,0.0111942,0.0116556,0.0114496,0.0110782,0.010655,
                0.0102051,0.0097416,0.0092742,0.0088112,0.0083774,0.0079523,0.0075373,
                0.0071516,0.0068003,0.0067657,0.006733,0.0066246,0.0064693,0.0063057,
                0.0061479,0.0059999,0.0058617,0.0057324,0.0056107,0.0054955,0.0053859,
                0.0052813,0.0051811,0.0050849,0.0049922,0.0049029,0.0048168,0.0047335,
                0.0046529,0.0045749,0.0044992,0.0044258,0.0043545,0.0042852,0.0042177,
                0.0041519,0.0040876,0.0040248,0.0039632,0.0039027,0.0038432,0.0037845,
                0.0037263,0.0036686,0.0036111,0.0035536,0.0034957,0.0034373,0.0033781,
                0.0033175,0.0032554,0.003191,0.0031239,0.0030534,0.0029787,0.0028986,
                0.0028119,0.0027171,0.0026122,0.0024949,0.002363,0.0022151,0.0020519,
                0.0018795,0.0017027,0.0015781])

c_D = np.array([0.0000357,0.0003208,0.0008397,0.0014596,0.0020646,0.0025514,0.0028909,
                0.003088,0.0031735,0.0038139,0.0042343,0.0044197,0.0045098,0.0045409,
                0.004528,0.0044808,0.0044068,0.0043127,0.0042037,0.0040849,0.0039651,
                0.0038523,0.0037507,0.0036512,0.0035507,0.0034548,0.0033618,0.0032762,
                0.0031978,0.0031252,0.0030573,0.002993,0.0029316,0.0028728,0.0028161,
                0.0027614,0.0027085,0.0026573,0.0026077,0.0025596,0.002513,0.0024677,
                0.0024238,0.0023812,0.0023397,0.0022994,0.0022603,0.0022222,0.0021851,
                0.0021489,0.0021136,0.0020792,0.0020455,0.0020126,0.0019803,0.0019485,
                0.0019174,0.0018866,0.0018562,0.0018262,0.0017963,0.0017665,0.0017368,
                0.0017069,0.0016768,0.0016464,0.0016154,0.0015837,0.0015511,0.0015173,
                0.0014821,0.0014451,0.0014061,0.0013647,0.001321,0.0012752,0.0012289,
                0.0011845,0.0011442,0.0011185])


print('c_tau initial ratio from data is:')
print(c_tau_sqrt[3]/c_tau_eq_sqrt[0])

start = 3 #Very first value with a c_tau
start = 4 #Location where the c_tau derivatives start to roughtly line up
print('START IS: '+str(s_ref[start]))
endpos = s_ref[-1]

dg_visc = DrelaGilesTurbulentMOD(nu=nu_inf, U_e=[s_ref, u_e],show_prog=False,c_tau_init=c_tau_sqrt[start]**2,src=True)
hm_visc = HeadMethod(nu=nu_inf, U_e=[s_ref, u_e])
fig, u_e_prof = plt.subplots(constrained_layout=True)
u_e_prof.plot(s_ref,u_e,color='#9FC9CD',label='Viscous')
u_e_prof.plot(s_ref,u_e_inv,color='#9FC9CD',label='Inviscid',linestyle='--')
u_e_prof.set_ylabel(r'$u_e$ [m/s]')
u_e_prof.set_xlabel(r's [m]')
u_e_prof.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'u_e_profs.png')


fig, du_e_prof = plt.subplots()
du_e_prof.plot(s_ref,dg_visc.du_e(s_ref),color='#BD8B13')
du_e_prof.set_title('du_e viscous profile')

dg_visc.initial_delta_m = delta_m[start]
dg_visc.initial_shape_d = shape_d[start]
hm_visc.initial_delta_m = delta_m[start]
hm_visc.initial_shape_d = shape_d[start]

rtn = dg_visc.solve(x0=s_ref[start],x_end=endpos)

print(rtn.message)
print(rtn.x_end)

rtn2 = hm_visc.solve(x0=s_ref[start],x_end=endpos)

print(rtn2.message)
print(rtn2.x_end)

# fig, ct_test = plt.subplots()
# ct_test.plot(s_ref[start:],c_tau_sqrt[start:],label='XFOIL',color='#FF6A39',linestyle='--',marker='o',markersize=4)
# ct_test.plot(np.array(dg_visc.xvec),dg_visc.ctvec,label='PyBL',color='#FF6A39')
# if rtn.success:
#     ct_test.plot(s_ref[start:],dg_visc.c_tau(s_ref[start:])**.5,label='PyBL output',marker='s',markersize=4,color='#FF6A39')
#     pass
# ct_test.legend()
# ct_test.set_title('c_tau test')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.delta_d(s_ref[start:]),label=r'D-G',color='#A4D65E')
dels.plot(s_ref[start:],hm_visc.delta_d(s_ref[start:]),label=r"Head",color='#A4D65E',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.delta_d(s_ref[start:])-delta_d[start:])/delta_d[start:],label=r'D-G',color='#A4D65E')
dels_err.plot(s_ref[start:],np.abs(hm_visc.delta_d(s_ref[start:])-delta_d[start:])/delta_d[start:],label=r"Head",color='#A4D65E',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.delta_m(s_ref[start:]),label=r'D-G',color='#BB00FF')
dels.plot(s_ref[start:],hm_visc.delta_m(s_ref[start:]),label=r"Head",color='#BB00FF',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.delta_m(s_ref[start:])-delta_m[start:])/delta_m[start:],label=r'D-G',color='#BB00FF')
dels_err.plot(s_ref[start:],np.abs(hm_visc.delta_m(s_ref[start:])-delta_m[start:])/delta_m[start:],label=r"Head",color='#BB00FF',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.delta_k(s_ref[start:]),label=r'D-G',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.delta_k(s_ref[start:])-delta_k[start:])/delta_k[start:],label=r'D-G',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.shape_d(s_ref[start:]),label=r'D-G',color='#612D00')
dels.plot(s_ref[start:],hm_visc.shape_d(s_ref[start:]),label=r"Head",color='#612D00',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.shape_d(s_ref[start:])-shape_d[start:])/shape_d[start:],label=r'D-G',color='#612D00')
dels_err.plot(s_ref[start:],np.abs(hm_visc.shape_d(s_ref[start:])-shape_d[start:])/shape_d[start:],label=r"Head",color='#612D00',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_d$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_d_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.shape_k(s_ref[start:]),label=r'D-G',color='#FF8400')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.shape_k(s_ref[start:])-shape_k[start:])/shape_k[start:],label=r'D-G',color='#FF8400')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_k$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_k_visc.png')

cf_dg = dg_visc.tau_w(s_ref,rho_inf)/(.5*rho_inf*u_inf**2)
cD_dg = dg_visc.dissipation(s_ref,rho_inf)/(rho_inf*u_inf**3)
cf_hm = hm_visc.tau_w(s_ref,rho_inf)/(.5*rho_inf*u_inf**2)

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_f,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],cf_dg[start:],label=r'D-G',color='#818181')
dels.plot(s_ref[start:],cf_hm[start:],label=r"Head",color='#818181',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],abs(c_f[start:]-cf_dg[start:])/abs(c_f[start:]),label=r'D-G',color='#818181')
dels_err.plot(s_ref[start:],abs(c_f[start:]-cf_hm[start:])/abs(c_f[start:]),label=r"Head",color='#818181',linestyle=':')
dels.set_ylabel(r'$c_f$')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_f_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_D,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],cD_dg[start:],label=r'D-G',color='#5CB8B2')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],abs(c_D[start:]-cD_dg[start:])/abs(c_D[start:]),label=r'D-G',color='#5CB8B2')
dels.set_ylabel(r'$c_D$')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_D_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_tau_sqrt**2,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.c_tau(s_ref[start:]),label=r'D-G',color='#A50000')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.c_tau(s_ref[start:])-c_tau_sqrt[start:]**2)/c_tau_sqrt[start:]**2,label=r'D-G',color='#A50000')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$c_{\tau}$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_tau_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref[3:],c_tau_eq_sqrt**2,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.c_tau_eq(s_ref[start:]),label=r'D-G',color="#805F0B")
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.c_tau_eq(s_ref[start:])-c_tau_eq_sqrt[int(start-(len(c_tau_sqrt)-len(c_tau_eq_sqrt))):]**2)/c_tau_eq_sqrt[int(start-(len(c_tau_sqrt)-len(c_tau_eq_sqrt))):]**2,label=r'D-G',color='#805F0B')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$c_{\tau EQ}$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_tau_EQ_visc.png')


#Inviscid profile
u_e = u_e_inv
endpos = s_ref[-1]

dg_visc = DrelaGilesTurbulentMOD(nu=nu_inf, U_e=[s_ref, u_e],show_prog=False,c_tau_init=c_tau_sqrt[start]**2,src=True,cf_crit=min(c_f)/10.)
hm_visc = HeadMethod(nu=nu_inf, U_e=[s_ref, u_e])
fig, u_e_prof = plt.subplots()
u_e_prof.plot(s_ref,u_e,color='#154734')
u_e_prof.set_title('u_e viscous profile')
fig, du_e_prof = plt.subplots()
du_e_prof.plot(s_ref,dg_visc.du_e(s_ref),color='#BD8B13')
du_e_prof.set_title('du_e viscous profile')

dg_visc.initial_delta_m = delta_m[start]
dg_visc.initial_shape_d = shape_d[start]
hm_visc.initial_delta_m = delta_m[start]
hm_visc.initial_shape_d = shape_d[start]

rtn = dg_visc.solve(x0=s_ref[start],x_end=endpos)

print(rtn.message)
print(rtn.x_end)
temp = s_ref[s_ref<rtn.x_end]
dg_end_idx = int(len(temp)-1)

rtn2 = hm_visc.solve(x0=s_ref[start],x_end=endpos)
temp = s_ref[s_ref<rtn2.x_end]
hm_end_idx = int(len(temp)-1)

print(rtn2.message)
print(rtn2.x_end)


fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.delta_d(s_ref[start:dg_end_idx]),label=r'D-G',color='#A4D65E')
dels.plot(s_ref[start:hm_end_idx],hm_visc.delta_d(s_ref[start:hm_end_idx]),label=r"Head",color='#A4D65E',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.delta_d(s_ref[start:dg_end_idx])-delta_d[start:dg_end_idx])/delta_d[start:dg_end_idx],label=r'D-G',color='#A4D65E')
dels_err.plot(s_ref[start:hm_end_idx],np.abs(hm_visc.delta_d(s_ref[start:hm_end_idx])-delta_d[start:hm_end_idx])/delta_d[start:hm_end_idx],label=r"Head",color='#A4D65E',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.delta_m(s_ref[start:dg_end_idx]),label=r'D-G',color='#BB00FF')
dels.plot(s_ref[start:hm_end_idx],hm_visc.delta_m(s_ref[start:hm_end_idx]),label=r"Head",color='#BB00FF',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.delta_m(s_ref[start:dg_end_idx])-delta_m[start:dg_end_idx])/delta_m[start:dg_end_idx],label=r'D-G',color='#BB00FF')
dels_err.plot(s_ref[start:hm_end_idx],np.abs(hm_visc.delta_m(s_ref[start:hm_end_idx])-delta_m[start:hm_end_idx])/delta_m[start:hm_end_idx],label=r"Head",color='#BB00FF',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.delta_k(s_ref[start:dg_end_idx]),label=r'D-G',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.delta_k(s_ref[start:dg_end_idx])-delta_k[start:dg_end_idx])/delta_k[start:dg_end_idx],label=r'D-G',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.shape_d(s_ref[start:dg_end_idx]),label=r'D-G',color='#612D00')
dels.plot(s_ref[start:hm_end_idx],hm_visc.shape_d(s_ref[start:hm_end_idx]),label=r"Head",color='#612D00',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.shape_d(s_ref[start:dg_end_idx])-shape_d[start:dg_end_idx])/shape_d[start:dg_end_idx],label=r'D-G',color='#612D00')
dels_err.plot(s_ref[start:hm_end_idx],np.abs(hm_visc.shape_d(s_ref[start:hm_end_idx])-shape_d[start:hm_end_idx])/shape_d[start:hm_end_idx],label=r"Head",color='#612D00',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_d$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_d_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.shape_k(s_ref[start:dg_end_idx]),label=r'D-G',color='#FF8400')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.shape_k(s_ref[start:dg_end_idx])-shape_k[start:dg_end_idx])/shape_k[start:dg_end_idx],label=r'D-G',color='#FF8400')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_k$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_k_inv.png')

cf_dg = dg_visc.tau_w(s_ref,rho_inf)/(.5*rho_inf*u_inf**2)
cD_dg = dg_visc.dissipation(s_ref,rho_inf)/(rho_inf*u_inf**3)
cf_hm = hm_visc.tau_w(s_ref,rho_inf)/(.5*rho_inf*u_inf**2)

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_f,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],cf_dg[start:dg_end_idx],label=r'D-G',color='#818181')
dels.plot(s_ref[start:hm_end_idx],cf_hm[start:hm_end_idx],label=r"Head",color='#818181',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],abs(c_f[start:dg_end_idx]-cf_dg[start:dg_end_idx])/abs(c_f[start:dg_end_idx]),label=r'D-G',color='#818181')
dels_err.plot(s_ref[start:hm_end_idx],abs(c_f[start:hm_end_idx]-cf_hm[start:hm_end_idx])/abs(c_f[start:hm_end_idx]),label=r"Head",color='#818181',linestyle=':')
dels.set_ylabel(r'$c_f$')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_f_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_D,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],cD_dg[start:dg_end_idx],label=r'D-G',color='#5CB8B2')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],abs(c_D[start:dg_end_idx]-cD_dg[start:dg_end_idx])/abs(c_D[start:dg_end_idx]),label=r'D-G',color='#5CB8B2')
dels.set_ylabel(r'$c_D$')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_D_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_tau_sqrt**2,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.c_tau(s_ref[start:dg_end_idx]),label=r'D-G',color="#A50000")
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.c_tau(s_ref[start:dg_end_idx])-c_tau_sqrt[start:dg_end_idx]**2)/c_tau_sqrt[start:dg_end_idx]**2,label=r'D-G',color='#A50000')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$c_{\tau}$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_tau_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref[3:],c_tau_eq_sqrt**2,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:dg_end_idx],dg_visc.c_tau_eq(s_ref[start:dg_end_idx]),label=r'D-G',color='#805F0B')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:dg_end_idx],np.abs(dg_visc.c_tau_eq(s_ref[start:dg_end_idx])-c_tau_eq_sqrt[int(start-(len(c_tau_sqrt)-len(c_tau_eq_sqrt))):int(dg_end_idx-(len(c_tau_sqrt)-len(c_tau_eq_sqrt)))]**2)/c_tau_eq_sqrt[int(start-(len(c_tau_sqrt)-len(c_tau_eq_sqrt))):int(dg_end_idx-(len(c_tau_sqrt)-len(c_tau_eq_sqrt)))]**2,label=r'D-G',color='#805F0B')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$c_{\tau EQ}$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_tau_EQ_inv.png')



start = 25 #Later locations make stuff line up better, to conclude there are some leading edge treatments that xfoil does that I don't do
print('START IS: '+str(s_ref[start]))

dg_visc = DrelaGilesTurbulentMOD(nu=nu_inf, U_e=[s_ref, u_e_v],show_prog=False,c_tau_init=c_tau_sqrt[start]**2,src=True)
hm_visc = HeadMethod(nu=nu_inf, U_e=[s_ref, u_e_v])
fig, u_e_prof = plt.subplots(constrained_layout=True)
u_e_prof.plot(s_ref,u_e,color='#154734',marker='o',markersize=4,label='Viscous')
u_e_prof.plot(s_ref,u_e_inv,color='#154734',marker='o',markersize=4,label='Inviscid',linestyle='--')
u_e_prof.set_ylabel(r'$u_e$ [m/s]')
u_e_prof.set_xlabel(r's [m]')
u_e_prof.legend(ncol=2,borderaxespad=-5.5)
fig, du_e_prof = plt.subplots()
du_e_prof.plot(s_ref,dg_visc.du_e(s_ref),color='#BD8B13')
du_e_prof.set_title('du_e viscous profile')

dg_visc.initial_delta_m = delta_m[start]
dg_visc.initial_shape_d = shape_d[start]
hm_visc.initial_delta_m = delta_m[start]
hm_visc.initial_shape_d = shape_d[start]

rtn = dg_visc.solve(x0=s_ref[start],x_end=endpos)

print(rtn.message)
print(rtn.x_end)

rtn2 = hm_visc.solve(x0=s_ref[start],x_end=endpos)

print(rtn2.message)
print(rtn2.x_end)

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.delta_d(s_ref[start:]),label=r'D-G',color='#A4D65E')
dels.plot(s_ref[start:],hm_visc.delta_d(s_ref[start:]),label=r"Head",color='#A4D65E',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.delta_d(s_ref[start:])-delta_d[start:])/delta_d[start:],label=r'D-G',color='#A4D65E')
dels_err.plot(s_ref[start:],np.abs(hm_visc.delta_d(s_ref[start:])-delta_d[start:])/delta_d[start:],label=r"Head",color='#A4D65E',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_visc_late.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.delta_m(s_ref[start:]),label=r'D-G',color='#BB00FF')
dels.plot(s_ref[start:],hm_visc.delta_m(s_ref[start:]),label=r"Head",color='#BB00FF',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.delta_m(s_ref[start:])-delta_m[start:])/delta_m[start:],label=r'D-G',color='#BB00FF')
dels_err.plot(s_ref[start:],np.abs(hm_visc.delta_m(s_ref[start:])-delta_m[start:])/delta_m[start:],label=r"Head",color='#BB00FF',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_visc_late.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.delta_k(s_ref[start:]),label=r'D-G',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.delta_k(s_ref[start:])-delta_k[start:])/delta_k[start:],label=r'D-G',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_visc_late.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref[start:],dg_visc.shape_d(s_ref[start:]),label=r'D-G',color='#612D00')
dels.plot(s_ref[start:],hm_visc.shape_d(s_ref[start:]),label=r"Head",color='#612D00',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref[start:],np.abs(dg_visc.shape_d(s_ref[start:])-shape_d[start:])/shape_d[start:],label=r'D-G',color='#612D00')
dels_err.plot(s_ref[start:],np.abs(hm_visc.shape_d(s_ref[start:])-shape_d[start:])/shape_d[start:],label=r"Head",color='#612D00',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_d_visc_late.png')

plt.show()

#Testing c_tau progression
xf_ct = c_tau_sqrt[3]**2
xf_ct_prog = [xf_ct]
for idx,xct in enumerate(c_tau_sqrt[4:]):
    idx = idx + 4
    delt = delta_m[idx]*(3.15 + 1.72/(shape_km[idx]-1.)) + delta_d[idx]
    dct = (xct**2/delt)*4.2*(c_tau_eq_sqrt[idx-4] - xct)
    #dct = 0.5*(xf_ct_prog[-1]**(-.5))*dct
    temp = xf_ct_prog[-1] + dct*(s_ref[idx] - s_ref[idx-1])
    xf_ct_prog.append(temp)

xf_ct_prog = np.array(xf_ct_prog)

fig, xfct = plt.subplots()
xfct.plot(s_ref[3:],xf_ct_prog**.5,label='manual step',marker='s',markersize=4)
xfct.plot(s_ref,c_tau_sqrt,label='Actual values',marker='s',markersize=4)
xfct.legend()
xfct.set_title('c_tau progression test')

#shape_k derivative
hk_xf_der = np.diff(shape_k)/np.diff(s_ref)
hk_py_der = np.diff(dg_visc.shape_k(s_ref))/np.diff(s_ref)
fig,hk_der_comp = plt.subplots()
hk_der_comp.plot(s_ref[:-1],hk_xf_der,label='XF')
hk_der_comp.plot(s_ref[start:-1],hk_py_der[start:],label='Py')
hk_der_comp.legend()
hk_der_comp.set_title('hk derivative')


#Demos how the IC are different
print('c_tau initial ratio is:')
print(c_tau_sqrt[3]/c_tau_eq_sqrt[0])
print('c_tau initial ratio from the SRC is:')
temp = 1.8*np.exp(-3.3/(shape_km[3]-1.))
print(temp)

plt.show()
pass