import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.drela_giles_laminar_OLD import DrelaGilesLaminar_OLD
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
from ibl.initial_condition import ManualCondition


import os

# NACA 0009 Re = 10000
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

file_name = "NACA0009_Re1E4_VISC"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

# NACA 0009 Re = 1e4
# Case similar to paper by Ozdemir

Re_inf = 1e4
chord = 1
rho_inf = 1.2
U_inf = 20
nu_inf = U_inf/Re_inf

s_ref = np.array([0.0008,0.0025,0.00432,0.00629,0.00846,0.01088,0.01361,
        0.01677,0.02046,0.02489,0.03028,0.03693,0.04511,0.05491,
        0.06616,0.07847,0.09149,0.10495,0.11871,0.13268,0.14679,
        0.16103,0.17536,0.18977,0.20424,0.21877,0.23334,0.24796,
        0.26261,0.2773,0.29201,0.30675,0.32152,0.33631,0.35111,
        0.36594,0.38078,0.39564,0.41052,0.4254,0.4403,0.45522,
        0.47014,0.48507,0.50001,0.51496,0.52992,0.54489,0.55986,
        0.57484,0.58983,0.60482,0.61982,0.63482,0.64983,0.66484,
        0.67985,0.69486,0.70988,0.7249,0.73992,0.75495,0.76997,
        0.785,0.80002,0.81505,0.83007,0.8451,0.86012,0.87514,
        0.89016,0.90517,0.92019,0.93518,0.95014,0.96496,0.97941,
        0.99286,1.00431,1.01217])

u_e_visc_c = np.array([0.10642,0.31831,0.50904,0.66446,0.78422,0.87441,0.94229,
            0.99387,1.03356,1.06443,1.08854,1.10723,1.12132,1.13146,
            1.13828,1.14254,1.14496,1.14609,1.14632,1.1459,1.14502,
            1.14377,1.14225,1.14051,1.1386,1.13654,1.13437,1.13211,
            1.12976,1.12736,1.1249,1.1224,1.11986,1.1173,1.11472,
            1.11212,1.10951,1.10689,1.10427,1.10165,1.09903,1.09642,
            1.09381,1.09121,1.08862,1.08604,1.08347,1.08091,1.07836,
            1.07583,1.07331,1.0708,1.0683,1.06582,1.06335,1.0609,
            1.05845,1.05602,1.0536,1.0512,1.0488,1.04643,1.04406,
            1.04172,1.03939,1.03708,1.03479,1.03252,1.03027,1.02805,
            1.02586,1.0237,1.02158,1.0195,1.01747,1.0155,1.01362,
            1.01191,1.01049,1.00955])

u_e_visc = U_inf*u_e_visc_c

fig, velcurve = plt.subplots()
velcurve.plot(s_ref,u_e_visc,'#154734',marker='o',markersize=4)
velcurve.set_xlabel('s [m]')
velcurve.set_ylabel(r'$u_e$ [m/s]')

delta_m = np.array([0.0002532,0.0002633,0.0002863,0.0003214,0.0003663,0.0004196,0.0004808,
            0.0005499,0.0006278,0.0007163,0.0008174,0.0009331,0.0010645,0.001209,
            0.0013613,0.0015152,0.0016665,0.0018133,0.0019551,0.0020921,0.0022247,
            0.0023535,0.0024787,0.002601,0.0027205,0.0028376,0.0029526,0.0030657,
            0.0031771,0.003287,0.0033955,0.0035026,0.0036087,0.0037137,0.0038177,
            0.0039208,0.0040231,0.0041246,0.0042254,0.0043255,0.0044251,0.004524,
            0.0046224,0.0047203,0.0048178,0.0049148,0.0050114,0.0051077,0.0052037,
            0.0052993,0.0053947,0.0054898,0.0055847,0.0056794,0.005774,0.0058684,
            0.0059628,0.006057,0.0061512,0.0062453,0.0063393,0.0064333,0.0065272,
            0.0066211,0.0067149,0.0068086,0.0069021,0.0069954,0.0070885,0.0071812,
            0.0072734,0.0073651,0.0074562,0.0075463,0.0076354,0.0077227,0.0078067,
            0.0078837,0.0079485,0.007992])

delta_d = np.array([0.0005644,0.0005889,0.0006457,0.0007318,0.0008431,0.0009766,0.0011309,
            0.0013067,0.0015066,0.0017349,0.0019973,0.0022994,0.0026436,0.0030238,
            0.0034259,0.0038339,0.0042367,0.0046295,0.005011,0.0053819,0.0057434,
            0.0060967,0.0064432,0.0067838,0.0071197,0.0074515,0.0077802,0.0081062,
            0.0084301,0.0087525,0.0090736,0.0093939,0.0097137,0.0100333,0.0103529,
            0.0106727,0.0109931,0.0113141,0.0116361,0.0119591,0.0122835,0.0126093,
            0.0129369,0.0132663,0.013598,0.013932,0.0142687,0.0146084,0.0149513,
            0.0152979,0.0156484,0.0160034,0.0163632,0.0167284,0.0170994,0.0174768,
            0.0178612,0.0182534,0.018654,0.0190637,0.0194834,0.019914,0.0203563,
            0.0208114,0.0212802,0.0217639,0.0222635,0.0227801,0.023315,0.0238694,
            0.0244443,0.025041,0.0256605,0.0263036,0.0269698,0.0276556,0.0283486,
            0.0290137,0.029596,0.0299601])

delta_k = np.array([0.0004104,0.0004265,0.000463,0.0005187,0.0005897,0.0006741,0.0007707,
                    0.0008797,0.0010025,0.0011417,0.0013007,0.0014827,0.001689,0.001916,
                    0.002155,0.0023965,0.0026337,0.0028637,0.0030857,0.0032999,0.0035071,
                    0.0037081,0.0039034,0.0040937,0.0042796,0.0044616,0.00464,0.0048153,
                    0.0049877,0.0051574,0.0053249,0.0054901,0.0056534,0.0058149,0.0059747,
                    0.0061329,0.0062896,0.006445,0.0065991,0.0067521,0.0069039,0.0070546,
                    0.0072044,0.0073532,0.0075012,0.0076483,0.0077947,0.0079403,0.0080853,
                    0.0082296,0.0083733,0.0085164,0.008659,0.0088011,0.0089428,0.0090841,
                    0.009225,0.0093655,0.0095057,0.0096455,0.0097851,0.0099243,0.0100633,
                    0.0102019,0.0103402,0.0104782,0.0106157,0.0107528,0.0108894,0.0110254,
                    0.0111607,0.0112952,0.0114288,0.0115613,0.0116923,0.0118209,0.011945,
                    0.0120592,0.0121554,0.0121574])

shape_k = np.array([1.6211,1.62,1.617,1.6137,1.61,1.6064,1.6029,
            1.5997,1.5967,1.5939,1.5913,1.5889,1.5867,1.5847,
            1.5831,1.5816,1.5804,1.5793,1.5783,1.5773,1.5764,
            1.5756,1.5747,1.5739,1.5731,1.5723,1.5715,1.5707,
            1.5699,1.569,1.5682,1.5674,1.5666,1.5658,1.565,
            1.5642,1.5634,1.5626,1.5618,1.561,1.5602,1.5594,
            1.5586,1.5578,1.557,1.5562,1.5554,1.5546,1.5538,
            1.553,1.5521,1.5513,1.5505,1.5497,1.5488,1.548,
            1.5471,1.5462,1.5453,1.5445,1.5436,1.5427,1.5417,
            1.5408,1.5399,1.539,1.538,1.5371,1.5362,1.5353,
            1.5345,1.5336,1.5328,1.532,1.5313,1.5307,1.5301,
            1.5296,1.5293,1.5212])

shape_d = np.array([2.2295,2.2364,2.2549,2.2767,2.3019,2.3272,2.3522,
                    2.3764,2.3996,2.422,2.4436,2.4642,2.4835,2.5011,
                    2.5167,2.5303,2.5423,2.5531,2.5631,2.5725,2.5816,
                    2.5905,2.5994,2.6082,2.617,2.626,2.635,2.6441,
                    2.6534,2.6628,2.6723,2.6819,2.6918,2.7017,2.7118,
                    2.7221,2.7325,2.7431,2.7538,2.7648,2.7759,2.7872,
                    2.7987,2.8105,2.8224,2.8347,2.8472,2.8601,2.8732,
                    2.8868,2.9007,2.9151,2.93,2.9454,2.9614,2.9781,
                    2.9955,3.0136,3.0326,3.0525,3.0734,3.0955,3.1187,
                    3.1432,3.1691,3.1965,3.2256,3.2564,3.2891,3.3239,
                    3.3608,3.3999,3.4415,3.4856,3.5322,3.5811,3.6313,
                    3.6802,3.7235,3.7487,])

T_air = 288.15
R_air = 287.
gamma = 1.4
mach = u_e_visc/np.sqrt(gamma*R_air*T_air)
shape_km = (shape_d - .29*mach**2)/(1. + .113*mach**2)

c_f = np.array([0.0301582,0.0859332,0.123239,0.139129,0.1392117,0.130834,0.1188604,
                0.1059584,0.0933915,0.0816563,0.070957,0.0613659,0.0529711,0.0458617,
                0.040046,0.0353914,0.0316748,0.0286719,0.0262015,0.0241305,0.0223637,
                0.020833,0.0194893,0.0182963,0.0172269,0.0162604,0.0153807,0.0145751,
                0.0138334,0.0131475,0.0125104,0.0119167,0.0113616,0.0108412,0.0103519,
                0.0098908,0.0094553,0.0090432,0.0086525,0.0082813,0.0079281,0.0075915,
                0.00727,0.0069625,0.0066679,0.0063852,0.0061134,0.0058517,0.0055993,
                0.0053553,0.0051191,0.0048901,0.0046676,0.004451,0.0042398,0.0040335,
                0.0038316,0.0036338,0.0034396,0.0032487,0.0030608,0.0028756,0.002693,
                0.0025127,0.0023346,0.0021588,0.0019851,0.0018137,0.0016447,0.0014782,
                0.0013146,0.001154,0.000997,0.000844,0.0006959,0.0005539,0.000421,
                0.000303,0.0002071,0.0001779,])

c_D = np.array([0.0009225,0.0078992,0.0183515,0.0274751,0.0330763,0.0353744,0.0353709,
                0.0339763,0.0318164,0.0292702,0.0265712,0.023871,0.0212961,0.0189613,
                0.0169457,0.0152642,0.0138795,0.0127348,0.0117775,0.0109652,0.0102664,
                0.0096575,0.0091211,0.0086439,0.0082158,0.0078289,0.0074771,0.0071553,
                0.0068596,0.0065866,0.0063338,0.0060987,0.0058795,0.0056745,0.0054823,
                0.0053018,0.0051318,0.0049714,0.0048199,0.0046765,0.0045405,0.0044114,
                0.0042886,0.0041718,0.0040604,0.0039541,0.0038525,0.0037554,0.0036624,
                0.0035732,0.0034876,0.0034055,0.0033265,0.0032506,0.0031774,0.003107,
                0.003039,0.0029735,0.0029103,0.0028492,0.0027902,0.0027331,0.002678,
                0.0026248,0.0025733,0.0025236,0.0024756,0.0024292,0.0023845,0.0023414,
                0.0022998,0.0022599,0.0022214,0.0021845,0.0021492,0.0021157,0.0020843,
                0.0020563,0.0020334,0.0040912,])

n_tilda = np.array([0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.,
                    0.0002848,0.0016371,0.0049209,0.0109398,0.0203366,0.033563,0.0508529,
                    0.072188,0.0972541,0.1253855,0.1555196,0.1868028,0.2190255,0.2522243,
                    0.2864344,0.3216878,0.3580099,0.3954082,0.4338357,0.4730763,0.512416,
                    0.5500006,0.5827707,0.0473897,])

re_delta_m = np.array([0.269,0.838,1.458,2.136,2.873,3.669,4.53,
                        5.465,6.489,7.625,8.897,10.332,11.936,13.679,
                        15.495,17.312,19.081,20.782,22.412,23.973,25.473,
                        26.918,28.313,29.664,30.975,32.251,33.494,34.708,
                        35.894,37.056,38.195,39.314,40.412,41.493,42.556,
                        43.604,44.636,45.655,46.66,47.652,48.633,49.602,
                        50.56,51.509,52.447,53.377,54.297,55.21,56.114,
                        57.011,57.901,58.785,59.662,60.533,61.398,62.258,
                        63.113,63.963,64.809,65.65,66.487,67.32,68.149,
                        68.973,69.794,70.61,71.422,72.229,73.03,73.826,
                        74.615,75.397,76.171,76.935,77.688,78.424,79.13,
                        79.776,80.319,80.683])


# Run PyBL (DG Laminar)

ic = ManualCondition(delta_d=delta_d[0],delta_m=delta_m[0],delta_k=delta_m[0]*shape_k[0])

n_crit = 9.
#u_e_visc[0] = 0
DG_testrun = DrelaGilesLaminarMOD(nu=nu_inf, U_e=[s_ref,u_e_visc],n_tilde_crit=n_crit,ic=ic,show_prog=False,src=True)
#DG_testrun = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref,u_e_visc])
#DG_testrun.initial_delta_m = delta_m[0]
fig, due_plot = plt.subplots()
due_plot.plot(s_ref,DG_testrun.du_e(s_ref),color='#BD8B13')
due_plot.set_title('du_e_dx')
due_plot.set_ylim([-4.,4.])
#plt.show()

rtn = DG_testrun.solve(s_ref[0],s_ref[-1])
print(rtn.message)
print(rtn.x_end)

TM_testrun = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref,u_e_visc])
TM_testrun.initial_delta_m = delta_m[0]
rtn = TM_testrun.solve(s_ref[0],s_ref[-1])
print(rtn.message)
print(rtn.x_end)


fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,DG_testrun.delta_d(s_ref),label=r'D-G',color='#A4D65E')
dels.plot(s_ref,TM_testrun.delta_d(s_ref),label=r"Thwaites",color='#A4D65E',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(DG_testrun.delta_d(s_ref)-delta_d)/delta_d,label=r'D-G',color='#A4D65E')
dels_err.plot(s_ref,np.abs(TM_testrun.delta_d(s_ref)-delta_d)/delta_d,label=r"Thwaites",color='#A4D65E',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,DG_testrun.delta_m(s_ref),label=r'D-G',color='#BB00FF')
dels.plot(s_ref,TM_testrun.delta_m(s_ref),label=r"Thwaites",color='#BB00FF',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(DG_testrun.delta_m(s_ref)-delta_m)/delta_m,label=r'D-G',color='#BB00FF')
dels_err.plot(s_ref,np.abs(TM_testrun.delta_m(s_ref)-delta_m)/delta_m,label=r"Thwaites",color='#BB00FF',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,DG_testrun.delta_k(s_ref),label=r'D-G',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(DG_testrun.delta_k(s_ref)-delta_k)/delta_k,label=r'D-G',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_visc.png')


fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,DG_testrun.shape_d(s_ref),label=r'D-G',color='#612D00')
dels.plot(s_ref,TM_testrun.shape_d(s_ref),label=r"Thwaites",color='#612D00',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(DG_testrun.shape_d(s_ref)-shape_d)/shape_d,label=r'D-G',color='#612D00')
dels_err.plot(s_ref,np.abs(TM_testrun.shape_d(s_ref)-shape_d)/shape_d,label=r"Thwaites",color='#612D00',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_d$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_d_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,DG_testrun.shape_k(s_ref),label=r'D-G',color='#FF8400')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,np.abs(DG_testrun.shape_k(s_ref)-shape_k)/shape_k,label=r'D-G',color='#FF8400')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_k$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_k_visc.png')


cf_dg = DG_testrun.tau_w(s_ref,rho_inf)/(.5*rho_inf*U_inf**2)
cD_dg = DG_testrun.dissipation(s_ref,rho_inf)/(rho_inf*U_inf**3)
cf_tw = TM_testrun.tau_w(s_ref,rho_inf)/(.5*rho_inf*U_inf**2)

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_f,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,cf_dg,label=r'D-G',color='#818181')
dels.plot(s_ref,cf_tw,label=r"Thwaites",color='#818181',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,abs(c_f-cf_dg)/abs(c_f),label=r'D-G',color='#818181')
dels_err.plot(s_ref,abs(c_f-cf_tw)/abs(c_f),label=r"Thwaites",color='#818181',linestyle=':')
dels.set_ylabel(r'$c_f$')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_f_visc.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,c_D,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(s_ref,cD_dg,label=r'D-G',color='#5CB8B2')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(s_ref,abs(c_D-cD_dg)/abs(c_D),label=r'D-G',color='#5CB8B2')
dels.set_ylabel(r'$c_D$')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_D_visc.png')

fig, n_tild = plt.subplots(constrained_layout=True)
n_tild.plot(s_ref,n_tilda,label='XFOIL',linestyle='--',color='black')
n_tild.plot(s_ref,DG_testrun.n_tilde(s_ref),label='D-G',color='#FF6A39')
n_tild.set_xlabel('s [m]')
n_tild.set_ylabel(r'$\tilde{n}$')
n_tild.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'n_tild_visc.png')

plt.show()
pass