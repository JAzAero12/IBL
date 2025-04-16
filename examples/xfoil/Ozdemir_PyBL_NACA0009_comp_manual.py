import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.drela_giles_laminar import DrelaGilesLaminar
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD
from ibl.initial_condition import ManualCondition

import time

# NACA 0009 Re = 1e4
# Case similar to paper by Ozdemir
START = time.time()
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
m_e_visc = u_e_visc/np.sqrt(1.4*287*288.15)

fig, velcurve = plt.subplots()
velcurve.plot(s_ref,u_e_visc,marker='*',color='#154734',markersize=2)
velcurve.grid(True)
velcurve.set_title(r'$u_e$')

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

shape_km = DrelaGilesLaminarMOD._shape_km(shape_d,m_e_visc)

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

n_tild = np.array([0.,0.,0.,0.,0.,0.,0.,
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

re_delm = np.array([0.269,0.838,1.458,2.136,2.873,3.669,4.53,
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
DG_testrun = DrelaGilesLaminarMOD(nu=nu_inf, U_e=[s_ref,u_e_visc],n_tilde_crit=n_crit,ic=ic,show_prog=False)
DG_old_testrun = DrelaGilesLaminar(nu=nu_inf, U_e=[s_ref,u_e_visc],n_tilde_crit=n_crit,ic=ic)
#DG_testrun = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref,u_e_visc])
#DG_testrun.initial_delta_m = delta_m[0]



due_visc = DG_testrun.du_e(s_ref)
fig, due_plot = plt.subplots()
due_plot.plot(s_ref,due_visc,marker='*',color='#BD8B13',markersize=4)
due_plot.set_title(r'$\frac{du_e}{dx}$')
due_plot.grid(True)
due_plot.set_ylim([-4.,4.])

fd_dh_k_dx2 = []
fd_dd_m_dx2 = []
fd_dd_d_dx2 = []
fd_dh_k_dx = []
fd_dd_m_dx = []
fd_dd_d_dx = []
fd_dh_k_dh_km2 = []
fd_dh_km_dx2 = []
fd_dh_km_dx = []

pybl_dh_k_dx_paper = []
pybl_d_m = []
pybl_d_d = []
pybl_h_km = []
pybl_dh_k_dx_code = []
pybl_hk_h_km = []
pybl_hk_h_km2 = []
pybl_cD = []
pybl_cf = []

chain_der_shape_km = []
chain_der_delta_d = []
chain_der_delta_d_dm_static = []
chain_der_shape_km_denom_static = []
chain_der_shape_km_denom_static_src = []
chain_der_shape_km_numer_static = []
chain_der_shape_km_numer_paper = []
chain_der_delta_h_d_partial_static = []
chain_der_delta_h_d_delm_static = []
chain_der_delta_h_d_full_static = []

start = 60

f_prog = [np.array([delta_m[start],delta_d[start],0.])]

f_prog_old = [np.array([delta_m[start],shape_km[start],0.])]

for idx, s in enumerate(s_ref[start:-1]):
    spacing = s_ref[idx+start+1] - s
    f_p = DrelaGilesLaminarMOD._ode_impl(self=DG_testrun,x=s,f=f_prog[idx])
    f_tmp = f_prog[idx] + spacing*f_p
    f_prog.append(f_tmp)
    f_p_hkm = DrelaGilesLaminar._ode_impl(self=DG_old_testrun,x=s,f=f_prog_old[idx])
    f_tmp_old = f_prog_old[idx] + spacing*f_p_hkm
    f_prog_old.append(f_tmp_old)

f_prog = np.array(f_prog)
f_prog_old = np.array(f_prog_old)

fig, delm_manual = plt.subplots()
delm_manual.plot(s_ref[start:],f_prog[:,0],label=r'$\delta_m$ PyBL, Manual',marker='o',color='#3A913F',markersize=4)
delm_manual.plot(s_ref[start:],delta_m[start:],label=r'$\delta_m$ XFOIL',marker='o',color='#3A913F',markersize=4,linestyle='--')
delm_manual.legend()

fig, deld_manual = plt.subplots()
deld_manual.plot(s_ref[start:],f_prog[:,1],label=r'$\delta_d$ PyBL, Manual',marker='o',color='#A4D65E',markersize=4)
deld_manual.plot(s_ref[start:],delta_d[start:],label=r'$\delta_d$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
deld_manual.legend()

fig, h_km_manual = plt.subplots()
h_km_manual.plot(s_ref[start:],f_prog_old[:,1],label=r'$H_km$ PyBL, Manual',marker='o',color='#A4D65E',markersize=4)
h_km_manual.plot(s_ref[start:],shape_km[start:],label=r'$H_km$ XFOIL',marker='o',color='#A4D65E',markersize=4,linestyle='--')
h_km_manual.legend()


for idx,(s,h_k,h_d,del_m,del_d,h_km,cf,ue) in enumerate(zip(s_ref[1:-1],shape_k[1:-1],shape_d[1:-1],delta_m[1:-1],delta_d[1:-1],shape_km[1:-1],c_f[1:-1],u_e_visc[1:-1])):
    
    #ode_impl
    f = np.array([del_m,del_d,0.])
    f_p = DrelaGilesLaminarMOD._ode_impl(self=DG_testrun,x=s,f=np.array([del_m,del_d,0.])) #Code Derivatives, delta_m and delta_d
    
    pybl_d_m.append(f_p[0])
    pybl_d_d.append(f_p[1])


    idx = idx + 1 #Make sure idx starts from 1
    #print(idx)
    spacing = s_ref[idx+1] - s_ref[idx-1] #change for non uniform spacing later TODO I think its right??????

    #Finite difference velocity derivative
    fd_due = (u_e_visc[idx+1] - u_e_visc[idx-1])/spacing

    #shape k finite difference
    hk_deriv2 = (shape_k[idx+1] - shape_k[idx-1])/spacing #Second order
    hk_deriv = (shape_k[idx+1] - h_k)/(s_ref[idx+1]-s) #First order
    fd_dh_k_dx2.append(hk_deriv2)
    fd_dh_k_dx.append(hk_deriv)

    #delta m finite difference
    dm_deriv2 = (delta_m[idx+1] - delta_m[idx-1])/spacing #Second order
    dm_deriv = (delta_m[idx+1] - del_m)/(s_ref[idx+1]-s) #First order
    fd_dd_m_dx2.append(dm_deriv2)
    fd_dd_m_dx.append(dm_deriv)

    #delta d finite difference
    dd_deriv2 = (delta_d[idx+1] - delta_d[idx-1])/spacing #Second order
    dd_deriv = (delta_d[idx+1] - del_d)/(s_ref[idx+1]-s) #First order
    fd_dd_d_dx2.append(dd_deriv2)
    fd_dd_d_dx.append(dd_deriv)

    #shape_km finite difference
    h_km_deriv2= (shape_km[idx+1] - shape_km[idx-1])/spacing #Second order
    h_km_deriv = (shape_km[idx+1] - h_km)/(s_ref[idx+1]-s) #First order
    fd_dh_km_dx2.append(h_km_deriv2)
    fd_dh_km_dx.append(h_km_deriv)


    #PyBL
    cdtemp = DrelaGilesLaminarMOD._c_D(shape_km=h_km,shape_k=h_k,re_delta_m=re_delm[idx])
    pybl_cD.append(cdtemp)

    cftemp = DrelaGilesLaminarMOD._c_f_dg(shape_km=h_km,re_delta_m=re_delm[idx])
    pybl_cf.append(cftemp)

    #shape k
    hk_deriv_py2 = DrelaGilesLaminarMOD._dshape_k_dx(delta_m=del_m,shape_km=h_km,u_e=ue,du_e=due_visc[idx],m_e=m_e_visc[idx],re_delta_m=re_delm[idx],c_f=cf)

    h_w = 2.*.00094
    h_stst = (.064/(h_km - 0.8))*m_e_visc[idx]**2
    hk_deriv_py = h_k*(2.*c_D[idx]/(del_m*h_k) - cf/(2.*del_m) - (2.*h_stst/h_k + 1. - h_d - h_w/del_m)*(1./ue)*fd_due)

    pybl_dh_k_dx_code.append(hk_deriv_py2)
    pybl_dh_k_dx_paper.append(hk_deriv_py)

    
    #Below is directly from the paper itself, using variables from XFOIL, equation 10
    dm_deriv_py = cf/2. - (2. + h_d - m_e_visc[idx]**2)*(del_m/ue)*fd_due

    #Finite difference dH_k/dH_km
    h_k_h_km_deriv2 = (shape_k[idx+1] - shape_k[idx-1])/(shape_km[idx+1] - shape_km[idx-1])
    fd_dh_k_dh_km2.append(h_k_h_km_deriv2)

    #PyBL dH_k/dH_km
    py_dhk_dhkm = DrelaGilesLaminarMOD._dshape_k_dshape_km(h_km)
    pybl_hk_h_km.append(py_dhk_dhkm)

    py_dhk_dhkm2 = DrelaGilesLaminarMOD._dshape_k_dshape_km(h_km,True) #This lines up with FD
    pybl_hk_h_km2.append(py_dhk_dhkm2)

    #The shape_km streamwise derivative, used in MOD as well
    f2 = np.array([del_m,h_km,0.])
    f_p2 = DrelaGilesLaminar._ode_impl(self=DG_old_testrun,x=s,f=f2)
    pybl_h_km.append(f_p2[1])

    #Sequenced Result variations of dh_km_dx
    chain_dhkm_dx = hk_deriv2/h_k_h_km_deriv2 #FiniteDiff/FiniteDiff
    chain_der_shape_km.append(chain_dhkm_dx)
    chain_dhkm_dx = hk_deriv2/py_dhk_dhkm #FiniteDiff/StaticMethod
    chain_der_shape_km_denom_static.append(chain_dhkm_dx)
    chain_dhkm_dx = hk_deriv2/DrelaGilesLaminarMOD._dshape_k_dshape_km(h_km,True) #FiniteDiff/StaticMethod,SRC
    chain_der_shape_km_denom_static_src.append(chain_dhkm_dx)
    chain_dhkm_dx = hk_deriv_py2/h_k_h_km_deriv2  #StaticMethod/FiniteDiff
    chain_der_shape_km_numer_static.append(chain_dhkm_dx)
    chain_dhkm_dx = hk_deriv_py/h_k_h_km_deriv2 #Paper/FiniteDiff
    chain_der_shape_km_numer_paper.append(chain_dhkm_dx)

    d_shape_km_dx = (shape_km[idx+1] - shape_km[idx-1])/spacing
    d_shape_km_dshape_d = (shape_km[idx+1] - shape_km[idx-1])/(shape_d[idx+1] - shape_d[idx-1]) #Finite Difference
    d_shape_km_dm_e = (shape_km[idx+1] - shape_km[idx-1])/(m_e_visc[idx+1] - m_e_visc[idx-1]) #Finite Difference
    d_m_e_dx = (m_e_visc[idx+1] - m_e_visc[idx-1])/spacing #Finite Difference
    d_shape_d_dx = (1/d_shape_km_dshape_d)*(d_shape_km_dx - d_shape_km_dm_e*d_m_e_dx) #TODO just equals 0, limit of finite difference?
    print(d_shape_d_dx)

    #Sequenced result variations of d_delta_d_dx

    #All finite difference
    d_shape_d_dx = (shape_d[idx+1] - shape_d[idx-1])/spacing #Finite Difference
    ddelta_m_dx = (delta_m[idx+1] - delta_m[idx-1])/spacing #Finite Difference
    d_delta_d_dx = del_m*d_shape_d_dx + h_d*ddelta_m_dx
    chain_der_delta_d.append(d_delta_d_dx)
    d_delta_d_dx = del_m*d_shape_d_dx + h_d*f_p[0]
    chain_der_delta_d_dm_static.append(d_delta_d_dx)

    #dshape_d_dx is analytical, except for d_shape_km_dshape_d
    d_shape_km_dshape_d = DrelaGilesLaminarMOD._dshape_km_dshape_d(m_e_visc[idx])
    d_shape_km_dm_e = DrelaGilesLaminarMOD._dshape_km_dm_e(h_d,m_e_visc[idx])
    d_m_e_dx = DrelaGilesLaminarMOD._dme_dx(fd_due,288.15,287,1.4)
    #Use the FD version of d_shape_km_dx
    d_shape_d_dx = (1/d_shape_km_dshape_d)*(d_shape_km_dx - d_shape_km_dm_e*d_m_e_dx)
    d_delta_d_dx = del_m*d_shape_d_dx + h_d*ddelta_m_dx
    chain_der_delta_h_d_partial_static.append(d_delta_d_dx)

    #ddelta_m_dx is analytical
    d_delta_d_dx = del_m*d_shape_d_dx + h_d*f_p[0]
    chain_der_delta_h_d_delm_static.append(d_delta_d_dx)

    #dshape_d_dx is all analytical
    d_shape_d_dx = (1/d_shape_km_dshape_d)*(f_p2[1] - d_shape_km_dm_e*d_m_e_dx) #Uses the older model
    d_delta_d_dx = del_m*d_shape_d_dx + h_d*ddelta_m_dx
    chain_der_delta_h_d_full_static.append(d_delta_d_dx)


fig, cdcomp = plt.subplots()
cdcomp.plot(s_ref,c_D,label='XFOIL',linestyle='--',color='#D0DF00',marker='o',markersize=4)
cdcomp.plot(s_ref[1:-1],pybl_cD,label='PyBL',color='#D0DF00')
cdcomp.set_title(r'$c_D$ Comparison')
cdcomp.grid(True)
cdcomp.legend()

fig, cfcomp = plt.subplots()
cfcomp.plot(s_ref,c_f,label='XFOIL',linestyle='--',color='#FF6A39',marker='o',markersize=4)
cfcomp.plot(s_ref[1:-1],pybl_cf,label='PyBL',color='#FF6A39')
cfcomp.set_title(r'$c_f$ Comparison')
cfcomp.grid(True)
cfcomp.legend()


fig, h_k_comp = plt.subplots()
h_k_comp.plot(s_ref[1:-1],fd_dh_k_dx2,label='XFOIL Finite Difference 2nd Order',marker='o',color='#CAC7A7',linestyle='--',markersize=4)
h_k_comp.plot(s_ref[1:-1],fd_dh_k_dx,label='XFOIL Finite Difference 1st Order',marker='o',color='#CAC7A7',linestyle=':',markersize=4)
h_k_comp.plot(s_ref[1:-1],pybl_dh_k_dx_paper,label='PyBL, paper',color='#CAC7A7',marker='s',markersize=4)
h_k_comp.plot(s_ref[1:-1],pybl_dh_k_dx_code,label='PyBL, code',color='#CAC7A7',marker='o',markersize=4)
h_k_comp.legend()
h_k_comp.set_title(r'$\frac{\partial H_k}{\partial\xi}$ Finite Difference vs Analytical Comparison')
h_k_comp.grid(True)
h_k_comp.set_ylim([min(fd_dh_k_dx2),max(fd_dh_k_dx2)+5])

fig, d_m_comp = plt.subplots()
d_m_comp.plot(s_ref[1:-1],fd_dd_m_dx2,label='XFOIL Finite Difference 2nd Order',marker='o',color='#ABCAE9',markersize=4,linestyle='--')
d_m_comp.plot(s_ref[1:-1],fd_dd_m_dx,label='XFOIL Finite Difference 1st Order',marker='o',color='#ABCAE9',markersize=4,linestyle=':')
d_m_comp.plot(s_ref[1:-1],pybl_d_m,label='PyBL f_p',color='#ABCAE9')
d_m_comp.legend()
d_m_comp.set_title(r'$\frac{\partial \delta_m}{\partial\xi}$ Finite Difference vs PyBL ode_impl')
d_m_comp.grid(True)
d_m_comp.set_ylim([min(fd_dd_m_dx2),max(fd_dd_m_dx2)])

#TODO f_p difference most likely stems from the early c_f and c_D values being different
#The most egregious deviations of c_f and c_D line up with the most egregious deviations of the derivatives
fig, d_d_comp = plt.subplots()
d_d_comp.plot(s_ref[1:-1],fd_dd_d_dx2,label='XFOIL Finite Difference 2nd Order',marker='o',color='#A4D65E',markersize=4,linestyle='--')
d_d_comp.plot(s_ref[1:-1],fd_dd_d_dx,label='XFOIL Finite Difference 1st Order',marker='o',color='#A4D65E',markersize=4,linestyle=':')
d_d_comp.plot(s_ref[1:-1],pybl_d_d,label='PyBL f_p',color='#A4D65E')
d_d_comp.plot(s_ref[1:-1],chain_der_delta_d,label='Chain Derivative',color='red',linestyle='--')
d_d_comp.plot(s_ref[1:-1],chain_der_delta_d_dm_static,label='Chain Derivative delta_m derivative f_p',color='red',marker='s',markersize=2)
d_d_comp.plot(s_ref[1:-1],chain_der_delta_h_d_partial_static,label='Chain Derivative shape_d derivative without static method shape_km derivative',color='red',linestyle=':',marker='s',markersize=2)
d_d_comp.plot(s_ref[1:-1],chain_der_delta_h_d_delm_static,label='Chain Derivative shape_d derivative and delta_m derivative',color='red',linestyle=':',marker='*',markersize=2) #TODO overlaps with the manual delta_m result
d_d_comp.plot(s_ref[1:-1],chain_der_delta_h_d_full_static,label='Chain Derivative shape_d derivative with static method shape_km derivative',color='red',marker='*',markersize=2)
d_d_comp.legend()
d_d_comp.set_title(r'$\frac{\partial \delta_d}{\partial\xi}$ Finite Difference vs PyBL ode_impl')
d_d_comp.grid(True)
d_d_comp.set_ylim([0.02,.07])

#fig, d_d_comp_err = plt.subplots()
#d_d_comp_err.plot(s_ref[1:-1],abs(np.array(pybl_d_d) - np.array(fd_dd_d_dx2))/abs(np.array(fd_dd_d_dx2)),marker='o',color='#A4D65E',markersize=4)
#d_d_comp_err.set_title(r'$\frac{\partial \delta_d}{\partial\xi}$ Relative Error')

#TODO the last bit is since xfoil forces a larger magnitude dhk_dhkm than what is analytically expected
fig, h_km_comp = plt.subplots()
h_km_comp.plot(s_ref[1:-1],fd_dh_km_dx2,label='XFOIL Finite Difference 2nd Order',marker='o',color='#A4D65E',markersize=4,linestyle='--')
h_km_comp.plot(s_ref[1:-1],fd_dh_km_dx,label='XFOIL Finite Difference 1st Order',marker='o',color='#A4D65E',markersize=4,linestyle=':')
h_km_comp.plot(s_ref[1:-1],pybl_h_km,label='PyBL f_p',color='#A4D65E')
h_km_comp.plot(s_ref[1:-1],chain_der_shape_km,label='Chain Derivative',color='red')
h_km_comp.plot(s_ref[1:-1],chain_der_shape_km_denom_static,label='Chain Derivative, denominator static method',color='red',linestyle=':')
h_km_comp.plot(s_ref[1:-1],chain_der_shape_km_denom_static_src,label='Chain Derivative, denominator static method, src',color='red',linestyle='--')
h_km_comp.plot(s_ref[1:-1],chain_der_shape_km_numer_static,label='Chain Derivative, numerator static method',color='blue',linestyle=':')
h_km_comp.plot(s_ref[1:-1],chain_der_shape_km_numer_paper,label='Chain Derivative, numerator verbatim equation, XFOIL data',color='blue',linestyle='--')
h_km_comp.legend()
h_km_comp.set_title(r'$\frac{\partial H_{km}}{\partial\xi} = \frac{\frac{\partial H_{k}}{\partial\xi}}{\frac{\partial H_{k}}{\partial H_{km}}}$ Finite Difference vs PyBL ode_impl')
h_km_comp.grid(True)
h_km_comp.set_ylim([0,15])

#TODO The large magnitude dhk_dhkm of the original decreases the magnitude of dhk_dx
fig, dhk_dhkm = plt.subplots()
dhk_dhkm.plot(s_ref[1:-1],fd_dh_k_dh_km2,label='XFOIL',color='#B5E3D8',marker='o',linestyle='--',markersize=4)
dhk_dhkm.plot(s_ref[1:-1],pybl_hk_h_km,label='PyBL',color='#B5E3D8',marker='s',markersize=4)
dhk_dhkm.plot(s_ref[1:-1],pybl_hk_h_km2,label='PyBL, SRC',color='red',marker='s',linestyle=':',markersize=4)
dhk_dhkm.set_title(r'$\frac{\partial H_k}{\partial H_{km}}$ Finite Difference vs Analytical Comparison')
dhk_dhkm.grid(True)
dhk_dhkm.legend()


plt.show()
pass
