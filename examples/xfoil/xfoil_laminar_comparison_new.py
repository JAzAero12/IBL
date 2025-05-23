# pylint: disable=too-many-statements,too-many-locals

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.drela_giles_laminar_OLD import DrelaGilesLaminar_OLD
from ibl.interaction_law import interaction_law
from ibl.initial_condition import ManualCondition
from ibl.drela_giles_laminar_mod import DrelaGilesLaminarMOD

import os

# NACA 0003 Re = 1000000
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

file_name = "NACA0003_Re1E6_VISC_INV"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

Re_inf = 1000000
chord = 1
rho_inf = 1.2
mu_inf = 1.8e-5 #Pa*s
nu_inf = mu_inf/rho_inf

U_inf = Re_inf*nu_inf

# From Linux XFOIL 6.99
s_ref = np.array([0.00069,0.00274,0.00668,0.01553,0.02793,0.04078,0.05372,
                0.06669,0.07968,0.09269,0.1057,0.11872,0.13175,0.14478,
                0.15781,0.17085,0.18389,0.19693,0.20997,0.22301,0.23605,
                0.2491,0.26214,0.27519,0.28824,0.30128,0.31433,0.32738,
                0.34043,0.35348,0.36653,0.37958,0.39263,0.40568,0.41874,
                0.43179,0.44484,0.45789,0.47095,0.484,0.49706,0.51011,
                0.52316,0.53622,0.54927,0.56233,0.57538,0.58844,0.60149,
                0.61455,0.6276,0.64066,0.65371,0.66677,0.67983,0.69288,
                0.70594,0.71899,0.73205,0.74511,0.75816,0.77122,0.78427,
                0.79733,0.81039,0.82344,0.8365,0.84955,0.86261,0.87567,
                0.88872,0.90178,0.91483,0.92789,0.94094,0.954,0.96706,
                0.98011,0.99297,1.00178])

u_e_U_inf = np.array([0.71505,1.00237,1.03357,1.05584,1.0534,1.05444,
                    1.05368,1.0531,1.0525,1.05189,1.05125,1.0506,1.04992,
                    1.04924,1.04854,1.04782,1.0471,1.04636,1.04561,1.04486,
                    1.04409,1.04332,1.04255,1.04177,1.04098,1.04019,1.0394,
                    1.0386,1.03781,1.03701,1.03621,1.03542,1.03462,1.03382,
                    1.03303,1.03224,1.03144,1.03065,1.02986,1.02908,1.02829,
                    1.02751,1.02672,1.02594,1.02516,1.02438,1.0236,1.02282,
                    1.02204,1.02126,1.02048,1.01969,1.0189,1.01811,1.01731,
                    1.0165,1.01568,1.01485,1.01401,1.01316,1.01229,1.01139,
                    1.01048,1.00954,1.00857,1.00757,1.00652,1.00543,1.00429,
                    1.0031,1.00183,1.0005,0.99908,0.99756,0.99594,0.99421,
                    0.99237,0.99044,0.98847,0.98735,])

u_e_U_inf_inv = np.array([0.92497,0.968,0.97681,0.98242,0.98638,0.98948,0.99205,
                        0.99425,0.9962,0.99794,0.99952,1.00098,1.00233,1.0036,
                        1.0048,1.00594,1.00702,1.00807,1.00907,1.01005,1.01099,
                        1.01192,1.01282,1.0137,1.01457,1.01543,1.01628,1.01712,
                        1.01795,1.01877,1.01959,1.02041,1.02122,1.02203,1.02284,
                        1.02365,1.02446,1.02526,1.02607,1.02688,1.02769,1.0285,
                        1.02932,1.03013,1.03094,1.03176,1.03258,1.03339,1.03421,
                        1.03503,1.03584,1.03666,1.03747,1.03828,1.03909,1.0399,
                        1.0407,1.0415,1.04229,1.04308,1.04385,1.04462,1.04538,
                        1.04613,1.04687,1.04759,1.0483,1.049,1.04968,1.05034,
                        1.05098,1.0516,1.0522,1.05276,1.05327,1.05359,1.05284,
                        1.03794,0.99722,0.72574])

u_e_U_inf_inv = np.flip(u_e_U_inf_inv)

u_e_visc = U_inf*u_e_U_inf
u_e_inv = U_inf*u_e_U_inf_inv

fig, sanity = plt.subplots()
sanity.plot(s_ref,u_e_visc,label='u_e visc')
sanity.plot(s_ref,u_e_inv,label='u_e inv')
plt.legend()
#sanity.plot(s_ref[:-1],np.diff(u_e)/np.diff(s_ref))
#sanity.set_title('')


delta_m = np.array([0.0000091,0.0000263,0.0000458,0.0000761,0.0001046,0.0001282,0.0001482,
                    0.0001658,0.0001819,0.0001967,0.0002105,0.0002236,0.000236,0.0002478,
                    0.0002592,0.0002701,0.0002807,0.000291,0.0003009,0.0003106,0.0003201,
                    0.0003293,0.0003383,0.0003472,0.0003558,0.0003644,0.0003727,0.0003809,
                    0.000389,0.000397,0.0004049,0.0004126,0.0004203,0.0004278,0.0004352,
                    0.0004426,0.0004499,0.0004571,0.0004642,0.0004712,0.0004782,0.0004851,
                    0.000492,0.0004987,0.0005055,0.0005121,0.0005188,0.0005253,0.0005319,
                    0.0005383,0.0005448,0.0005512,0.0005576,0.0005639,0.0005703,0.0005766,
                    0.0005829,0.0005892,0.0005955,0.0006018,0.0006081,0.0006144,0.0006207,
                    0.0006271,0.0006336,0.0006401,0.0006467,0.0006534,0.0006602,0.0006671,
                    0.0006742,0.0006815,0.0006891,0.0006969,0.000705,0.0007134,0.0007223,
                    0.0007314,0.0007406,0.000746])

delta_d = np.array([0.0000202,0.0000747,0.0001055,0.0002055,0.0002629,0.0003307,0.0003814,
                    0.0004269,0.0004685,0.000507,0.000543,0.0005771,0.0006095,0.0006406,
                    0.0006704,0.0006993,0.0007272,0.0007544,0.0007809,0.0008067,0.0008319,
                    0.0008566,0.0008809,0.0009047,0.0009281,0.0009511,0.0009737,0.0009961,
                    0.0010181,0.0010398,0.0010613,0.0010825,0.0011035,0.0011243,0.0011448,
                    0.0011651,0.0011853,0.0012052,0.001225,0.0012446,0.0012641,0.0012834,
                    0.0013026,0.0013216,0.0013406,0.0013595,0.0013782,0.0013969,0.0014155,
                    0.0014341,0.0014527,0.0014713,0.0014898,0.0015085,0.0015271,0.0015459,
                    0.0015648,0.0015838,0.0016031,0.0016226,0.0016425,0.0016627,0.0016833,
                    0.0017045,0.0017264,0.001749,0.0017725,0.0017972,0.0018231,0.0018507,
                    0.0018801,0.0019119,0.0019464,0.0019844,0.0020265,0.0020735,0.0021264,
                    0.0021855,0.0022492,0.0022792])

shape_k = np.array([1.6211,1.5558,1.61,1.5659,1.5834,1.5767,1.5772,
                1.5772,1.577,1.5769,1.5767,1.5765,1.5763,1.5761,
                1.576,1.5758,1.5756,1.5754,1.5752,1.575,1.5748,
                1.5746,1.5744,1.5741,1.5739,1.5737,1.5735,1.5733,
                1.5731,1.5729,1.5727,1.5725,1.5723,1.5721,1.5719,
                1.5717,1.5715,1.5713,1.5711,1.5709,1.5707,1.5705,
                1.5703,1.5702,1.57,1.5698,1.5696,1.5694,1.5692,
                1.5689,1.5687,1.5685,1.5683,1.568,1.5678,1.5675,
                1.5672,1.5669,1.5666,1.5662,1.5659,1.5655,1.565,
                1.5645,1.564,1.5634,1.5627,1.562,1.5612,1.5603,
                1.5593,1.5581,1.5568,1.5554,1.5537,1.5518,1.5497,
                1.5474,1.5451,1.5131,])

c_f = np.array([0.056556,0.0108949,0.0146672,0.0050065,0.0048467,0.0035884,0.0031274,
                0.0027899,0.0025368,0.0023385,0.0021777,0.0020438,0.0019298,0.0018311,
                0.0017445,0.0016676,0.0015987,0.0015364,0.0014797,0.0014278,0.0013801,
                0.0013359,0.0012949,0.0012567,0.001221,0.0011875,0.0011559,0.0011262,
                0.0010982,0.0010716,0.0010464,0.0010225,0.0009997,0.000978,0.0009572,
                0.0009374,0.0009184,0.0009002,0.0008827,0.0008659,0.0008497,0.0008342,
                0.0008191,0.0008046,0.0007905,0.0007768,0.0007636,0.0007507,0.0007381,
                0.0007258,0.0007138,0.000702,0.0006904,0.000679,0.0006677,0.0006565,
                0.0006453,0.0006342,0.000623,0.0006118,0.0006005,0.0005889,0.0005772,
                0.0005651,0.0005527,0.0005398,0.0005263,0.0005122,0.0004973,0.0004814,
                0.0004644,0.0004461,0.0004263,0.0004046,0.0003809,0.000355,0.0003268,
                0.0002966,0.0002658,0.0002542,])

c_D = np.array([0.0116248,0.0062872,0.0045925,0.0024737,0.0018909,0.0015119,0.0013085,
                0.0011676,0.001063,0.0009813,0.0009151,0.0008602,0.0008135,0.0007731,
                0.0007378,0.0007065,0.0006786,0.0006533,0.0006304,0.0006095,0.0005902,
                0.0005725,0.000556,0.0005407,0.0005264,0.000513,0.0005004,0.0004885,
                0.0004773,0.0004667,0.0004567,0.0004472,0.0004381,0.0004294,0.0004212,
                0.0004133,0.0004058,0.0003985,0.0003916,0.0003849,0.0003785,0.0003723,
                0.0003664,0.0003607,0.0003551,0.0003498,0.0003446,0.0003395,0.0003347,
                0.0003299,0.0003253,0.0003208,0.0003165,0.0003122,0.000308,0.0003039,
                0.0002999,0.000296,0.0002921,0.0002883,0.0002845,0.0002808,0.0002771,
                0.0002734,0.0002697,0.000266,0.0002623,0.0002586,0.0002548,0.000251,
                0.0002471,0.0002431,0.000239,0.0002348,0.0002305,0.0002261,0.0002216,
                0.0002171,0.0002127,0.0010086])


n_tilda = np.array([0.,0.,0.,0.,0.,0.,0.,
                    0.,0.,0.,0.,0.,0.,0.0034424,
                    0.0198919,0.0559678,0.1137862,0.1909174,0.2817432,0.3785361,0.475254,
                    0.5701761,0.6634782,0.7553134,0.8458151,0.9351008,1.0232739,1.1104264,
                    1.19664,1.2819877,1.366535,1.450341,1.5334593,1.6159385,1.6978231,
                    1.7791541,1.8599694,1.9403046,2.020193,2.0996665,2.1787555,2.2574898,
                    2.3358986,2.4140111,2.4918567,2.5694654,2.6468683,2.724098,2.8011889,
                    2.8781777,2.955104,3.0320106,3.1089447,3.1859578,3.2631071,3.3404563,
                    3.4180767,3.4960482,3.5744611,3.653418,3.7330357,3.8134482,3.8948095,
                    3.9772984,4.0611226,4.1465259,4.2337956,4.323273,4.415366,4.5105659,
                    4.6094682,4.7127999,4.8214528,4.9365243,5.0593634,5.191613,5.3352232,
                    5.492293,5.6622923,0.0307591])

re_delta_m = np.array([6.486,26.369,47.367,80.335,110.193,135.208,156.135,
                    174.645,191.428,206.897,221.325,234.903,247.771,260.034,
                    271.775,283.059,293.938,304.456,314.649,324.548,334.179,
                    343.565,352.724,361.674,370.43,379.004,387.409,395.656,
                    403.753,411.709,419.532,427.229,434.806,442.269,449.624,
                    456.876,464.03,471.089,478.059,484.942,491.743,498.465,
                    505.112,511.687,518.193,524.633,531.01,537.328,543.589,
                    549.796,555.954,562.064,568.13,574.155,580.144,586.1,
                    592.027,597.929,603.811,609.679,615.537,621.392,627.252,
                    633.123,639.014,644.936,650.9,656.919,663.007,669.183,
                    675.464,681.875,688.439,695.184,702.137,709.32,716.742,
                    724.373,732.014,736.527])

shape_d = np.array([2.2295,2.8408,2.3017,2.7009,2.5133,2.5788,2.5736,
                2.5743,2.5758,2.5775,2.5792,2.581,2.5828,2.5847,
                2.5866,2.5886,2.5907,2.5927,2.5949,2.597,2.5992,
                2.6014,2.6036,2.6058,2.608,2.6102,2.6125,2.6147,
                2.6169,2.6192,2.6214,2.6236,2.6258,2.628,2.6302,
                2.6324,2.6346,2.6368,2.639,2.6411,2.6433,2.6455,
                2.6477,2.6499,2.6522,2.6544,2.6567,2.6591,2.6615,
                2.664,2.6665,2.6692,2.6719,2.6748,2.6779,2.6811,
                2.6846,2.6882,2.6922,2.6965,2.7011,2.7062,2.7118,
                2.7179,2.7248,2.7324,2.741,2.7506,2.7616,2.7741,
                2.7885,2.8052,2.8247,2.8475,2.8744,2.9063,2.9441,
                2.9882,3.0372,3.0554])

T_air = 288.15
R_air = 287.
gamma = 1.4
mach = u_e_visc/np.sqrt(gamma*R_air*T_air)

shape_km = (shape_d - .29*mach**2)/(1. + .113*mach**2)

c_f = np.array([0.056556,0.0108949,0.0146672,0.0050065,0.0048467,0.0035884,0.0031274,
                0.0027899,0.0025368,0.0023385,0.0021777,0.0020438,0.0019298,0.0018311,
                0.0017445,0.0016676,0.0015987,0.0015364,0.0014797,0.0014278,0.0013801,
                0.0013359,0.0012949,0.0012567,0.001221,0.0011875,0.0011559,0.0011262,
                0.0010982,0.0010716,0.0010464,0.0010225,0.0009997,0.000978,0.0009572,
                0.0009374,0.0009184,0.0009002,0.0008827,0.0008659,0.0008497,0.0008342,
                0.0008191,0.0008046,0.0007905,0.0007768,0.0007636,0.0007507,0.0007381,
                0.0007258,0.0007138,0.000702,0.0006904,0.000679,0.0006677,0.0006565,
                0.0006453,0.0006342,0.000623,0.0006118,0.0006005,0.0005889,0.0005772,
                0.0005651,0.0005527,0.0005398,0.0005263,0.0005122,0.0004973,0.0004814,
                0.0004644,0.0004461,0.0004263,0.0004046,0.0003809,0.000355,0.0003268,
                0.0002966,0.0002658,0.0002542])

delta_k = np.array([0.0000147,0.0000409,0.0000738,0.0001191,0.0001656,0.0002022,0.0002337,
                0.0002616,0.0002868,0.0003102,0.0003319,0.0003525,0.000372,0.0003906,
                0.0004085,0.0004257,0.0004423,0.0004584,0.000474,0.0004892,0.000504,
                0.0005185,0.0005326,0.0005465,0.0005601,0.0005734,0.0005865,0.0005994,
                0.000612,0.0006245,0.0006367,0.0006488,0.0006608,0.0006725,0.0006842,
                0.0006957,0.000707,0.0007182,0.0007293,0.0007403,0.0007511,0.0007619,
                0.0007726,0.0007831,0.0007936,0.0008039,0.0008142,0.0008244,0.0008346,
                0.0008446,0.0008546,0.0008646,0.0008744,0.0008843,0.0008941,0.0009038,
                0.0009135,0.0009232,0.0009328,0.0009425,0.0009521,0.0009618,0.0009715,
                0.0009812,0.0009909,0.0010007,0.0010106,0.0010206,0.0010307,0.0010409,
                0.0010513,0.0010619,0.0010728,0.0010839,0.0010953,0.0011071,0.0011193,
                0.0011317,0.0011442,0.0011287])

x = np.array([1.,0.99119,0.97834,0.96529,0.95225,0.9392,0.92615,0.9131,
            0.90005,0.887,0.87395,0.8609,0.84785,0.8348,0.82175,
            0.8087,0.79565,0.7826,0.76954,0.75649,0.74344,0.73039,
            0.71734,0.70429,0.69123,0.67818,0.66513,0.65208,0.63902,
            0.62597,0.61292,0.59987,0.58681,0.57376,0.56071,0.54766,
            0.5346,0.52155,0.5085,0.49545,0.48239,0.46934,0.45629,
            0.44324,0.43019,0.41714,0.40408,0.39103,0.37798,0.36493,
            0.35188,0.33883,0.32578,0.31273,0.29968,0.28664,0.27359,
            0.26054,0.2475,0.23445,0.22141,0.20837,0.19533,0.18229,
            0.16926,0.15622,0.1432,0.13017,0.11715,0.10414,0.09114,
            0.07815,0.06518,0.05224,0.03934,0.02656,0.01427,0.0056,
            0.0019,0.00024,0.00024,0.0019,0.0056,0.01427,0.02656,
            0.03934,0.05224,0.06518,0.07815,0.09114,0.10414,0.11715,
            0.13017,0.1432,0.15622,0.16926,0.18229,0.19533,0.20837,
            0.22141,0.23445,0.2475,0.26054,0.27359,0.28664,0.29968,
            0.31273,0.32578,0.33883,0.35188,0.36493,0.37798,0.39103,
            0.40408,0.41714,0.43019,0.44324,0.45629,0.46934,0.48239,
            0.49545,0.5085,0.52155,0.5346,0.54766,0.56071,0.57376,
            0.58681,0.59987,0.61292,0.62597,0.63902,0.65208,0.66513,
            0.67818,0.69123,0.70429,0.71734,0.73039,0.74344,0.75649,
            0.76954,0.7826,0.79565,0.8087,0.82175,0.8348,0.84785,
            0.8609,0.87395,0.887,0.90005,0.9131,0.92615,0.9392,
            0.95225,0.96529,0.97834,0.99119,1.])

y = np.array([0.00031,0.00062,0.00106,0.00151,0.00194,0.00237,0.00279,
            0.00321,0.00362,0.00402,0.00442,0.00481,0.00519,0.00557,
            0.00595,0.00632,0.00668,0.00703,0.00739,0.00773,0.00807,
            0.0084,0.00873,0.00906,0.00937,0.00968,0.00999,0.01028,
            0.01058,0.01086,0.01114,0.01141,0.01168,0.01193,0.01218,
            0.01242,0.01266,0.01288,0.0131,0.01331,0.0135,0.01369,
            0.01387,0.01404,0.01419,0.01434,0.01447,0.01459,0.01469,
            0.01479,0.01486,0.01492,0.01497,0.015,0.015,0.01499,
            0.01496,0.01491,0.01484,0.01474,0.01461,0.01446,0.01427,
            0.01406,0.01381,0.01352,0.01318,0.0128,0.01237,0.01188,0.01132,
            0.01067,0.00993,0.00905,0.00801,0.00672,0.00504,0.00323,0.00191,
            0.00069,-0.00069,-0.00191,-0.00323,-0.00504,-0.00672,-0.00801,
            -0.00905,-0.00993,-0.01067,-0.01132,-0.01188,-0.01237,-0.0128,
            -0.01318,-0.01352,-0.01381,-0.01406,-0.01427,-0.01446,-0.01461,-0.01474,
            -0.01484,-0.01491,-0.01496,-0.01499,-0.015,-0.015,-0.01497,-0.01492,
            -0.01486,-0.01479,-0.01469,-0.01459,-0.01447,-0.01434,-0.01419,
            -0.01404,-0.01387,-0.01369,-0.0135,-0.01331,-0.0131,-0.01288,
            -0.01266,-0.01242,-0.01218,-0.01193,-0.01168,-0.01141,-0.01114,
            -0.01086,-0.01058,-0.01028,-0.00999,-0.00968,-0.00937,-0.00906,
            -0.00873,-0.0084,-0.00807,-0.00773,-0.00739,-0.00703,-0.00668,
            -0.00632,-0.00595,-0.00557,-0.00519,-0.00481,-0.00442,-0.00402,
            -0.00362,-0.00321,-0.00279,-0.00237,-0.00194,-0.00151,-0.00106,
            -0.00062,-0.00031,])

#Load and create the Drela-Giles model (laminar)

n_crit = 9 #Test out if n_crit properties agree with XFOIL, the program

# For XFOIL, n_crit of .5 leads to transition at x = .2358
# For PyBL, n_crit of ~1.567

ic = ManualCondition(delta_d[0],delta_m[0],delta_k[0])
#ic = None
#u_e = u_e_inv
u_e = u_e_visc

#something fun
#func_corrections,_,_,_,u_e_prepro = interaction_law(s_ref,U_inf,u_e_inv,nu_inf,debug=True)
#u_e = u_e_prepro + func_corrections[len(u_e)] #Preprocessed velocity + corrections
#u_e = U_inf*u_e_U_inf_inv + func_corrections[len(u_e)] #Unprocessed velocity + corrections

fig, sanity = plt.subplots(constrained_layout=True)
#sanity.plot(s_ref,u_e,label='Velocity Profile Fed into PyBL')
#sanity.plot(s_ref,u_e_prepro,label='Preprocessed Velocity')
sanity.plot(s_ref,U_inf*u_e_U_inf_inv,label='Inviscid Velocity',marker='o',markersize=4,color='#154734',linestyle='--')
sanity.plot(s_ref,u_e_visc,label='Viscous Velocity',marker='o',markersize=4,color='#154734')
sanity.set_ylabel(r'$u_e$ [m/s]')
sanity.set_xlabel(r's [m]')
sanity.legend(ncol=2,borderaxespad=-5)


# fig, u_e_prof = plt.subplots(constrained_layout=True)
# u_e_prof.plot(s_ref,u_e_visc,color='#154734',label='Viscous')
# u_e_prof.plot(s_ref,U_inf*u_e_U_inf_inv,color='#154734',label='Inviscid',linestyle='--')
# u_e_prof.set_ylabel(r'$u_e$ [m/s]')
# u_e_prof.set_xlabel(r's [m]')
# u_e_prof.legend(ncol=2,borderaxespad=-5.5)
# fig.tight_layout()
# fig.savefig(file_name+'\\'+'u_e_profs.png')

#DG model
DG_testrun = DrelaGilesLaminarMOD(nu=nu_inf, U_e=[s_ref,u_e],n_tilde_crit=n_crit,ic=ic,
                                  #cf_crit=2.5e-5 #setting cf crit like this helps
                                  )
#What about Thwaites Method?
TM_testrun = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[s_ref,u_e])
TM_testrun.initial_delta_m = delta_m[0]
rtnTM = TM_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
print('Thwaites did:'+rtnTM.message)
print('Thwaites Ends at s = '+str(rtnTM.x_end))
rtn = DG_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
print('DG Model did:'+rtn.message)
print('DG Model Ends at s = '+str(rtn.x_end))

mach = u_e/np.sqrt(gamma*R_air*T_air)
shape_km_PyBL = (DG_testrun.shape_d(s_ref) - .29*mach**2)/(1. + .113*mach**2)

shape_k_from_xi_XF = CubicSpline(s_ref,shape_k)
dshape_k_from_xi_XF = shape_k_from_xi_XF.derivative()

shape_k_from_xi_PyBL = CubicSpline(s_ref,DG_testrun.shape_k(s_ref))
dshape_k_from_xi_PyBL = shape_k_from_xi_PyBL.derivative()


#cf
cf_py = DG_testrun.tau_w(s_ref,rho_inf)


rtn = TM_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
if not rtn.success:
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
dels.plot(s_ref,DG_testrun.delta_m(s_ref),label=r'D-G',color="#BB00FF")
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
dels.plot(s_ref,DG_testrun.shape_d(s_ref),label=r'D-G',color="#612D00")
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
dels.plot(s_ref,cf_dg,label=r'D-G',color="#818181")
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
dels_err.legend(ncol=2,borderaxespad=-5.5)
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'c_D_visc.png')



print('delta d avg error, DG')
print(np.average(abs(delta_d-DG_testrun.delta_d(s_ref))/abs(delta_d)))
print('delta d avg error, TM')
print(np.average(abs(delta_d-TM_testrun.delta_d(s_ref))/abs(delta_d)))

print('delta m avg error, DG')
print(np.average(abs(delta_m-DG_testrun.delta_m(s_ref))/abs(delta_m)))
print('delta m avg error, TM')
print(np.average(abs(delta_m-TM_testrun.delta_m(s_ref))/abs(delta_m)))

print('shape d avg error, DG')
print(np.average(abs(shape_d-DG_testrun.shape_d(s_ref))/abs(shape_d)))
print('shape d avg error, TM')
print(np.average(abs(shape_d-TM_testrun.shape_d(s_ref))/abs(shape_d)))

print('cf avg error, DG')
print(np.average(abs(c_f-cf_dg)/abs(c_f)))
print('cf avg error, TM')
print(np.average(abs(c_f-cf_tw)/abs(c_f)))

print('shape km error, DG')
print(np.average(abs(shape_km - DG_testrun.shape_d(s_ref))/abs(shape_km)))

# print('shape km derivative error, DG')
# print(np.average(abs(dg_hk_der-xf_hk_der)/abs(dg_hk_der)))

# fig, shape_Ks = plt.subplots()
# shape_Ks.plot(s_ref,shape_k,label= 'XFOIL, shape_k')
# shape_Ks.plot(s_ref,DG_testrun.shape_k(s_ref),label= 'PyBL-DG, shape_k')
# shape_Ks.set_ylim([min(shape_k),max(shape_k)])
# shape_Ks.legend()

# fig, shape_k_err = plt.subplots()
# shape_k_err.plot(s_ref,abs(shape_k - DG_testrun.shape_k(s_ref))/abs(shape_k))

fig, n_tild = plt.subplots(constrained_layout=True)
n_tild.plot(s_ref,n_tilda,label='XFOIL',linestyle='--',color='black')
n_tild.plot(s_ref,DG_testrun.n_tilde(s_ref),label='D-G',color='#FF6A39')
n_tild.set_xlabel('s [m]')
n_tild.set_ylabel(r'$\tilde{n}$')
R_crit_loc = 0.0012382134832399619
xfoil_R_crit_loc = 0.13175
n_tild.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'n_tild_visc.png')

# dn_xfoil = np.diff(n_tilda)
# dn_pybl = np.diff(DG_testrun.n_tilde(s_ref))
# xdiff = np.diff(s_ref)

#Inviscid Results
u_e = u_e_inv

#corrects,u_e_proc = interaction_law(s_ref,U_inf,u_e,nu_inf)
#u_e = u_e_proc + corrects[:len(u_e_proc)]

fig, u_e_prof = plt.subplots(constrained_layout=True)
u_e_prof.plot(s_ref,u_e_visc,color='#9FC9CD',label='Viscous')
#u_e_prof.plot(s_ref,U_inf*u_e_U_inf_inv,color='#154734',label='Inviscid',linestyle='--')
u_e_prof.plot(s_ref,u_e,color='#9FC9CD',label='Inviscid',linestyle='--')
u_e_prof.set_ylabel(r'$u_e$ [m/s]')
u_e_prof.set_xlabel(r's [m]')
u_e_prof.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'u_e_profs.png')

#DG model
DG_testrun = DrelaGilesLaminarMOD(nu=nu_inf, U_e=[s_ref,u_e],n_tilde_crit=n_crit,ic=ic,
                                  #cf_crit=2.5e-5 #setting cf crit like this helps
                                  )
#What about Thwaites Method?
TM_testrun = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[s_ref,u_e])
TM_testrun.initial_delta_m = delta_m[0]
rtnTM = TM_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
print('Thwaites did:'+rtnTM.message)
print('Thwaites Ends at s = '+str(rtnTM.x_end))
rtn = DG_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
print('DG Model did:'+rtn.message)
print('DG Model Ends at s = '+str(rtn.x_end))

DG_inv_s = s_ref[s_ref<=rtn.x_end]
TM_inv_s = s_ref[s_ref<=rtnTM.x_end]

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.delta_d(DG_inv_s),label=r'D-G',color='#A4D65E')
dels.plot(TM_inv_s,TM_testrun.delta_d(TM_inv_s),label=r"Thwaites",color='#A4D65E',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.delta_d(DG_inv_s)-delta_d[0:len(DG_inv_s)])/delta_d[0:len(DG_inv_s)],label=r'D-G',color='#A4D65E')
dels_err.plot(TM_inv_s,np.abs(TM_testrun.delta_d(TM_inv_s)-delta_d[0:len(TM_inv_s)])/delta_d[0:len(TM_inv_s)],label=r"Thwaites",color='#A4D65E',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.delta_m(DG_inv_s),label=r'D-G',color='#BB00FF')
dels.plot(TM_inv_s,TM_testrun.delta_m(TM_inv_s),label=r"Thwaites",color='#BB00FF',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.delta_m(DG_inv_s)-delta_m[0:len(DG_inv_s)])/delta_m[0:len(DG_inv_s)],label=r'D-G',color='#BB00FF')
dels_err.plot(TM_inv_s,np.abs(TM_testrun.delta_m(TM_inv_s)-delta_m[0:len(TM_inv_s)])/delta_m[0:len(TM_inv_s)],label=r"Thwaites",color='#BB00FF',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_inv.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.delta_k(DG_inv_s),label=r'D-G',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.delta_k(DG_inv_s)-delta_k[0:len(DG_inv_s)])/delta_k[0:len(DG_inv_s)],label=r'D-G',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_inv.png')


fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,shape_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.shape_d(DG_inv_s),label=r'D-G',color='#612D00')
dels.plot(TM_inv_s,TM_testrun.shape_d(TM_inv_s),label=r"Thwaites",color='#612D00',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.shape_d(DG_inv_s)-shape_d[0:len(DG_inv_s)])/shape_d[0:len(DG_inv_s)],label=r'D-G',color='#612D00')
dels_err.plot(TM_inv_s,np.abs(TM_testrun.shape_d(TM_inv_s)-shape_d[0:len(TM_inv_s)])/shape_d[0:len(TM_inv_s)],label=r"Thwaites",color='#612D00',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$H_d$')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'shape_d_inv.png')


cf_dg = DG_testrun.tau_w(s_ref,rho_inf)/(.5*rho_inf*U_inf**2)
cD_dg = DG_testrun.dissipation(s_ref,rho_inf)/(rho_inf*U_inf**3)
cf_tw = TM_testrun.tau_w(s_ref,rho_inf)/(.5*rho_inf*U_inf**2)

fig, del_errs = plt.subplots(constrained_layout=True)
del_errs.plot(s_ref,c_f,label=r'$c_f$, XFOIL',marker='o',markersize=4,color='#818181',linestyle='--')
del_errs.plot(DG_inv_s,cf_dg[0:len(DG_inv_s)],label=r'$c_f$, PyBL, Drela-Giles',marker='s',markersize=4,color='#818181')
del_errs.plot(s_ref,c_D,label=r'$c_D$, XFOIL',marker='o',markersize=4,color='#5CB8B2',linestyle='--')
del_errs.plot(DG_inv_s,cD_dg[0:len(DG_inv_s)],label=r'$c_D$, PyBL, Drela-Giles',marker='s',markersize=4,color='#5CB8B2')
del_errs.plot(TM_inv_s,cf_tw[0:len(TM_inv_s)],label=r"$c_f$, PyBL, Thwaites",marker='v',markersize=4,color='#818181')
del_errs.legend(ncol=3)
# del_errs.set_title('Boundary Layer CF, Relative Errors')
del_errs.set_xlabel('s [m]')
del_errs.set_ylabel('Coefficients')

fig, del_errs = plt.subplots(constrained_layout=True)
del_errs.plot(DG_inv_s,abs(c_f[0:len(DG_inv_s)]-cf_dg[0:len(DG_inv_s)])/abs(c_f[0:len(DG_inv_s)]),label=r'$c_f$, PyBL, Drela-Giles',marker='s',markersize=4,color='#F8E08E')
del_errs.plot(DG_inv_s,abs(c_D[0:len(DG_inv_s)]-cD_dg[0:len(DG_inv_s)])/abs(c_D[0:len(DG_inv_s)]),label=r'$c_D$, PyBL, Drela-Giles',marker='s',markersize=4,color='#5CB8B2')
del_errs.plot(TM_inv_s,abs(c_f[0:len(TM_inv_s)]-cf_tw[0:len(TM_inv_s)])/abs(c_f[0:len(TM_inv_s)]),label=r"$c_f$, PyBL, Thwaites",marker='v',markersize=4,color='#F8E08E')
del_errs.set_yscale('log')
del_errs.legend(ncol=3,borderaxespad=-7.)
# del_errs.set_title('Boundary Layer CF, Relative Errors')
del_errs.set_xlabel('s [m]')
del_errs.set_ylabel('Relative Difference')

fig, n_tild = plt.subplots(constrained_layout=True)
n_tild.plot(s_ref,n_tilda,label='XFOIL',linestyle='--',color='black')
n_tild.plot(DG_inv_s,DG_testrun.n_tilde(DG_inv_s),label='D-G',color='#FF6A39')
n_tild.set_xlabel('s [m]')
n_tild.set_ylabel(r'$\tilde{n}$')
R_crit_loc = 0.0012382134832399619
xfoil_R_crit_loc = 0.13175
n_tild.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'n_tild_inv.png')


#Inviscid Results with corrections
u_e = u_e_inv

corrects,u_e_proc = interaction_law(s_ref,U_inf,u_e,nu_inf)
u_e = u_e_proc + corrects[:len(u_e_proc)]

fig, u_e_prof = plt.subplots(constrained_layout=True)
u_e_prof.plot(s_ref,u_e_visc,color="#9FC9CD",label='Viscous')
u_e_prof.plot(s_ref,U_inf*u_e_U_inf_inv,color='#9FC9CD',label='Inviscid',linestyle=':')
u_e_prof.plot(s_ref,u_e,color='#9FC9CD',label='Preproc Inv. + Int. Law',linestyle='--')
u_e_prof.set_ylabel(r'$u_e$ [m/s]')
u_e_prof.set_xlabel(r's [m]')
u_e_prof.set_ylim([14,16])
u_e_prof.legend(ncol=3,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'u_e_profs_int.png')

#DG model
DG_testrun = DrelaGilesLaminarMOD(nu=nu_inf, U_e=[s_ref,u_e],n_tilde_crit=n_crit,ic=ic,
                                  #cf_crit=2.5e-5 #setting cf crit like this helps
                                  )
#What about Thwaites Method?
TM_testrun = ThwaitesMethodNonlinear(nu=nu_inf,U_e=[s_ref,u_e])
TM_testrun.initial_delta_m = delta_m[0]
rtnTM = TM_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
print('Thwaites did:'+rtnTM.message)
print('Thwaites Ends at s = '+str(rtnTM.x_end))
rtn = DG_testrun.solve(x0=s_ref[0], x_end=s_ref[-1])
print('DG Model did:'+rtn.message)
print('DG Model Ends at s = '+str(rtn.x_end))

DG_inv_s = s_ref[s_ref<=rtn.x_end]
TM_inv_s = s_ref[s_ref<=rtnTM.x_end]

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_d,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.delta_d(DG_inv_s),label=r'D-G',color='#A4D65E')
dels.plot(TM_inv_s,TM_testrun.delta_d(TM_inv_s),label=r"Thwaites",color='#A4D65E',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.delta_d(DG_inv_s)-delta_d[0:len(DG_inv_s)])/delta_d[0:len(DG_inv_s)],label=r'D-G',color='#A4D65E')
dels_err.plot(TM_inv_s,np.abs(TM_testrun.delta_d(TM_inv_s)-delta_d[0:len(TM_inv_s)])/delta_d[0:len(TM_inv_s)],label=r"Thwaites",color='#A4D65E',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_d$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_d_int.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_m,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.delta_m(DG_inv_s),label=r'D-G',color='#BB00FF')
dels.plot(TM_inv_s,TM_testrun.delta_m(TM_inv_s),label=r"Thwaites",color='#BB00FF',linestyle=':')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.delta_m(DG_inv_s)-delta_m[0:len(DG_inv_s)])/delta_m[0:len(DG_inv_s)],label=r'D-G',color='#BB00FF')
dels_err.plot(TM_inv_s,np.abs(TM_testrun.delta_m(TM_inv_s)-delta_m[0:len(TM_inv_s)])/delta_m[0:len(TM_inv_s)],label=r"Thwaites",color='#BB00FF',linestyle=':')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_m$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_m_int.png')

fig, (dels,dels_err) = plt.subplots(constrained_layout=True,nrows=2, ncols=1, sharex=True)
dels.plot(s_ref,delta_k,label=r'XFOIL',linestyle='--',color='black',linewidth=4)
dels.plot(DG_inv_s,DG_testrun.delta_k(DG_inv_s),label=r'D-G',color='#F2C75C')
dels_err.plot([0],[-.1],linestyle='--',label='XFOIL',color='black')
dels_err.plot(DG_inv_s,np.abs(DG_testrun.delta_k(DG_inv_s)-delta_k[0:len(DG_inv_s)])/delta_k[0:len(DG_inv_s)],label=r'D-G',color='#F2C75C')
dels_err.legend(ncol=3,borderaxespad=-5.5)
dels.set_ylabel(r'$\delta_k$ [m]')
dels_err.set_yscale('log')
dels_err.set_ylabel('Relative Difference')
dels_err.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig(file_name+'\\'+'delta_k_int.png')

fig, n_tild = plt.subplots(constrained_layout=True)
n_tild.plot(s_ref,n_tilda,label='XFOIL',linestyle='--',color='black')
n_tild.plot(DG_inv_s,DG_testrun.n_tilde(DG_inv_s),label='D-G',color='#FF6A39')
n_tild.set_xlabel('s [m]')
n_tild.set_ylabel(r'$\tilde{n}$')
R_crit_loc = 0.0012382134832399619
xfoil_R_crit_loc = 0.13175
n_tild.legend(ncol=2,borderaxespad=-5.5)
fig.tight_layout()
fig.savefig(file_name+'\\'+'n_tild_int.png')

#Seminar nice picture
fig, (pretty_pic, pretty_pic2) = plt.subplots(2, 1, figsize=(8, 12))
fig.subplots_adjust(wspace=0.4, hspace=0.4)
pretty_pic.set_xlim([0,1.4])
pretty_pic.set_ylim([-.1,.1])
#pretty_pic.grid()
deltam_vec = y[:80]+DG_testrun.delta_m(s_ref)
deltam_vec[-1] = y[79]
deltad_vec = y[:80]+DG_testrun.delta_d(s_ref)
deltad_vec[-1] = y[79]
pretty_pic.plot(x[:80],deltam_vec,label='$\\theta$',color='#BB00FF')
pretty_pic.plot(x[:80],-1*(deltam_vec),color='#BB00FF')
pretty_pic.fill_between(x[:80],deltam_vec,y[:80],color='#BB00FF', alpha=.8)
pretty_pic.fill_between(x[:80],-1*deltam_vec,-1*y[:80],color='#BB00FF', alpha=.8)
#pretty_pic.plot([x[0],1.5],[deltam_vec[0],deltam_vec[0]],color='#3A913F')
#pretty_pic.plot([x[0],1.5],[-deltam_vec[0],-deltam_vec[0]],color='#3A913F')
pretty_pic.plot(x[:80],deltad_vec,label='$\\delta^*$',color='#A4D65E')
pretty_pic.plot(x[:80],-1*(deltad_vec),color='#A4D65E')
#pretty_pic.plot([x[0],1.5],[deltad_vec[0],deltad_vec[0]],color='#A4D65E')
#pretty_pic.plot([x[0],1.5],[-deltad_vec[0],-deltad_vec[0]],color='#A4D65E')
pretty_pic.fill_between(x[:80],deltad_vec,deltam_vec,color='#A4D65E', alpha=.8)
pretty_pic.fill_between(x[:80],-1*deltad_vec,-1*deltam_vec,color='#A4D65E', alpha=.8)
pretty_pic.plot(x,y,color='#54585A')
#pretty_pic.plot([x[0],1.5],[0,0],color='#54585A')
pretty_pic.fill(x,y, color='#8E9089', alpha=.8,label='NACA 0003')
pretty_pic.axis('equal')

#pretty_pic2.grid()
deltam_vec2 = y[:80]+DG_testrun.delta_m(s_ref)
deltam_vec2[-1] = y[79]
deltad_vec2 = y[:80]+DG_testrun.delta_d(s_ref)
deltad_vec2[-1] = y[79]
pretty_pic2.plot(x[:80],deltam_vec,label=r'$\delta_m$',color='#BB00FF')
pretty_pic2.plot(x[:80],-1*(deltam_vec),color='#BB00FF')
pretty_pic2.fill_between(x[:80],deltam_vec,y[:80],color='#BB00FF', alpha=.8)
pretty_pic2.fill_between(x[:80],-1*deltam_vec,-1*y[:80],color='#BB00FF', alpha=.8)
pretty_pic2.plot([x[0],1.5],[deltam_vec[0],deltam_vec[0]],color='#BB00FF')
pretty_pic2.plot([x[0],1.5],[-deltam_vec[0],-deltam_vec[0]],color='#BB00FF')
pretty_pic2.plot(x[:80],deltad_vec,label=r'$\delta_d$',color='#A4D65E')
pretty_pic2.plot(x[:80],-1*(deltad_vec),color='#A4D65E')
pretty_pic2.plot([x[0],1.5],[deltad_vec[0],deltad_vec[0]],color='#A4D65E')
pretty_pic2.plot([x[0],1.5],[-deltad_vec[0],-deltad_vec[0]],color='#A4D65E')
pretty_pic2.fill_between(x[:80],deltad_vec,deltam_vec,color='#A4D65E', alpha=.8)
pretty_pic2.fill_between(x[:80],-1*deltad_vec,-1*deltam_vec,color='#A4D65E', alpha=.8)
pretty_pic2.set_xlabel('s [m]')
pretty_pic2.plot(x,y,color='#54585A')
pretty_pic2.plot([x[0],1.5],[0,0],color='#54585A')
pretty_pic2.fill(x,y, color='#8E9089', alpha=1)
tmp = pretty_pic2.legend(loc='best')
tmp.set_draggable(state=True)
pretty_pic2.set_ylim([-0.02,0.02])
pretty_pic2.set_xlim([-0.02,.1])
fig.subplots_adjust(bottom=0.25)

plt.show()
pass