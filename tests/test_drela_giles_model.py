"""Module to test the Drela-Giles Model functionality."""

# copied from test_head_method.py
# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

import unittest
import numpy as np
import numpy.testing as np_test

from ibl.drela_giles_laminar import DrelaGilesLaminar
from ibl.typing import InputParam


class TestDrelaGilesModel(unittest.TestCase):
    """Class to test the implementation of the Head method"""

    def test_setters(self) -> None:
        """Test setting parameters."""
        def u_e_fun(x: InputParam) -> InputParam:
            return 0.5*(x-0.8)**2 + 3.5*(x-0.8) + 11.5  # accelerating flow
        def du_e_fun(x: InputParam) -> InputParam:
            return 0.5*(x-0.8) + 3.5  # accelerating flow
        #TODO ^ removes errors but is it neccessary? Head Method test doesn't need this (but it does use Manual Condition)
        dgl = DrelaGilesLaminar(U_e=u_e_fun,dU_edx=du_e_fun)

        #dgl.initial_delta_m = 0.5
        #self.assertEqual(dgl.initial_delta_m, 0.5)
        #dgl.initial_shape_d = 2.0
        #self.assertEqual(dgl.initial_shape_d, 2.0)
        #dgl._ic.du_e = 0.5
        #self.assertEqual(dgl._ic.du_e ,0.5)

        #Declaration of Initial Conditions
        dgl._ic.nu = 2
        self.assertEqual(dgl._ic.nu, 2)
        dgl._ic.du_e = 3
        self.assertEqual(dgl._ic.du_e,3)

        #dgl.n_tilde_crit = 
        #TODO add things like cf_crit or n_crit are added, verifying that when input they save that input
        #TODO how?

        with self.assertRaises(ValueError):
            _ = dgl.v_e(1.0)
        with self.assertRaises(ValueError):
            _ = dgl.delta_m(1.0)
        with self.assertRaises(ValueError):
            _ = dgl.shape_d(1.0)
        with self.assertRaises(ValueError):
            _ = dgl.tau_w(1.0, 1.0)

    def test_shape_k_calculation(self) -> None:
        """Test the shape_k (in paper referred to as H*) calculation.
        Specifically equation 16 and is a piecewise function dependent on shape_km, 
        or kinematic shape factor."""

        eps = 1e-9

        # confirm that shape_k is continuous over shape_km = 4
        shape_km_break = 4
        shape_km_low = shape_km_break - eps
        shape_km_high = shape_km_break + eps
        shape_k_low = DrelaGilesLaminar._shape_k(shape_km_low)
        shape_k_high = DrelaGilesLaminar._shape_k(shape_km_high)
        self.assertIsNone(np_test.assert_allclose(shape_k_low,
                                                  shape_k_high))
        #TODO
        #ABOVE plotting something like that would go under example
        #Instead, use the program enguage digitizer (find it online) to find graph locations?
        #Enguage digitizer of plot
        testpts_shape_km = np.array([1.97822,
                            2.2323,
                            3.01633,
                            4.09074,
                            5.0127,
                            5.83303])
        testpts_shape_k = np.array([1.672074,
                           1.621151,
                           1.538957,
                           1.515373,
                           1.522783,
                           1.538357])
        func_outputs = DrelaGilesLaminar._shape_k(testpts_shape_km)
        #TODO do something similar like cf portion
        self.assertIsNone(np_test.assert_allclose(func_outputs,testpts_shape_k,rtol=1e-03,atol=1e-3))


    def test_cf_calculation(self) -> None:
        """Test the skin friction coefficient (cf) calculation.
        Specifically equation 17 and is a piecewise function dependent on shape_km, 
        or kinematic shape factor."""

        eps = 1e-9

        # confirm that shape_k is continuous over shape_km = 7.4
        shape_km_break = 7.4
        shape_km_low = shape_km_break - eps
        shape_km_high = shape_km_break + eps
        re_delta_m = 400 #TODO best way to have this value?
        cf_low = DrelaGilesLaminar._c_f_dg(shape_km_low,re_delta_m)
        cf_high = DrelaGilesLaminar._c_f_dg(shape_km_high,re_delta_m)
        self.assertIsNone(np_test.assert_allclose(cf_low,cf_high))
        #Enguage Digitizer plot
        testpts_shape_km = np.array([2.06855,
                                    2.28905,
                                    2.58878,
                                    3.1207,
                                    4.02881,
                                    5.84209])
        testpts_cf = np.array([0.46135,
                                0.33252,
                                0.22393,
                                0.10245,
                                0.00675,
                                -0.06319])
        testpts_cf = testpts_cf/(re_delta_m/2)
        func_outputs = DrelaGilesLaminar._c_f_dg(testpts_shape_km,re_delta_m)
        #self.assertIsNone(np_test.assert_allclose(func_outputs,testpts_cf,rtol=1e-03,atol=1e-3))
        self.assertIsNone(np_test.assert_allclose(func_outputs,testpts_cf,atol=1e-4))

    def test_cD_calculations(self) -> None:
        """Test the dissipation coefficient (cD) calculation.
        Specifically equation 18 and is a piecewise function dependent on shape_km, 
        or kinematic shape factor."""  

        eps = 1e-9
        # confirm that shape_k is continuous over shape_km = 7.4
        shape_km_break = 4
        shape_km_low = shape_km_break - eps
        shape_km_high = shape_km_break + eps
        re_delta_m = 400 #TODO best way to have this value?
        shape_k_low = DrelaGilesLaminar._shape_k(shape_km_low)
        shape_k_high = DrelaGilesLaminar._shape_k(shape_km_high)
        cD_low = DrelaGilesLaminar._c_D(shape_km_low,shape_k_low,re_delta_m)
        cD_high = DrelaGilesLaminar._c_D(shape_km_high,shape_k_high,re_delta_m)
        self.assertIsNone(np_test.assert_allclose(cD_low,cD_high))
        #Enguage Digitizer 
        testpts_shape_km = np.array([2.09348,
                                    2.41638,
                                    2.81683,
                                    3.46951,
                                    4.38984,
                                    5.209])
        testpts_shape_k = DrelaGilesLaminar._shape_k(testpts_shape_km)
        testpts_cD = np.array([0.277719,
                                0.232872,
                                0.211942,
                                0.206382,
                                0.205859,
                                0.202377])
        testpts_cD = testpts_cD/(2*re_delta_m/testpts_shape_k)
        func_outputs = DrelaGilesLaminar._c_D(testpts_shape_km,testpts_shape_k,re_delta_m) #TODO talk about debugging this thing
        self.assertIsNone(np_test.assert_allclose(func_outputs,testpts_cD,rtol=1e-03,atol=1e-3))


#TODO some 'simple' derivatives like d_re_m_dx can use a finite difference/small perturbance test, just use numbers that work
    #do a sanity check style test where an alternate form of the derivative func is in this code, and compared with
    
    
    def test_dh_k_dh_km_calculations(self) -> None:
        """Test the derivative of equation 16 with respect to shape_km. The function is piecewise"""

        eps = 1e-9
        # confirm that shape_k is continuous over shape_km = 4
        shape_km_break = 4
        shape_km_low = shape_km_break - eps
        shape_km_high = shape_km_break + eps
        shape_k_der_low = DrelaGilesLaminar._dshape_k_dshape_km(shape_km_low)
        shape_k_der_high = DrelaGilesLaminar._dshape_k_dshape_km(shape_km_high)
        #print('Im here')
        #print(shape_k_der_high)
        #print(shape_k_der_low)
        #TODO so the thing is that the relative difference is kinda high because its like small number divided by small number
        self.assertIsNone(np_test.assert_allclose(shape_k_der_low,shape_k_der_high,rtol=3))

        # compare the output results with a simple finite difference scheme
        # finite difference scheme, the 'low' side
        diff = 1e-5
        shape_km_lo = float(2) #TODO thanks to Aero 525 I remembered this little trick
        finite_diff_lo = (DrelaGilesLaminar._shape_k(shape_km_lo+diff) - DrelaGilesLaminar._shape_k(shape_km_lo))/diff

        # actual function, the 'low' side
        shape_k_der_test_lo = DrelaGilesLaminar._dshape_k_dshape_km(shape_km_lo)

        # finite difference scheme, the 'high' side
        shape_km_hi = float(5) 
        finite_diff_hi = (DrelaGilesLaminar._shape_k(shape_km_hi+diff) - DrelaGilesLaminar._shape_k(shape_km_hi))/diff

        # actual function, the 'low' side
        shape_k_der_test_hi = DrelaGilesLaminar._dshape_k_dshape_km(shape_km_hi)


        #TODO For loop of a few (like 10?) range of values
        #Loop through and do finite difference and code's calcs as their own numpy arrays
        #use assert all close (change tolerance)

        self.assertIsNone(np_test.assert_allclose(finite_diff_lo,shape_k_der_test_lo,rtol=1e-05,atol=1e-5)) #TODO what is 'good enough' for the tolerance?
        self.assertIsNone(np_test.assert_allclose(finite_diff_hi,shape_k_der_test_hi,rtol=1e-05,atol=1e-5))

    #TODO do similar 1 = 1 type checkers for the equations used
    
    def test_density_function(self) -> None:
        """Test density shape factor function."""
        # testing values
        m_e = .3
        shape_km = 3

        test_den_shape = (.064/(shape_km -.8) + .251)*m_e**2
        test_den_func = DrelaGilesLaminar._shape_den(shape_km,m_e)
        self.assertIsNone(np_test.assert_allclose(test_den_shape,test_den_func))
        #TODO check other mach numbers
    
    def test_shape_d(self) -> None:
        """Test displacement shape factor function."""
        eps = 1e-9
        # When mach number is effectively 0, dispacement shape factor and kinematic shape factor should be equal
        shape_km = 5
        shape_d = DrelaGilesLaminar._shape_d(shape_km,eps)
        self.assertIsNone(np_test.assert_allclose(shape_km,shape_d))

    def test_mach(self) -> None:
        """Test the velocity to mach conversion function. Under the assumption that mach continues to be a function."""
        #TODO make sure to expand number of test points
        vel = 330
        gamma = 1.4
        R_air = 287
        t_air = 288.15
        a = np.sqrt(gamma*R_air*t_air)
        mach_test = vel/a
        self.assertIsNone(np_test.assert_allclose(DrelaGilesLaminar._mach(vel,t_air,R_air,gamma),mach_test))
    
    def test_l_func(self) -> None:
        """Test the l(shape_km) function"""
        shape_km = 5
        l_res = (6.54*shape_km - 14.07)/shape_km**2
        self.assertIsNone(np_test.assert_allclose(DrelaGilesLaminar._lfunc(shape_km),l_res))
    
    def test_m_func(self) -> None:
        """Tests the m(shape_km) function."""
        shape_km = 5
        l_temp = (6.54*shape_km - 14.07)/shape_km**2
        f_res = (0.058*((shape_km - 4)**2)/(shape_km - 1) - 0.068)*(1/l_temp)
        self.assertIsNone(np_test.assert_allclose(DrelaGilesLaminar._mfunc(shape_km),f_res))

    #TODO maybe add a test for _dshape_k_dre_m, since its 0 and it should probably stay that way
    # any way to do it that also takes into account the lack of input in its original form?
    

#TODO goes under examples as a demo of model working
#    def blasius_test(self) -> None:
#        """Run a Blasius solution with this model to verify functionality"""
#
#        u_inf = 20
#        u_e = u_inf * np.ones([1,20]) # Blasius -> edge velocity profile should be constant
#        T_inf = 288.15
#        gamma = 1.4
#        R_air = 287
#        a = np.sqrt(gamma*R_air*T_inf)
#        m_e = u_e/a
#        c = np.linspace(0,1,20) # Flat plate
#        # Pulled from laminar xfoil test
#        re = 1000
#        rho_inf = 1.2
#        nu_inf = u_inf*c/re
#        # The 'solutions' that can be used for comparison
#        blas_delta_d = 1.7208*np.sqrt(nu_inf*c/u_inf) # 'True' delta d profile from Blasius
#        blas_delta_m = 0.664*np.sqrt(nu_inf*c/u_inf) # 'True' delta m profile from Blasius
#        blas_cf = 0.664/np.sqrt(u_e*c/nu_inf) #TODO double check if this is right

#TODO like thwaites, a blasius/falkner skan based unit test can be implemented

    #TODO keep around as a reference
    #Essentially an elaborate 1=1 case, where at one time in history the method created the array values below
    #very 'last' test when confident that this model is in a mature state

    #def test_sample_calculations(self) -> None:
    #    """Test sample calculations."""
    #    x = np.linspace(0.8, 3.0, 20)
    #    nu = 1e-5
    #    rho = 1.0
#
    #    def u_e_fun(x: InputParam) -> InputParam:
    #        return 0.5*(x-0.8)**2 + 3.5*(x-0.8) + 11.5  # accelerating flow
#
    #    def du_e_fun(x: InputParam) -> InputParam:
    #        return 0.5*(x-0.8) + 3.5  # accelerating flow
#
    #    def d2u_e_fun(x: InputParam) -> InputParam:
    #        _ = x  # avoid unused variable warning
    #        return 0.5  # accelerating flow
#
    #    hm = HeadMethod(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
    #                    d2U_edx2=d2u_e_fun)
    #    hm.initial_delta_m = 0.0014
    #    hm.initial_shape_d = 1.42
    #    rtn = hm.solve(x0=x[0], x_end=x[-1])
    #    self.assertTrue(rtn.success)
#
    #    # # print out reference values
    #    # print("u_e_ref =", hm.u_e(x))
    #    # print("v_e_ref =", hm.v_e(x))
    #    # print("delta_d_ref =", hm.delta_d(x))
    #    # print("delta_m_ref =", hm.delta_m(x))
    #    # print("delta_k_ref =", hm.delta_k(x))
    #    # print("shape_d_ref =", hm.shape_d(x))
    #    # print("shape_k_ref =", hm.shape_k(x))
    #    # print("tau_w_ref =", hm.tau_w(x, rho))
    #    # print("dissipation_ref =", hm.dissipation(x, rho))
#
    #    # reference data
    #    u_e_ref = [11.5,        11.91196676, 12.33734072, 12.77612188,
    #               13.22831025, 13.69390582, 14.17290859, 14.66531856,
    #               15.17113573, 15.69036011, 16.22299169, 16.76903047,
    #               17.32847645, 17.90132964, 18.48759003, 19.08725762,
    #               19.70033241, 20.32681440, 20.96670360, 21.62]
    #    v_e_ref = [0.00728674, 0.01083954, 0.01249412, 0.01339932,
    #               0.01396050, 0.01435180, 0.01465730, 0.01492076,
    #               0.01516601, 0.01540642, 0.01564952, 0.01589951,
    #               0.01615867, 0.01642807, 0.01670814, 0.01699885,
    #               0.01729995, 0.01761106, 0.01793173, 0.01826148]
    #    delta_d_ref = [0.00198800, 0.00201035, 0.00205305, 0.00210297,
    #                   0.00215477, 0.00220609, 0.00225588, 0.00230370,
    #                   0.00234942, 0.00239310, 0.00243483, 0.00247477,
    #                   0.00251307, 0.00254989, 0.00258536, 0.00261962,
    #                   0.00265280, 0.00268499, 0.00271630, 0.00274680]
    #    delta_m_ref = [0.0014,     0.00144945, 0.00150204, 0.00155435,
    #                   0.00160495, 0.00165328, 0.00169917, 0.00174267,
    #                   0.00178392, 0.00182309, 0.00186037, 0.00189593,
    #                   0.00192996, 0.00196260, 0.00199400, 0.00202428,
    #                   0.00205355, 0.00208192, 0.00210948, 0.00213629]
    #    delta_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #    shape_d_ref = [1.42,       1.38698027, 1.36684448, 1.35295507,
    #                   1.34257558, 1.33437679, 1.32763781, 1.32193439,
    #                   1.31700027, 1.31265931, 1.30878965, 1.30530373,
    #                   1.30213659, 1.29923860, 1.29657098, 1.29410276,
    #                   1.29180876, 1.28966821, 1.28766376, 1.28578071]
    #    shape_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #    tau_w_ref = [0.24496919, 0.27160252, 0.29500543, 0.31735812,
    #                 0.33964136, 0.36236513, 0.38582495, 0.41020865,
    #                 0.43564600, 0.46223385, 0.49004963, 0.51915895,
    #                 0.54962008, 0.58148666, 0.61480943, 0.64963728,
    #                 0.68601793, 0.7239985, 0.76362573, 0.80494622]
    #    dissipation_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
    #    self.assertIsNone(np_test.assert_allclose(u_e_ref, hm.u_e(x),
    #                                              atol=1e-7))
    #    self.assertIsNone(np_test.assert_allclose(v_e_ref, hm.v_e(x),
    #                                              atol=1e-7))
    #    self.assertIsNone(np_test.assert_allclose(delta_d_ref, hm.delta_d(x),
    #                                              atol=1e-7))
    #    self.assertIsNone(np_test.assert_allclose(delta_m_ref, hm.delta_m(x),
    #                                              atol=1e-7))
    #    self.assertIsNone(np_test.assert_allclose(delta_k_ref, hm.delta_k(x)))
    #    self.assertIsNone(np_test.assert_allclose(shape_d_ref, hm.shape_d(x)))
    #    self.assertIsNone(np_test.assert_allclose(shape_k_ref, hm.shape_k(x)))
    #    self.assertIsNone(np_test.assert_allclose(tau_w_ref, hm.tau_w(x, rho)))
    #    self.assertIsNone(np_test.assert_allclose(dissipation_ref,hm.dissipation(x, rho)))

#I can just run this file
if __name__ == "__main__":
    _ = unittest.main(verbosity=1)
