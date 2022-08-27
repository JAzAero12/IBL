#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:51:51 2022

@author: ddmarshall
"""
import unittest
import numpy as np
import numpy.testing as npt
from scipy.misc import derivative as fd

from pyBL.head_method import HeadMethod


class TestHeadMethod(unittest.TestCase):
    """Class to test the implementation of the Head method"""
    
    def testH1Calculation(self):
        eps = 1e-9
        
        ## test that H1 is continuous over H_d=1.6
        H_d_break = 1.6
        H_d_low = H_d_break-eps
        H_d_high = H_d_break+eps
        H1_low = HeadMethod._H1(H_d_low)
        H1_high = HeadMethod._H1(H_d_high)
        self.assertIsNone(npt.assert_allclose(H1_low, H1_high))
        
        ## test that H_d is continuous of H1~3.6374
        H1_break = HeadMethod._H1(H_d_break)
        H1_low = H1_break-eps
        H1_high = H1_break+eps
        H_d_low = HeadMethod._H_d(H1_low)
        H_d_high = HeadMethod._H_d(H1_high)
        self.assertIsNone(npt.assert_allclose(H_d_low, H_d_high))
        
        ## test H1 for a range of H_d
        def H1_fun(H_d):
            if H_d <= 1.6:
                return 3.3 + 0.8234/(H_d - 1.1)**1.287
            else:
                return 3.32254659218600974 + 1.5501/(H_d - 0.6778)**3.064
        H_d = np.linspace(1.11, 2.4, 101)
        H1_ref = np.zeros_like(H_d)
        for i, H_di in enumerate(H_d):
            H1_ref[i] = H1_fun(H_di)
        H1 = HeadMethod._H1(H_d)
        self.assertIsNone(npt.assert_allclose(H1, H1_ref))
        
        ## test H_d can be recoverd from H1 function
        H_d_ref = H_d
        H_d = HeadMethod._H_d(H1)
        self.assertIsNone(npt.assert_allclose(H_d, H_d_ref))
        
        ## test for invalid values
        with self.assertRaises(ValueError):
            HeadMethod._H1(1.1)
        with self.assertRaises(ValueError):
            HeadMethod._H_d(3.322)
    
    def testEntrainmentVelocityCalculation(self):
        ## test calculation of term
        def fun(H1):
            return 0.0306/(H1-3)**0.6169
        
        H1 = np.linspace(3.01, 5, 101)
        Eterm_ref = np.zeros_like(H1)
        for i, H1i in enumerate(H1):
            Eterm_ref[i] = fun(H1i)
        Eterm = HeadMethod._S(H1)
        self.assertIsNone(npt.assert_allclose(Eterm, Eterm_ref))
        
        ## test invalid values
        with self.assertRaises(ValueError):
            HeadMethod._S(3)
    
# I don't think this will work because c_f needs momentum Reynolds number with 
# requires edge velocity, and edge velocity 
#    def testSkinFrictionCalculation(self):
#        nu = 1e-5
#        U_e = 10
#        H_d_range = [1.2, 3.4]
#        Re_delta_m_range = [1e2, 1e5]
#        
#        ## calculate range of displacement shape parameter
#        Re_delta_m = np.average(Re_delta_m_range);
#        H_d = np.linspace(H_d_range[0], H_d_range[-1])
#        c_f_ref = c_f_LudwiegTillman(Re_delta_m, H_d)
#        H1 = HeadMethod._H1(H_d)
#        delta_m = Re_delta_m*nu/U_e
#        c_f = HeadMethod._c_f(delta_m, H1)
#        
#        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))
#        
#        ## calculate range of Reynolds number
#        Re_delta_m = np.logspace(np.log10(Re_delta_m_range[0]),
#                                 np.log10(Re_delta_m_range[-1]));
#        H_d = np.average(H_d_range)
#        c_f_ref = fun(Re_delta_m, H_d)
#        c_f = c_f_LudwiegTillman(Re_delta_m, H_d)
#        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))
#        pass


if (__name__ == "__main__"):
    unittest.main(verbosity=1)
