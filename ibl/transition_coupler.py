from typing import Union
from typing import Tuple, cast, Optional, Any
import warnings

import numpy as np
import numpy.typing as npt

from ibl.initial_condition import ManualCondition
from ibl.typing import InputParam
from ibl.ibl_method import TermReason

#TODO Head's Method changed, unittests are ok so far
class transition_coupler:
    def __init__(self, solution_range: npt.NDArray =np.array([]), 
                 laminar_model: Optional[Any] =None, turbulent_class: Optional[Any] =None,
                 nu: float = 1.0, U_e: Optional[Any] =None, 
                 dU_edx: Optional[Any] =None, d2U_edx2: Optional[Any] =None,
                ):
        """laminar_model must include some kind of initial condition
           turbulent_class will just be the class itself"""
        
        #TODO add in relevant checkers to inform user that something is missing

        self.laminar_model = laminar_model
        rtn_lam = self.laminar_model.solve(x0=solution_range[0], x_end=solution_range[1])
        if not rtn_lam.success:
            print("Could not get solution for laminar method: " + rtn_lam.message)
            return
        
        self.transition_point = rtn_lam.x_end

        if rtn_lam._status == TermReason.REACHED_END:
            # There is no need for a turbulent model
            self.turbulent_model = laminar_model # Turbulent model gets overwritten
        elif rtn_lam._status == TermReason.SEPARATED:
            # The flow separated, so the whole thing is gone
            pass
            warnings.warn(f"Flow separation detected at x = {rtn_lam.x_end}")
        else:
            # There is now the need for the turbulent model
            lam_end_del_d = self.laminar_model.delta_d(rtn_lam.x_end)
            lam_end_del_m = self.laminar_model.delta_m(rtn_lam.x_end) 
            lam_end_del_k = self.laminar_model.delta_k(rtn_lam.x_end)
            #TODO, so to get a good IC for the turb model, preprocessing is needed
            #TODO corrections most likely dependent on type, D-G model doesn't need any kind of correction, probably
            turb_ic = ManualCondition(delta_d=lam_end_del_d,delta_m=lam_end_del_m,delta_k=lam_end_del_k)
            self.turbulent_model = turbulent_class(nu = nu, U_e= U_e, dU_edx= dU_edx, d2U_edx2= d2U_edx2, ic= turb_ic)
            self.turbulent_model.initial_delta_m = lam_end_del_m
            self.turbulent_model.initial_shape_d = lam_end_del_d
            self.turbulent_model.initial_shape_k = lam_end_del_k
            rtn_turb = self.turbulent_model.solve(x0=rtn_lam.x_end,x_end=solution_range[1])
            if not rtn_turb.success:
                print("Could not get solution for turbulent method: " + rtn_turb.message)
                return
            if rtn_turb._status == TermReason.SEPARATED:
                pass
                warnings.warn(f"Flow separation detected at x = {rtn_turb.x_end}")
    
    # Below are the combined equations
    def v_e_combined(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.v_e, self.turbulent_model.v_e])
        return piecewise_func(x)

    def delta_m_combined(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.delta_m, self.turbulent_model.delta_m])
        return piecewise_func(x)
    
    def delta_d_combined(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.delta_d, self.turbulent_model.delta_d])
        return piecewise_func(x)
    
    def delta_k_combined(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.delta_k, self.turbulent_model.delta_k])
        return piecewise_func(x)
    
    def shape_d_combined(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.shape_d, self.turbulent_model.shape_d])
        return piecewise_func(x)
    
    def shape_k_combined(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.shape_k, self.turbulent_model.shape_k])
        return piecewise_func(x)
    
    def tau_w_combined(self, x: InputParam, rho: float) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x,[x < transition_point, x >= transition_point],
                                [lambda x: self.laminar_model.tau_w(x,rho), lambda x: self.turbulent_model.tau_w(x,rho)])
        return piecewise_func(x)
    
    def dissipation_combined(self, x: InputParam, rho: float) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x,[x < transition_point, x >= transition_point],
                                [lambda x: self.laminar_model.dissipation(x,rho), lambda x: self.turbulent_model.dissipation(x,rho)])
        return piecewise_func(x)