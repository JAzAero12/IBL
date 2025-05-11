from typing import Union
from typing import Tuple, cast, Optional, Any
import warnings

import numpy as np
import numpy.typing as npt

from ibl.initial_condition import ManualCondition
from ibl.drela_giles_turbulent_mod import DrelaGilesTurbulentMOD
from ibl.typing import InputParam
from ibl.ibl_method import TermReason


class transition_coupler:
    def __init__(self, solution_range: npt.NDArray =np.array([]), 
                 laminar_model: Optional[Any] =None, turbulent_class: Optional[Any] =None,
                 nu: float = 1.0, U_e: Optional[Any] =None, 
                 dU_edx: Optional[Any] =None, d2U_edx2: Optional[Any] =None,
                transition_loc: Optional[Any] =None, turb_ic_preprocessor: Optional[Any] =None,cf_crit:float = 0,sep_tran:bool = True):
        """The laminar_model must be initialized, tubulent model input is the turbulent model class itself"""
        

        if transition_loc is None:
            transition_loc = solution_range[-1]

        self.laminar_model = laminar_model
        U_e = self.laminar_model.u_e
        dU_edx = self.laminar_model.du_e
        d2U_edx2 = self.laminar_model.d2u_e
        rtn_lam = self.laminar_model.solve(x0=solution_range[0], x_end=transition_loc)
        if not rtn_lam.success:
            print("Could not get solution for laminar method: " + rtn_lam.message)
            return
        
        self.transition_point = rtn_lam.x_end

        self.turb_x_end = solution_range[-1]
        already_done = False

        if rtn_lam._status == TermReason.REACHED_END and transition_loc == solution_range[-1]:
            # There is no need for a turbulent model
            self.turbulent_model = laminar_model # Turbulent model gets overwritten
            already_done = True
            self.turb_x_end = rtn_lam.x_end
        else: #Either the flow has transitioned, or the forced transition location is hit
            if rtn_lam._status == TermReason.SEPARATED:
                if not sep_tran:
                    print("FLOW SEPARATION DETECTED AT X = " +str(rtn_lam.x_end))
                    print("RESULTS ARE ONLY LAMINAR")
                    self.turbulent_model = laminar_model
                    already_done = True
                    self.turb_x_end = rtn_lam.x_end
                else:
                    print("FLOW SEPARATION DETECTED AT X = " +str(rtn_lam.x_end)+ " SWITCHING TO TURBULENT MODEL")
            elif rtn_lam._status == TermReason.REACHED_END:
                print('FORCED TRANSITION LOCATION REACHED')
            else:
                print('BOUNDARY LAYER TRANSITIONED TO TURUBLENT AT X = '+str(rtn_lam.x_end))
            # There is now the need for the turbulent model

            lam_end_del_d = self.laminar_model.delta_d(rtn_lam.x_end)
            lam_end_del_m = self.laminar_model.delta_m(rtn_lam.x_end) 
            lam_end_del_k = self.laminar_model.delta_k(rtn_lam.x_end)


                
            if turb_ic_preprocessor is not None:
                #Provides the user the option to input a function that modifies the laminar model outputs for the tubulent ICs
                #Feed in everything the transition coupler function has
                if d2U_edx2 is None:
                    (lam_end_del_d,lam_end_del_m,lam_end_del_k) = turb_ic_preprocessor(lam_end_del_d,lam_end_del_m,lam_end_del_k,
                                                                                   rtn_lam.x_end,nu,U_e(rtn_lam.x_end),dU_edx(rtn_lam.x_end))
                else:
                    (lam_end_del_d,lam_end_del_m,lam_end_del_k) = turb_ic_preprocessor(lam_end_del_d,lam_end_del_m,lam_end_del_k,
                                                                                       rtn_lam.x_end,nu,U_e(rtn_lam.x_end),
                                                                                       dU_edx(rtn_lam.x_end),d2U_edx2(rtn_lam.x_end))

            #By default no corrections are used
            if not already_done:
                turb_ic = ManualCondition(delta_d=lam_end_del_d,delta_m=lam_end_del_m,delta_k=lam_end_del_k)
                self.turbulent_model = turbulent_class(nu = nu, U_e= U_e, dU_edx= dU_edx, d2U_edx2= d2U_edx2, ic= turb_ic)

                #For cases where cf_crit is required for the turbulent Drela-Giles model
                if turbulent_class == DrelaGilesTurbulentMOD:
                    self.turbulent_model = turbulent_class(nu = nu, U_e= U_e, dU_edx= dU_edx, d2U_edx2= d2U_edx2, ic= turb_ic, cf_crit = cf_crit)

                self.turbulent_model.initial_delta_m = lam_end_del_m
                self.turbulent_model.initial_shape_d = lam_end_del_d/lam_end_del_m
                self.turbulent_model.initial_shape_k = lam_end_del_k/lam_end_del_m


                rtn_turb = self.turbulent_model.solve(x0=rtn_lam.x_end,x_end=solution_range[-1])

                if not rtn_turb.success:
                    print("COULD NOT GET SOLUTION FOR TURBULENT MODEL: " + rtn_turb.message)
                    return
                if rtn_turb._status == TermReason.SEPARATED:
                    print("FLOW SEPARATION DETECTED AT X = " +str(rtn_turb.x_end))
                    self.turb_x_end = rtn_turb.x_end
                if rtn_turb.success and rtn_turb._status != TermReason.SEPARATED:
                    print("SUCCESSFULLY COMPLETED ENTIRE DOMAIN. YIPPEE.")
                    self.turb_x_end = rtn_turb.x_end
    
    # Below are the combined equations
    def v_e(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.v_e, self.turbulent_model.v_e])
        return piecewise_func(x)

    def delta_m(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.delta_m, self.turbulent_model.delta_m])
        return piecewise_func(x)
    
    def delta_d(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.delta_d, self.turbulent_model.delta_d])
        return piecewise_func(x)
    
    def delta_k(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.delta_k, self.turbulent_model.delta_k])
        return piecewise_func(x)
    
    def shape_d(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.shape_d, self.turbulent_model.shape_d])
        return piecewise_func(x)
    
    def shape_k(self, x:InputParam) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x, [x < transition_point, x >= transition_point], 
                               [self.laminar_model.shape_k, self.turbulent_model.shape_k])
        return piecewise_func(x)
    
    def tau_w(self, x: InputParam, rho: float) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x,[x < transition_point, x >= transition_point],
                                [lambda x: self.laminar_model.tau_w(x,rho), lambda x: self.turbulent_model.tau_w(x,rho)])
        return piecewise_func(x)
    
    def dissipation(self, x: InputParam, rho: float) -> npt.NDArray:
        transition_point = self.transition_point
        def piecewise_func(x):
            return np.piecewise(x,[x < transition_point, x >= transition_point],
                                [lambda x: self.laminar_model.dissipation(x,rho), lambda x: self.turbulent_model.dissipation(x,rho)])
        return piecewise_func(x)