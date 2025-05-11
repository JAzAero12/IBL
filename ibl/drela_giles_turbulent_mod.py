"""
Implementation of the Drela and Giles IBL models, with the turbulent closure.

This module contains the necessary classes and data for the implementation of
the Drela and Giles IBL method and uses the turbulent closure functions.
"""

from typing import Tuple, cast, Optional, Any
from typing_extensions import override
from typing import Callable
import numpy as np
import numpy.typing as npt

from ibl.ibl_method import IBLMethod
from ibl.ibl_method import TermReason
from ibl.ibl_method import TermEvent
from ibl.initial_condition import ManualCondition
from ibl.initial_condition import FalknerSkanStagCondition
from ibl.typing import InputParam
from scipy.optimize import minimize
import time


class DrelaGilesTurbulentMOD(IBLMethod):
    """
    Models a turbulent bondary layer using the Drela Giles model (1986).

    Solves the system of ODEs from Drela Giles method when provided the edge
    velocity profile and other configuration information. This method employs the turbulent closure functions.

    References to equation numbers come from the paper by M. Drela and M. Giles: (1986)
    Viscous-inviscid analysis of transonic and low Reynolds number airfoils (1986)
    """

    # Requires nu, u_e, du_edx, M_e and dM_edx
    def __init__(self, nu: float = 1.0, U_e: Optional[Any] = None,
                 dU_edx: Optional[Any] = None, d2U_edx2: Optional[Any] = None,
                 T_air: float = 288.15, R_air: float = 287, gamma: float = 1.4,
                 cf_crit: float = 0., ic = None, show_prog = False, old_ctau_ratio = False, c_tau_init = None, src = True) -> None:

        if ic is None:
            ic = ManualCondition(delta_d=np.inf, delta_m=np.inf, delta_k=0)
        super().__init__(nu=nu, u_e=U_e, du_e=dU_edx, ic = ic) 

        if ic is None:
            self._ic.du_e = float(self.du_e(0))

        self.t_air = T_air
        self.R_air = R_air
        self.gamma = gamma
        self.shape_km_bank_lo = np.linspace(.1,4.,400)
        self.shape_km_bank_hi = np.arange(3.,7.401,.001)
        self.src = src
        self.set_separation_event(self.u_e,cf_crit,self.du_e)

        #Flag to switch between the two different shape_km equations
        self.shape_km_hi_flag = False
        self.xvec = np.array([])
        self.ctvec = np.array([])
        self.show_prog = show_prog
        self.old_ctau_ratio = old_ctau_ratio
        self.c_tau_init = c_tau_init

    @property
    def initial_delta_m(self) -> float:
        """
        Momentum thickness at start of integration.
        Must be greater than zero.
        """
        return self._ic.delta_m()

    @initial_delta_m.setter
    def initial_delta_m(self, delta_m0: float) -> None:
        if delta_m0 <= 0:
            raise ValueError(f"Invalid initial momentum thickness: {delta_m0}")
        cast(ManualCondition, self._ic).del_m = delta_m0

    @property
    def initial_shape_d(self) -> float:
        """
        Dispacement thickness at start of integration.
        Must be greater than zero
        """
        return self._ic.shape_d()

    @initial_shape_d.setter
    def initial_shape_d(self, shape_d: float) -> None:
        if shape_d <= 0:
            raise ValueError(f"Invalid displacement shape factor: {shape_d}")
        cast(ManualCondition, self._ic).del_d = shape_d*self.initial_delta_m

    # The separation event is the same
    def set_separation_event(self, u_e:Callable[[InputParam],npt.NDArray], cf_crit: float, du_e:Callable[[InputParam],npt.NDArray]) -> None:
        """
        Set the cf value for flow separation, along with other necessary variables.

        Parameters
        ----------
        cf_crit : float
            Critical skin friction, local skin friction below this value
            flags flow separation
        T_air   : float
            Freestream temperature
        R_air   : float
            Ideal gas constant

        T_air and R_air must be in consistent units

        gamma   : float
            Specific heat ratio
        self.nu : float
            Kinematic viscosity
        u_e     : function
            Edge velocity
        """
        T_air = self.t_air
        R_air = self.R_air
        gamma = self.gamma
        #shape_km_bank_lo = self.shape_km_bank_lo
        #shape_km_bank_hi = self.shape_km_bank_hi
        self._add_kill_event(_DrelaGilesSeparationEvent(cf_crit,u_e,self.nu,T_air,R_air,gamma,self.src
                                                        #shape_km_bank_lo,shape_km_bank_hi,du_e
                                                        ))


    @override
    def v_e(self, x: InputParam) -> npt.NDArray: #shouldn't need to change
        """
        Calculate the transpiration velocity.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired transpiration velocity at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        y_p = self._ode_impl(x, self._solution(x))
        shape_d = self.shape_d(x)
        u_e = self.u_e(x)
        du_e = self.du_e(x)
        delta_m = self.delta_m(x)
        return du_e*shape_d*delta_m + u_e*y_p[1]*delta_m + u_e*shape_d*y_p[0]

    @override
    def delta_d(self, x: InputParam) -> npt.NDArray:
        """
        Calculate the displacement thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement thickness at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self._solution(x)[1]

    @override
    def delta_m(self, x: InputParam) -> npt.NDArray:
        """
        Calculate the momentum thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired momentum thickness at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        return self._solution(x)[0]

    @override
    def delta_k(self, x: InputParam) -> npt.NDArray:
        """
        Calculate the kinetic energy thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy thickness at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")
        delta_m = self.delta_m(x)        
        shape_d = self.shape_d(x)
        u_e = self.u_e(x)
        re_delta_m = u_e*delta_m/self.nu
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)

        shape_km = self._shape_km(shape_d,m_e)
        shape_k  = self._shape_k(shape_km,re_delta_m,self.src,m_e)


        return shape_k*delta_m

    @override
    def shape_d(self, x: InputParam) -> npt.NDArray:
        """
        Calculate the displacement shape factor.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement shape factor at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")
        #u_e = self.u_e(x)
        #du_e = self.du_e(x)
        #delta_k = self._solution(x)[1]
        delta_m = self._solution(x)[0]
        delta_d = self._solution(x)[1]
        #shape_k = delta_k/delta_m
        #re_delta_m = delta_m*u_e/self.nu
        #shape_km = self._shape_k_inverse(shape_k,re_delta_m,self.shape_km_bank_lo,self.shape_km_bank_hi,du_e)
        #m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        #shape_d = self._shape_d(shape_km,m_e)
        return delta_d/delta_m

    @override
    def shape_k(self, x: InputParam) -> npt.NDArray:
        """
        Calculate the kinetic energy shape factor.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy shape factor at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        delta_m = self.delta_m(x)        
        shape_d = self.shape_d(x)
        u_e = self.u_e(x)
        re_delta_m = u_e*delta_m/self.nu
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)

        shape_km = self._shape_km(shape_d,m_e)
        shape_k  = self._shape_k(shape_km,re_delta_m,self.src,m_e)

        return shape_k
    
    
    @override
    def tau_w(self, x: InputParam, rho: float) -> npt.NDArray:
        """
        Calculate the wall shear stress.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired wall shear stress at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        #delta_m = self._solution(x)[0]
        u_e = self.u_e(x)
        #du_e = self.du_e(x)
        u_e[np.abs(u_e) < 1e-6] = 1e-6
        #re_delta_m = u_e*delta_m/self._nu
        #delta_k = self._solution(x)[1]
        #shape_k = delta_k/delta_m
        #shape_km = self._shape_k_inverse(shape_k,re_delta_m,self.shape_km_bank_lo,self.shape_km_bank_hi,du_e)
        delta_m = self.delta_m(x)        
        shape_d = self.shape_d(x)
        #u_e = self.u_e(x)
        re_delta_m = u_e*delta_m/self.nu
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)

        shape_km = self._shape_km(shape_d,m_e)
        #shape_k  = self._shape_k(shape_km,re_delta_m)
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        fc = self._fc(m_e,self.src,self.gamma)
        c_f = self._c_f_dg(shape_km,re_delta_m,fc,self.src) # eq 17
        return 0.5*rho*u_e**2*c_f
        #return self._solution(x)[2] #FOR DEBUGGING

    @override
    def dissipation(self, x: InputParam, rho: float) -> npt.NDArray:
        """
        Calculate the dissipation integral.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired dissipation integral at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")
        
        delta_m = self.delta_m(x)        
        shape_d = self.shape_d(x)
        u_e = self.u_e(x)
        #du_e = self.du_e(x)
        re_delta_m = u_e*delta_m/self._nu
        #delta_k = self._solution(x)[1]
        #shape_k = delta_k/delta_m
        #shape_km = self._shape_k_inverse(shape_k,re_delta_m,self.shape_km_bank_lo,self.shape_km_bank_hi,du_e)     
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        shape_km = self._shape_km(shape_d,m_e)  
        fc = self._fc(m_e,self.src,self.gamma)
        c_f = self._c_f_dg(shape_km,re_delta_m,fc,self.src)
        u_s = self._u_s(shape_km,re_delta_m,m_e,self.src)
        if self.src:
            c_tau = self._solution(x)[2]**2.
        else:
            c_tau = self._solution(x)[2]
        c_D = self._c_D(c_f=c_f,u_s=u_s,c_tau=c_tau,src=self.src) # eq 18

        return .5*c_D*rho*u_e**3
    
    def c_tau(self, x: InputParam) -> npt.NDArray:
        """
        Calculate the shear stress coefficient.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired shear stress coefficient at the specified locations.
        """
        
        if self.src:
            c_tau = self._solution(x)[2]**2.
        else:
            c_tau = self._solution(x)[2]
        
        
        return c_tau
    
    def c_tau_eq(self,x:InputParam) -> npt.NDArray:
        """
        Calculate the equilibrium shear stress coefficient.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired equilibrium shear stress coefficient at the specified locations.
        """

        delta_m = self.delta_m(x)
        shape_d = self.shape_d(x)
        u_e = self.u_e(x)
        re_delta_m = delta_m*u_e/self.nu
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        shape_km = self._shape_km(shape_d,m_e)
        src = self.src
        cteq = self._c_tau_eq(shape_km,re_delta_m,m_e,src)

        return cteq

    @override
    def _ode_setup(self) -> Tuple[npt.NDArray, float, float]:
        """
        Set the solver specific parameters.

        Returns
        -------
        3-Tuple
            IBL initialization array
            Relative tolerance for ODE solver
            Absolute tolerance for ODE solver
        """
        
        shape_d_ic = self._ic.shape_d()
        u_e_ic = self._ic.u_e
        if abs(u_e_ic) < 1e-9: #Remove div by 0 errors
            u_e_ic = 1e-9
        m_e_ic = self._mach(u_e_ic,self.t_air,self.R_air,self.gamma)
        shape_km_ic = (shape_d_ic - .29*m_e_ic**2)/(1.+.113*m_e_ic**2)
        re_delta_m_ic = u_e_ic*self._ic.delta_m()/self.nu
        #shape_k_ic = self._shape_k(shape_km_ic,re_delta_m_ic)
        #delta_k_ic = float(shape_k_ic)*self._ic.delta_m()
        c_tau_eq_init = self._c_tau_eq(shape_km_ic,re_delta_m_ic,m_e_ic,self.src)
        
        #Line 1400 in xblsys.f, coefficient appears to be reliant on an equation

        #Line 1561 in xbl.f
        ctrcon = 1.8
        ctrcex = 3.3
        const  = ctrcon*np.exp(-1.*ctrcex/(shape_km_ic-1.0))

        if self.old_ctau_ratio:
            #Flat plate cases don't like above method, like 0.7 instead
            const = 0.7

        c_tau_init = const**2 *c_tau_eq_init

        #As per described in the original paper
        #c_tau_init = .7**2 *c_tau_eq_init
        #In XFOIL source code, c_tau's square root is ST
        #c_tau_eq is CQT


        if self.c_tau_init is not None:
            c_tau_init = self.c_tau_init #Way to have manual input of c_tau

        if self.src:
            c_tau_init = np.sqrt(c_tau_init)

        return np.array([self._ic.delta_m(),self._ic.delta_d(),float(c_tau_init)]), 1e-8, 1e-11
        #return np.array([self._ic.delta_m(),self._ic.delta_d(),float(c_tau_init)]), 1e-6, 1e-6

    @override
    def _ode_impl(self, x: InputParam,
                  f: npt.NDArray) -> npt.NDArray:
        """
        Right-hand-side of the ODE representing Thwaites method.

        Parameters
        ----------
        x: numpy.ndarray
            Streamwise location of current step.
        f: numpy.ndarray
            Current step's solution vector of momentum thickness and
            displacement shape factor.

        Returns
        -------
        numpy.ndarray
            The right-hand side of the ODE at the given state.
        """

        f_p = np.zeros_like(f)
        u_e = self.u_e(x)
        du_e_dx = self.du_e(x)

        delta_m = f[0]
        #delta_k = f[1]
        delta_d = f[1]

        if self.src:
            c_tau = f[2]**2
        else:
            c_tau = f[2]

        shape_d = delta_d/delta_m

        if isinstance(u_e,(int,float)):
            if abs(u_e) < 1e-9:
                u_e = 1e-9
        re_delta_m = u_e*delta_m/self._nu

        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        shape_km = self._shape_km(shape_d,m_e)

        c_tau_eq = self._c_tau_eq(shape_km,re_delta_m,m_e,self.src)

        delta = self._delta(delta_m,shape_km,m_e)

        dshape_k_dre_m = self._dshape_k_dre_m(shape_km,re_delta_m,self.src,m_e)
        ddelta_m_dx = self._ddelta_m_dx(delta_m, shape_km, re_delta_m, m_e, u_e, du_e_dx, self.src, self.gamma)  # eq 10
        dre_m_dx = self._dre_m_dx(u_e, delta_m, du_e_dx, ddelta_m_dx, self._nu)
        dshape_k_dshape_km = self._dshape_k_dshape_km(shape_km,re_delta_m,self.src,m_e)
        dshape_k_dx = self._dshape_k_dx(delta_m, shape_km, u_e, du_e_dx, m_e, re_delta_m,c_tau,self.src,self.gamma)  # eq 11

        dshape_km_dx = (dshape_k_dx - dshape_k_dre_m*dre_m_dx)/dshape_k_dshape_km
        d_shape_km_dshape_d = self._dshape_km_dshape_d(m_e)
        d_shape_km_dm_e = self._dshape_km_dm_e(shape_d,m_e)
        d_m_e_dx = self._dme_dx(du_e_dx,self.t_air,self.R_air,self.gamma)
        d_shape_d_dx = (1/d_shape_km_dshape_d)*(dshape_km_dx - d_shape_km_dm_e*d_m_e_dx)
        ddelta_d_dx = delta_m*d_shape_d_dx + shape_d*ddelta_m_dx

        if self.src:
            dc_tau_dx = self._dc_tau_dx_src(f[2],delta_d,delta,shape_km,re_delta_m,m_e,u_e,du_e_dx,self.gamma)
        else:
            dc_tau_dx = self._dc_tau_dx(c_tau,c_tau_eq,delta)
        #u_s = self._u_s(shape_km,re_delta_m,m_e)
        #f_c = self._fc(m_e)
        #c_f = self._c_f_dg(shape_km,re_delta_m,f_c)
        #c_D = self._c_D(c_f,u_s,c_tau)
        #shape_den = self._shape_den(shape_km,m_e)

        f_p[0] = ddelta_m_dx
        f_p[1] = ddelta_d_dx
        f_p[2] = dc_tau_dx

        #h_kc = shape_km - 1. - 18./re_delta_m
        #fc = self._fc(m_e,self.src,self.gamma)
        #c_f = self._c_f_dg(shape_km,re_delta_m,fc,self.src)
        #u_q = (.5*c_f - (h_kc/(6.7*1.*shape_km))**2)/(.75*delta_d)
        #u_s = self._u_s(shape_km,re_delta_m,m_e,self.src)
        #
        #temp1 = 2*delta*(u_q - 1./u_e*du_e_dx)
        #temp2 = 5.6/(.75*(1. + u_s)) * (np.sqrt(c_tau_eq) - f[2])
        #f_p[2] = f[2]/delta * (temp1 + temp2)
        #

        self.xvec = np.append(self.xvec,x)
        self.ctvec = np.append(self.ctvec,f[2])
        if self.show_prog:
            print('~~~~~~~~~~~~~~')
            print('x = '+str(x))
            print('f = '+str(f))
            print('f_p = '+str(f_p))
            print('~~~~~~~~~~~~~~')

        return f_p

    @staticmethod
    def _mach(u_e: InputParam,t_air: InputParam,R_air: InputParam,gamma: InputParam) -> InputParam: # the conversion between velocity and mach number
        'Add description here'
        a = np.sqrt(gamma*R_air*t_air)
        return u_e/a
    
    @staticmethod
    def _dme_dx(du_e: InputParam,t_air: InputParam,R_air: InputParam,gamma: InputParam) -> InputParam:
        "Streamwise Derivative of Local Edge Mach Number"
        dme_dx = 1/np.sqrt(gamma*R_air*t_air)*du_e
        return dme_dx

    @staticmethod
    def _shape_km(shape_d:InputParam,m_e:InputParam) -> InputParam:
        "Kinematic Shape Factor: Equation 15"
        temp = shape_d - .29*m_e**2
        return temp/(1.+.113*m_e**2)

    @staticmethod
    def _dshape_km_dshape_d(m_e:InputParam) -> InputParam:
        "Partial Derivative of Kinematic Shape Factor with respect to Displacement Shape Factor"
        return (1.)/(1.+.113*m_e**2)
    
    @staticmethod
    def _dshape_km_dm_e(shape_d:InputParam,m_e:InputParam) -> InputParam:
        "Partial Derivative of Kinematic Shape Factor with respect to Local Edge Mach Number"
        temp = (-.226*shape_d - .58)*m_e
        return temp/(1.+.113*m_e**2)**2

    @staticmethod
    def _fc(m_e: InputParam, src: InputParam, gamma: InputParam) -> InputParam:
        'Term used for finding Skin Friction Coefficient: Equation 21'
        if src:
            return np.sqrt(1. + .5*(gamma - 1.)*m_e**2)
        else:
            return np.sqrt(1.0 + 0.2*m_e**2)
    
    @staticmethod
    def _shape_den(shape_km: InputParam, m_e: InputParam) -> InputParam:  # eq 19
        'Density Shape Factor: Equation 19'
        return (0.064/(shape_km - 0.8) + 0.251)*m_e**2
    
    @staticmethod
    def _dre_m_dx(u_e: InputParam, delta_m: InputParam, du_e_dx: InputParam, ddelta_m_dx: InputParam, nu: InputParam) -> InputParam:
        'Streamwise Derivative of the Momentum Thickness Reynolds Number'
        return (1/nu)*(delta_m*du_e_dx + u_e*ddelta_m_dx)

    @staticmethod
    def _u_s(shape_km:InputParam,re_delta_m:InputParam,m_e:InputParam,src:InputParam) -> InputParam:
        'Normalized Wall Slip Velocity: Equation 27'
        shape_k = DrelaGilesTurbulentMOD._shape_k(shape_km,re_delta_m,src,m_e)
        shape_d = DrelaGilesTurbulentMOD._shape_d(shape_km,m_e)
        if src:
            return .5*shape_k*(1. - (shape_km - 1.)/(shape_d*.75))
        else:
            return (shape_k/2)*(1. - 4./3. * (shape_km-1)/shape_d)

    @staticmethod
    def _c_tau_eq(shape_km:InputParam,re_delta_m:InputParam,m_e:InputParam,src:InputParam) -> InputParam:
        'Equilibrium Shear Stress Coefficient: Equation 30'
        shape_k = DrelaGilesTurbulentMOD._shape_k(shape_km,re_delta_m,src,m_e)
        shape_d = DrelaGilesTurbulentMOD._shape_d(shape_km,m_e)
        u_s     = DrelaGilesTurbulentMOD._u_s(shape_km,re_delta_m,m_e,src)

        h_kc = shape_km - 1. - 18./re_delta_m
        temp = shape_k*(shape_km - 1.)*h_kc**2
        temp2 = 2.*6.7**2*.75*(1.-u_s)*shape_d*shape_km**2
        if src:
            return temp/temp2
        else:
            return (shape_k*.015*(shape_km-1.)**3)/((1.-u_s)*shape_d*shape_km**2)
    
    @staticmethod
    def _dc_tau_dx(c_tau:InputParam,c_tau_eq:InputParam,delta:InputParam) -> InputParam:
        'Streamwise Derivative of Shear Stress Coefficient: Equation 28'
        return 4.2*(c_tau/delta)*(np.sqrt(c_tau_eq) - np.sqrt(c_tau))
    
    @staticmethod
    def _dc_tau_dx_src(ct_sqrt:InputParam, delta_d:InputParam, delta:InputParam, shape_km:InputParam, re_delta_m:InputParam, m_e:InputParam, u_e:InputParam, du_e_dx:InputParam, gamma:InputParam) -> InputParam:
        h_kc = shape_km - 1. - 18./re_delta_m
        src = True #This relation is explicitly the 'new' equation (from the sourcecode)
        fc = DrelaGilesTurbulentMOD._fc(m_e,src,gamma)
        c_f = DrelaGilesTurbulentMOD._c_f_dg(shape_km,re_delta_m,fc,src)
        u_q = (.5*c_f - (h_kc/(6.7*1.*shape_km))**2)/(.75*delta_d)
        u_s = DrelaGilesTurbulentMOD._u_s(shape_km,re_delta_m,m_e,src)
        c_tau_eq = DrelaGilesTurbulentMOD._c_tau_eq(shape_km,re_delta_m,m_e,src)

        temp1 = 2*delta*(u_q - 1./u_e*du_e_dx)
        temp2 = 5.6/(.75*(1. + u_s)) * (np.sqrt(c_tau_eq) - ct_sqrt)
        return ct_sqrt/delta * (temp1 + temp2)

    @staticmethod
    def _delta(delta_m:InputParam,shape_km:InputParam,m_e:InputParam) -> InputParam:
        'Boundary Layer Thickness: Equation 29'
        shape_d = DrelaGilesTurbulentMOD._shape_d(shape_km,m_e)
        delta_d = delta_m*shape_d

        delta = delta_m*(3.15 + 1.72/(shape_km-1.)) + delta_d
        #if not isinstance(delta,np.ndarray):
        #    delta = np.asarray([delta])
        #delta = np.array([12.*delta_m if dl >12.*delta_m else dl for dl in delta])
        #return delta_m*(3.15 + 1.72/(shape_km-1.)) + delta_d
        return delta

    @staticmethod
    def _c_f_dg(shape_km: InputParam, re_delta_m: InputParam, fc:InputParam, src:InputParam) -> npt.NDArray:
        'Skin Friction Coefficient: Equation 20'
        if not isinstance(shape_km,np.ndarray):
            shape_km = np.asarray(shape_km) #needed this line to declare that everthing is treated as array
        shape_km[abs(shape_km) > 1e9] = 1e9
        shape_km[abs(shape_km) < 1e-9] = 1e-9
        re_delta_m = np.asarray(re_delta_m) #Avoids divide by zero errors
        re_delta_m[abs(re_delta_m) < 1e-9] = 1e-9

        if src:
            grt = np.log(re_delta_m/fc)
            if not isinstance(grt,np.ndarray):
                grt = np.array([grt])
            grt[grt < 3.0] = 3.0
            
            gex = -1.74 - .31*shape_km
            
            arg = -1.33*shape_km
            if not isinstance(arg,np.ndarray):
                arg = np.array([arg])
            arg[arg < -20.] = -20.
            
            thk = np.tanh(4. - shape_km/.875)
            
            cffac = 1.0
            cfo = cffac * .3*np.exp(arg) * (grt/2.3026)**gex
            return (cfo + .00011*(thk-1.))/fc

            #acf = -1.33*shape_km
            #if not isinstance(acf,np.ndarray):
            #    acf = np.array([acf])
            #acf = np.array([-20.+3.*np.exp((a+17.)/3.) if a < -17. else a for a in acf])
            #bcf = np.log10(re_delta_m/fc)
            #if not isinstance(bcf,np.ndarray):
            #    bcf = np.array([bcf])
            #bcf = np.array([1.303 if b < 1.303 else b for b in bcf])
            #
            #return .3*np.exp(acf)*bcf**(-1.74-.31*shape_km) + .00011*(np.tanh(4.-shape_km/.875) - 1.)

        else:
            temp1 = .3*np.exp(-1.33*shape_km)/(np.log10(re_delta_m/fc))**(1.74+.31*shape_km)
            temp2 = .00011*(np.tanh(4-shape_km/.875) - 1)
            return (1/fc)*(temp1 + temp2)

    @staticmethod
    def _ddelta_m_dx(delta_m: InputParam, shape_km: InputParam, re_delta_m: InputParam, m_e: InputParam, 
                     u_e: InputParam, du_e_dx: InputParam, src: InputParam, gamma: InputParam) -> InputParam:
        'Streamwise Derivative of the Momentum Thickness: Equation 10'
        fc = DrelaGilesTurbulentMOD._fc(m_e,src,gamma)
        c_f = DrelaGilesTurbulentMOD._c_f_dg(shape_km,re_delta_m,fc,src)  # eq 17
        shape_d = DrelaGilesTurbulentMOD._shape_d(shape_km, m_e)
        
        return c_f/2 - (2+shape_d-m_e**2)*(delta_m/u_e)*du_e_dx
    
    @staticmethod
    def _shape_d(shape_km: InputParam, m_e: InputParam) -> InputParam:
        'Displacement Shape Factor as a Function of Kinematic Shape Factor and Edge Mach Number: Equation 15'
        return shape_km*(1+0.113*m_e**2) + 0.29*m_e**2

    @staticmethod
    def _dshape_k_dre_m(shape_km:InputParam,re_delta_m:InputParam,src:InputParam, m_e:InputParam) -> InputParam:
        'Kinetic Energy Shape Factor Derivative with respect to Momentum Thickness Reynolds Number'

        if not isinstance(shape_km,np.ndarray):
            shape_km = np.asarray([shape_km])
        if not isinstance(re_delta_m,np.ndarray):
            re_delta_m = np.asarray([re_delta_m])

        shape_0 = np.array([4. if Re < 400 else 3. + 400./Re for Re in re_delta_m])
        dh0_drem = np.array([0. if Re < 400 else -400./Re**2 for Re in re_delta_m])

        if src:
            rtz = np.array([Re if Re > 200. else 200. for Re in re_delta_m])
            rtz_rt = np.array([1. if Re > 200. else 0. for Re in re_delta_m])
            def dhk_drem_low_src(shape_km:InputParam,re_delta_m:InputParam,shape_0:InputParam,dh0_drem:InputParam,re_z:InputParam,dre_z:InputParam) -> InputParam:
                hsmin = 1.5
                #dhsinf = .015
                hr = (shape_0 - shape_km)/(shape_0 - 1.)
                hr_rt = (1. - hr)/(shape_0 - 1.)*dh0_drem
                temp1 = (2.-hsmin-4./re_z)*hr*2. * 1.5/(shape_km+.5) * hr_rt
                temp2 = (hr**2 * 1.5/(shape_km+.5) - 1.)*4./re_z**2 * dre_z
                return temp1 + temp2

            def dhk_drem_high_src(shape_km:InputParam,re_delta_m:InputParam,shape_0:InputParam,dh0_drem:InputParam,re_z:InputParam,dre_z:InputParam) -> InputParam:
                hsmin = 1.5
                dhsinf = .015
                grt = np.log(re_z)
                hdif = shape_km - shape_0
                rtmp = shape_km - shape_0 + 4./grt
                htmp = .007*grt/rtmp**2. + dhsinf/shape_km
                htmp_rt = -.014*grt/rtmp**3. * (-1.*dh0_drem - 4./grt**2/re_z * dre_z) + .007/rtmp**2./re_z * dre_z
                temp1 = hdif**2 * htmp_rt - 4./re_z**2. * dre_z
                temp2 = hdif*2.*htmp*(-1.*dh0_drem)
                return temp1 + temp2

            result = np.empty_like(shape_km)
            for i, (Hkm, re, H0, dH0, rez, drez) in enumerate(zip(shape_km, re_delta_m, shape_0, dh0_drem, rtz, rtz_rt)):
                if Hkm <= H0:
                    result[i] = dhk_drem_low_src(Hkm, re, H0, dH0, rez, drez)
                else:
                    result[i] = dhk_drem_high_src(Hkm, re, H0, dH0, rez, drez)
        else:
            def dhk_drem_low(shape_km:InputParam,re_delta_m:InputParam,shape_0:InputParam,dh0_drem:InputParam) -> InputParam:
                temp = -4./re_delta_m**2
                temp2 = .5*1.6/re_delta_m**1.5
                temp3 = ((shape_0-shape_km)**1.6)/shape_km
                temp4 = .165 - 1.6/np.sqrt(re_delta_m)
                temp5 = 1.6*(shape_0-shape_km)**.6/shape_km * dh0_drem
                return temp + temp2*temp3 + temp4*temp5

            def dhk_drem_high(shape_km:InputParam,re_delta_m:InputParam,shape_0:InputParam,dh0_drem:InputParam) -> InputParam:
                temp = -4./re_delta_m**2
                temp2 = -2.*(shape_km-shape_0)*dh0_drem
                temp3 = .04/shape_km + .007*np.log(re_delta_m)/(shape_km - shape_0 + 4./np.log(re_delta_m))**2
                temp4 = (shape_km-shape_0)**2.
                temp5 = (.007/re_delta_m)/(shape_km-shape_0+4./np.log(re_delta_m))**2
                temp6 = -(2*.007*np.log(re_delta_m))*(-dh0_drem - 4./np.log(re_delta_m)*1/re_delta_m)/(shape_km-shape_0+4./np.log(re_delta_m))**3
                return temp + temp2*temp3 + temp4*(temp5+temp6)

            result = np.empty_like(shape_km)
            for i, (Hkm, re, H0, dH0) in enumerate(zip(shape_km, re_delta_m, shape_0, dh0_drem)):
                if Hkm <= H0:
                    result[i] = dhk_drem_low(Hkm, re, H0, dH0)
                else:
                    result[i] = dhk_drem_high(Hkm, re, H0, dH0)
        
        if src:
            fm = 1. + .014*m_e**2.
            result = result/fm

        return result

    @staticmethod
    def _shape_k(shape_km: InputParam,re_delta_m: InputParam, src:InputParam, m_e) -> npt.NDArray:
        'Kinetic Energy Shape Factor: Equation 24'
        if not isinstance(shape_km,np.ndarray):
            shape_km = np.asarray([shape_km])
        if not isinstance(re_delta_m,np.ndarray):
            re_delta_m = np.asarray([re_delta_m])
        shape_0 = np.array([4. if Re < 400. else 3. + 400./Re for Re in re_delta_m])
        if src:
            rtz = np.array([Re if Re > 200. else 200. for Re in re_delta_m])

            def shape_k_low_src(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam, re_z: InputParam) -> InputParam:
                hr = (shape_0 - shape_km)/(shape_0 - 1.)
                hsmin = 1.5
                #dhsinf = .015
                temp1 = (2. - hsmin - 4./re_z)*hr**2 * 1.5/(shape_km+.5)
                temp2 = hsmin + 4./re_z
                return temp1 + temp2

            def shape_k_high_src(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam, re_z:InputParam) -> InputParam:
                hsmin = 1.5
                dhsinf = .015
                grt = np.log(re_z)
                hdif = shape_km - shape_0
                rtmp = shape_km - shape_0 + 4./grt
                htmp = .007*grt/rtmp**2. + dhsinf/shape_km
                return hdif**2. * htmp + hsmin + 4./re_z

            result = np.empty_like(shape_km)
            for i, (Hkm, re, H0, rez) in enumerate(zip(shape_km, re_delta_m, shape_0, rtz)):
                if Hkm <= H0:
                    result[i] = shape_k_low_src(Hkm, re, H0, rez)
                else:
                    result[i] = shape_k_high_src(Hkm, re, H0, rez)
        else:
            def shape_k_low(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam) -> InputParam:
                temp = 1.505 + 4./re_delta_m
                temp2 = (.165 - 1.6/np.sqrt(re_delta_m))*((shape_0-shape_km)**1.6)/shape_km
                return temp + temp2

            def shape_k_high(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam) -> InputParam:
                temp = 1.505 + 4./re_delta_m
                temp2 = ((shape_km-shape_0)**2)*(.04/shape_km + .007*np.log(re_delta_m)/(shape_km - shape_0 + 4/np.log(re_delta_m))**2)
                return temp + temp2

            result = np.empty_like(shape_km)
            for i, (Hkm, re, H0) in enumerate(zip(shape_km, re_delta_m, shape_0)):
                if Hkm <= H0:
                    result[i] = shape_k_low(Hkm, re, H0)
                else:
                    result[i] = shape_k_high(Hkm, re, H0)
        
        if src:
            fm = 1. + .014*m_e**2.
            result = (result + .028*m_e**2)/fm

        return result

    #@staticmethod
    #def _shape_k_inverse_lo(shape_k:InputParam, re_delta_m:InputParam,shape_km_bank_lo:InputParam) -> npt.NDArray:
    #    if not isinstance(shape_k,np.ndarray):
    #        shape_k = np.array([shape_k])
    #    if not isinstance(re_delta_m,np.ndarray):
    #        re_delta_m = np.array([re_delta_m])
    #    
    #    re_delta_m[abs(re_delta_m) < 1e-9] = 1e-9
    #
    #    shape_0 = np.array([4. if Re < 400. else 3. + 400./Re for Re in re_delta_m])
    #    shape_km = np.zeros_like(shape_k)
    #
    #    for i, (h_k, r_dm, h0) in enumerate(zip(shape_k,re_delta_m,shape_0)):
    #        def e24_lo(shape_km,shape_k):
    #            temp  = 1.505 + 4./r_dm
    #            temp2 = .165-1.6/np.sqrt(r_dm)
    #            return (shape_k - (temp + temp2*((h0-shape_km)**1.6/shape_km)))**2
    #        shape_k_bank_lo = DrelaGilesTurbulentMOD._shape_k(shape_km_bank_lo,r_dm*np.ones_like(shape_km_bank_lo),False)
    #        guess_idx = np.argmin(abs(shape_k_bank_lo - h_k))
    #        shape_km_lo_guess = shape_km_bank_lo[guess_idx]
    #        result_lo = minimize(e24_lo, shape_km_lo_guess, args=(h_k,), bounds=[(0.1, 4.)], method='L-BFGS-B')
    #        if result_lo.success:
    #                result_lo = result_lo.x[0]
    #        else:
    #            result_lo = shape_km_lo_guess
    #        if shape_km_lo_guess >= 4.:
    #            pass
    #        #result_lo = shape_km_lo_guess
    #        shape_km[i] = result_lo
    #    return shape_km
    #
    #@staticmethod
    #def _shape_k_inverse_hi(shape_k:InputParam, re_delta_m:InputParam, shape_km_bank_hi:InputParam) -> npt.NDArray:
    #    if not isinstance(shape_k,np.ndarray):
    #        shape_k = np.array([shape_k])
    #    if not isinstance(re_delta_m,np.ndarray):
    #        re_delta_m = np.array([re_delta_m])
    #    
    #    re_delta_m[abs(re_delta_m) < 1e-9] = 1e-9
    #    shape_0 = np.array([4. if Re < 400. else 3. + 400./Re for Re in re_delta_m])
    #    shape_km = np.zeros_like(shape_k)
    #    for i, (h_k, r_dm, h0) in enumerate(zip(shape_k,re_delta_m,shape_0)):
    #        temp = 1.505 + 4./r_dm
    #        temp2 = ((shape_km_bank_hi-h0)**2)*(.04/shape_km_bank_hi + .007*np.log(r_dm)/(shape_km_bank_hi - h0 + 4/np.log(r_dm))**2)
    #        shape_k_bank_hi = temp + temp2
    #        low_idx = np.argmin(abs(shape_k_bank_hi - h_k))
    #        shape_km[i] = shape_km_bank_hi[low_idx]
    #
    #    return shape_km
    #
    #@staticmethod
    #def _shape_k_inverse(shape_k:InputParam, re_delta_m:InputParam, shape_km_bank_lo:InputParam, shape_km_bank_hi:InputParam, du_e_dx:InputParam) -> npt.NDArray: #TODO may not be the best implementation of the two equations
    #
    #    if not isinstance(shape_k,np.ndarray):
    #        shape_k = np.array([shape_k])
    #    if not isinstance(re_delta_m,np.ndarray):
    #        re_delta_m = np.array([re_delta_m])
    #
    #    du_e_dx = np.asarray(du_e_dx)
    #    if du_e_dx.shape ==():
    #        du_e_dx = du_e_dx.reshape(-1)
    #
    #    du_e_dx = np.asarray(du_e_dx)
    #
    #    hiflag = False
    #    shape_km = np.zeros_like(shape_k)
    #    for idx, (h_k,r_dm,due) in enumerate(zip(shape_k,re_delta_m,du_e_dx)):
    #        if not hiflag:
    #            shape_km_temp = DrelaGilesTurbulentMOD._shape_k_inverse_lo(h_k,r_dm,shape_km_bank_lo)
    #            if shape_km_temp >= (4.-1e-3) and due < 0:
    #                shape_km[idx] = DrelaGilesTurbulentMOD._shape_k_inverse_hi(h_k,r_dm,shape_km_bank_hi)
    #                hiflag = True
    #            else:
    #                shape_km[idx] = shape_km_temp
    #        else:
    #            shape_km[idx] = DrelaGilesTurbulentMOD._shape_k_inverse_hi(h_k,r_dm,shape_km_bank_hi)
    #            if shape_km[idx] <= (4.+1e-3) and due > 0:
    #                shape_km[idx] = DrelaGilesTurbulentMOD._shape_k_inverse_lo(h_k,r_dm,shape_km_bank_lo)
    #                hiflag = False
    #    
    #    return shape_km

    @staticmethod
    def _c_D(c_f: InputParam, u_s: InputParam, c_tau: InputParam, src: InputParam) -> npt.NDArray:
        'Dissipation Coefficient: Equation 26'
        u_s = np.asarray(u_s)

        if src:
            return .5*c_f*u_s + c_tau*(1.-u_s)
        else:
            return u_s*c_f/2. + c_tau*(1.-u_s)

    @staticmethod
    def _dshape_k_dshape_km(shape_km: InputParam,re_delta_m: InputParam,src: InputParam,m_e: InputParam) -> npt.NDArray:
        'Kinetic Energy Shape Factor Derivative with respect to Kinematic Shape Factor'
        if not isinstance(shape_km,np.ndarray):
            shape_km = np.asarray([shape_km])
        if not isinstance(re_delta_m,np.ndarray):
            re_delta_m = np.asarray([re_delta_m])
        shape_0 = np.array([4. if Re < 400 else 3. + 400./Re for Re in re_delta_m])
        if src:
            rtz = np.array([Re if Re > 200. else 200. for Re in re_delta_m])

            def dhk_dhkm_low_src(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam,re_z: InputParam) -> InputParam:
                hsmin = 1.5
                #dhsinf = .015
                hr_hk = -1./(shape_0 - 1.)
                hr = (shape_0 - shape_km)/(shape_0 - 1.)
                temp1 = -1.*(2.-hsmin-4./re_z)*hr**2. *1.5/(shape_km+.5)**2.
                temp2 = (2. - hsmin - 4./re_z)*hr*2. *1.5/(shape_km+.5)*hr_hk
                #HS_HK =-(2.0-HSMIN-4.0/RTZ)*HR**2  * 1.5/(HK+0.5)**2
                #+ (2.0-HSMIN-4.0/RTZ)*HR*2.0 * 1.5/(HK+0.5) * HR_HK
                return temp1+temp2

            def dhk_dhkm_high_src(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam,re_z: InputParam) -> InputParam:
                #hsmin = 1.5
                dhsinf = .015
                hdif = shape_km - shape_0
                grt = np.log(re_z)
                rtmp = shape_km - shape_0 + 4./grt
                htmp = .007*grt/rtmp**2 + dhsinf/shape_km
                htmp_hk = -.014*grt/rtmp**3. - dhsinf/shape_km**2
                temp1 = hdif*2.*htmp
                temp2 = hdif**2.*htmp_hk
                return temp1 + temp2

            result = np.empty_like(shape_km)
            for i, (Hkm, re, H0, rez) in enumerate(zip(shape_km, re_delta_m, shape_0, rtz)):
                if Hkm <= H0:
                    result[i] = dhk_dhkm_low_src(Hkm, re, H0, rez)
                else:
                    result[i] = dhk_dhkm_high_src(Hkm, re, H0, rez)
        else:
            def dhk_dhkm_low(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam) -> InputParam:
                temp = (.165-1.6/np.sqrt(re_delta_m))
                temp2 = -1.*(1.6*(shape_0-shape_km)**.6/shape_km + (shape_0-shape_km)**1.6/shape_km**2)
                return temp*temp2

            def dhk_dhkm_high(shape_km: InputParam,re_delta_m: InputParam,shape_0: InputParam) -> InputParam:
                temp = 2*(shape_km-shape_0)
                temp2 = .04/shape_km + .007*np.log(re_delta_m)/(shape_km - shape_0 + 4/np.log(re_delta_m))**2
                temp3 = (shape_km-shape_0)**2
                temp4 = -.04/shape_km**2 - 2.*.007*np.log(re_delta_m)/(shape_km - shape_0 + 4/np.log(re_delta_m))**3
                return temp*temp2 + temp3*temp4

            result = np.empty_like(shape_km)
            for i, (Hkm, re, H0) in enumerate(zip(shape_km, re_delta_m, shape_0)):
                if Hkm <= H0:
                    result[i] = dhk_dhkm_low(Hkm, re, H0)
                else:
                    result[i] = dhk_dhkm_high(Hkm, re, H0)

        if src:
            fm = 1. + .014*m_e**2
            result = result/fm
        return result

    @staticmethod
    def _dshape_k_dx(delta_m: InputParam, shape_km: InputParam, u_e: InputParam, du_e: InputParam, m_e: InputParam, re_delta_m: InputParam, c_tau: InputParam, src:InputParam, gamma:InputParam) -> InputParam:
        # dH*/dxi, eq 11
        # requires delta_m, shape_ke, H**, u_e, du_e, CD, C_f, shape_d
        fc = DrelaGilesTurbulentMOD._fc(m_e,src,gamma)
        c_f = DrelaGilesTurbulentMOD._c_f_dg(shape_km, re_delta_m,fc,src)
        shape_den = DrelaGilesTurbulentMOD._shape_den(shape_km, m_e)
        shape_k = DrelaGilesTurbulentMOD._shape_k(shape_km,re_delta_m,src,m_e)
        shape_d = DrelaGilesTurbulentMOD._shape_d(shape_km, m_e)
        u_e = np.asarray(u_e) #Avoids divide by zero errors
        u_e[abs(u_e) < 1e-9] = 1e-9
        u_s = DrelaGilesTurbulentMOD._u_s(shape_km,re_delta_m,m_e,src)
        c_D = DrelaGilesTurbulentMOD._c_D(c_f=c_f,u_s=u_s,c_tau=c_tau,src=src)
        temp1 = 2*c_D - 0.5*shape_k*c_f
        temp2 = (2*shape_den + shape_k*(1 - shape_d))*delta_m*du_e/u_e
        return (1/delta_m)*(temp1 - temp2)
        #TODO so as a joke...
        #return (1/delta_m)*(temp1 - temp2)/(np.exp(shape_k))
        #return (1/delta_m)*(temp1 - temp2)/(shape_k)


class _DrelaGilesSeparationEvent(TermEvent):
    """
    Detects separation and will terminate integration when it occurs.

    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.

    Attributes
    ----------
        cf_crit: Displacement shape factor value that indicates separation
    """

    def __init__(self, cf_crit: float, u_e:Callable[[InputParam],npt.NDArray], nu: float,T_air:float,R_air:float,gamma:float,src:bool
                 #shape_km_bank_lo:npt.NDArray,shape_km_bank_hi:npt.NDArray, du_e:Callable[[InputParam],npt.NDArray]
                 ) -> None:
        """
        Initialize separation criteria for Head's method.

        Parameters
        ----------
        cf_crit : float
            Critical skin friction coefficient for separatation.
        nu      : float
            Kinematic viscosity.
        u_e     : float
            Edge velocity.
        T_air   : float
            Freestream temperature.
        R_air   : float
            Freestream ideal gas constant.
        gamma   : float
            Freestream pressure coefficient ratio.
        """
        super().__init__()
        self._cf_crit = cf_crit
        self._u_e = u_e
        self._nu = nu
        self._T_air = T_air
        self._R_air = R_air
        self._gamma = gamma
        self._src = src
        #self._shape_km_bank_lo = shape_km_bank_lo
        #self._shape_km_bank_hi = shape_km_bank_hi
        #self._du_e = du_e

    @override
    def _call_impl(self, x: float, f: npt.NDArray) -> float:
        """
        Determine if Drela-Giles method integrator should terminate.

        This will terminate once the skin friction is below 0.

        Parameters
        ----------
        x : float
            Streamwise location of current step.
        f : numpy.ndarray
            Current step's solution vector of momentum thickness and
            displacement shape factor.

        Returns
        -------
        float
            Current value of the difference between the critical displacement
            shape factor and the current displacement shape factor.
        """

        # f[0] is momentum thickness, f[1] is kinetic energy thickness, f[2] is n_tilde
        #delta_k = f[1]
        delta_d = f[1]
        delta_m = f[0]
        u_e = self._u_e(x)
        #d_u_e_dx = self._du_e(x)
        re_delta_m = u_e*delta_m/self._nu
        #shape_k = delta_k/f[0]

        #shape_km = DrelaGilesTurbulentMOD._shape_k_inverse(shape_k,re_delta_m,self._shape_km_bank_lo,self._shape_km_bank_hi,d_u_e_dx)

        m_e = DrelaGilesTurbulentMOD._mach(u_e,self._T_air,self._R_air,self._gamma)
        fc = DrelaGilesTurbulentMOD._fc(m_e,self._src,self._gamma)
        shape_d = delta_d/delta_m
        shape_km = DrelaGilesTurbulentMOD._shape_km(shape_d,m_e)
        current_cf = DrelaGilesTurbulentMOD._c_f_dg(shape_km,re_delta_m,fc,self._src)
        return float(current_cf - self._cf_crit)

    @override
    def event_info(self) -> Tuple[TermReason, str]:
        return TermReason.SEPARATED, ""
