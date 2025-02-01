"""
Implementation of the Drela and Giles IBL models, with the laminar closure.

This module contains the necessary classes and data for the implementation of
the Drela and Giles IBL method and uses the laminar closure functions.
"""

from typing import Tuple, cast, Optional, Any
from typing_extensions import override
from typing import Callable #TODO Ok if I do this?
import numpy as np
import numpy.typing as npt

from ibl.ibl_method import IBLMethod
from ibl.ibl_method import TermReason
from ibl.ibl_method import TermEvent
#from ibl.initial_condition import ManualCondition
from ibl.initial_condition import FalknerSkanStagCondition
from ibl.typing import InputParam


class DrelaGilesLaminar(IBLMethod):
    """
    Models a laminar bondary layer using the Drela Giles model (1986).

    Solves the system of ODEs from Drela Giles method when provided the edge
    velocity profile and other configuration information. This method employs the laminar closure functions.
    """

    # Requires nu, u_e, du_edx, M_e and dM_edx
    def __init__(self, nu: float = 1.0, U_e: Optional[Any] = None,
                 dU_edx: Optional[Any] = None, #M_e: Optional[Any] = None,
                 #dM_edx: Optional[Any] = None,
                 #T_air: Optional[Any] = 288.15, #TODO, get second opinion
                 T_air: float = 288.15, R_air: float = 287, gamma: float = 1.4,
                 n_tilde_crit: float = 9, cf_crit: float = 0, ic = None) -> None:
                 #n_tilde_init: float = 0) -> None:
        if ic is None:
            ic = FalknerSkanStagCondition(du_e=1,nu=nu)
        super().__init__(nu=nu, u_e=U_e, du_e=dU_edx,
                         #ic=ManualCondition(delta_d=np.inf, delta_m=np.inf,
                          #                  delta_k=0))
                         ic = ic) #TODO is this the right way to do it?
    # For now anything related to 'kinematic' can just have the moniker 'km' -> shape_km
    #TODO get rid of setters? force the user to use FSStagCondition

        self._ic.nu = nu
        self._ic.du_e = float(self.du_e(0))

        self.set_n_tilde_critical(n_tilde_crit)
        self.set_separation_event(self.u_e,cf_crit)
        self.n_tilde_init = 0
        self.t_air = T_air
        self.R_air = R_air
        self.gamma = gamma
    #TODO remove getters/setters for delta_d delta_m
    #@property
    #def initial_delta_m(self) -> float:
    #    """
    #    Momentum thickness at start of integration.
    #    Must be greater than zero.
    #    """
    #    return self._ic.delta_m()

    #@initial_delta_m.setter
    #def initial_delta_m(self, delta_m0: float) -> None:
    #    if delta_m0 <= 0:
    #        raise ValueError(f"Invalid initial momentum thickness: {delta_m0}")
    #    #cast(ManualCondition, self._ic).del_m = delta_m0


    


    #@property
    #def initial_shape_d(self) -> float:
    #    """
    #    Dispacement thickness at start of integration.
    #    Must be greater than zero
    #    """
    #    return self._ic.shape_d()

    #@initial_shape_d.setter
    #def initial_shape_d(self, shape_d: float) -> None:
    #    if shape_d <= 0:
    #        raise ValueError(f"Invalid displacement shape factor: {shape_d}")
    #    cast(FalknerSkanStagCondition, self._ic)._shape_d = shape_d
    #    #cast(ManualCondition, self._ic).shape_d = shape_d

    def set_n_tilde_critical(self, n_tilde_crit: float) -> None:
        """
        Set the n tilde value for transition.

        n tilde is the logarithm of the maximum amplification ratio. [Drela, Giles, 1986]
        When exceeding a certain value (default 9), the boundary layer is assumed to be transitioning to turbulent flow.

        Parameters
        ----------
        n_tilde_crit : float
            New value for the amplification ratio to be used to indicate
            that the boundary layer has transitioned.
        """
        self._set_kill_event(_DrelaGilesTransitionEvent(n_tilde_crit))

    def set_separation_event(self, u_e:Callable[[InputParam],npt.NDArray], cf_crit: float) -> None:
        """
        Set the cf value for flow separation.

        ADD TEXT HERE

        Parameters
        ----------
        cf_crit : float
            ADD DESCRIPTION HERE
        """
        self._set_kill_event(_DrelaGilesSeparationEvent(cf_crit,u_e,self.nu))


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
        return self.delta_m(x)*self.shape_d(x)

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
        
        return self._solution(x)[0] #TODO pretty sure its wrong, unless...

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
        shape_km = self._solution(x)[1]
        u_e = self.u_e(0)
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        shape_d = self._shape_d(shape_km,m_e)
        return np.array(shape_d) # eq 15

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
        shape_km = self._solution(x)[1]
        return self._shape_k(shape_km)
    
    
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

        delta_m = self._solution(x)[0]
        u_e = self.u_e(x)
        u_e[np.abs(u_e) < 0.001] = 0.001
        re_delta_m = u_e*delta_m/self._nu
        c_f = self._c_f_dg(self._solution(x)[1],re_delta_m) # eq 17
        return 0.5*rho*u_e**2*c_f

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
        
        delta_m = self._solution(x)[0]
        shape_km = self._solution(x)[1]
        u_e = self.u_e(x)
        shape_k = self._shape_k(shape_km)
        re_delta_m = u_e*delta_m/self._nu

        c_D = self._c_D(shape_km,shape_k,re_delta_m) # eq 18

        return .5*c_D*rho*u_e**3

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

        self._ic.du_e = float(self.du_e(0))
        self._ic.nu = self.nu
        shape_d_ic = self._ic.shape_d()
        u_e_ic = self.u_e(0)
        m_e_ic = self._mach(u_e_ic,self.t_air,self.R_air,self.gamma)
        shape_km_ic = (shape_d_ic - .29*m_e_ic**2)/(1+.113*m_e_ic**2)
        n_tilde_init = self.n_tilde_init
        return np.array([self._ic.delta_m(),shape_km_ic,n_tilde_init]), 1e-8, 1e-11

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
        shape_km = f[1]
        #n_tilde = f[2]


        #TODO recheck everything -> probably good, good enough to check
        re_delta_m = u_e*delta_m/self._nu
        m_e = self._mach(u_e,self.t_air,self.R_air,self.gamma)
        dshape_k_dx = self._dshape_k_dx(delta_m, shape_km, u_e, du_e_dx, m_e, re_delta_m)  # eq 11
        dshape_k_dre_m = self._dshape_k_dre_m()  # should be 0 for laminar
        ddelta_m_dx = self._ddelta_m_dx(delta_m, shape_km, re_delta_m, m_e, u_e, du_e_dx)  # eq 10
        dre_m_dx = self._dre_m_dx(u_e, delta_m, du_e_dx, ddelta_m_dx, self._nu)
        dshape_k_dshape_km = self._dshape_k_dshape_km(shape_km)
        d_ntild_dre_m = self._d_ntild_dre_m(shape_km)  # eq 35
        m_Hk = self._mfunc(shape_km) # eq 40
        l_Hk = self._lfunc(shape_km) # eq 39

        f_p[0] = ddelta_m_dx
        f_p[1] = (dshape_k_dx - dshape_k_dre_m*dre_m_dx)/dshape_k_dshape_km  # d_Hk_xi
        f_p[2] = d_ntild_dre_m*((m_Hk + 1)/2) * l_Hk * (1/f[0])  # d_ntildae_xi
        
        return f_p

    @staticmethod
    def _mach(u_e: InputParam,t_air: InputParam,R_air: InputParam,gamma: InputParam) -> InputParam: # the conversion between velocity and mach number
        'Add description here'
        a = np.sqrt(gamma*R_air*t_air)
        return u_e/a
    
    @staticmethod
    def _shape_den(shape_km: InputParam, m_e: InputParam) -> InputParam:  # eq 19
        'Add description here'
        return (0.064/(shape_km - 0.8) + 0.251)*m_e**2
    
    @staticmethod
    def _dre_m_dx(u_e: InputParam, delta_m: InputParam, du_e_dx: InputParam, ddelta_m_dx: InputParam, nu: InputParam) -> InputParam:
        'Add description here'
        return (1/nu)*(delta_m*du_e_dx + u_e*ddelta_m_dx)

    @staticmethod
    def _d_ntild_dre_m(shape_km: InputParam) -> InputParam:
        'Add description here'
        return 0.01 * np.sqrt((2.4*shape_km - 3.7 + 2.5*np.arctan(1.5*shape_km - 4.65))**2 + 0.25)
    
    @staticmethod
    def _lfunc(shape_km: InputParam) -> InputParam:
        'Add description here'
        return (6.54*shape_km - 14.07)/shape_km**2

    @staticmethod
    def _mfunc(shape_km: InputParam) -> InputParam:
        'Add description here'
        lfunc = DrelaGilesLaminar._lfunc(shape_km)
        return (0.058*((shape_km - 4)**2)/(shape_km - 1) - 0.068)*(1/lfunc)
    
    @staticmethod
    def _c_f_dg(shape_km: InputParam, re_delta_m: InputParam) -> npt.NDArray:
        'Add description here'
        shape_km = np.asarray(shape_km) #needed this line to declare that everthing is treated as array
        #TODO add a similar (to heads method) checking scheme for 'reasonable' values?
        def lam_fric_low(shape_km: InputParam) -> InputParam:
            temp =  -0.067 + 0.01977*(7.4-shape_km)**2/(shape_km - 1)
            return (2/re_delta_m)*temp

        def lam_fric_high(shape_km: InputParam) -> InputParam:
            temp = -0.067 + 0.022*(1 - 1.4/(shape_km-6))**2
            return (2/re_delta_m)*temp

        return np.piecewise(shape_km, [shape_km <= 7.4, shape_km > 7.4], [lam_fric_low, lam_fric_high]) #not sure what's wrong

    @staticmethod
    def _ddelta_m_dx(delta_m: InputParam, shape_km: InputParam, re_delta_m: InputParam, m_e: InputParam, u_e: InputParam, du_e_dx: InputParam) -> InputParam:
        'Add description here'
        c_f = DrelaGilesLaminar._c_f_dg(shape_km,re_delta_m)  # eq 17
        shape_d = DrelaGilesLaminar._shape_d(shape_km, m_e)
        return c_f/2 - (2+shape_d-m_e**2)*(delta_m/u_e)*du_e_dx
    
    @staticmethod
    def _shape_d(shape_km: InputParam, m_e: InputParam) -> InputParam:
        'Add description here'
        return shape_km*(1+0.113*m_e**2) + 0.29*m_e**2

    @staticmethod
    def _dshape_k_dre_m() -> InputParam:
        'Add description here'
        return 0

    @staticmethod
    def _shape_k(shape_km: InputParam) -> npt.NDArray:
        'Add description here'

        shape_km = np.asarray(shape_km)
        #TODO add a similar (to heads method) checking scheme for 'reasonable' values?

        def lam_Hk_low(shape_km: InputParam) -> InputParam:
            return  1.515 + 0.076*((4 - shape_km)**2)/shape_km

        def lam_Hk_high(shape_km: InputParam) -> InputParam:
            return  1.515 + 0.040*((shape_km - 4)**2)/shape_km

        return np.piecewise(shape_km, [shape_km <= 4, shape_km > 4], [lam_Hk_low, lam_Hk_high])



    @staticmethod
    def _c_D(shape_km: InputParam, shape_k: InputParam, re_delta_m: InputParam) -> npt.NDArray:
        'Add description here'

        shape_km = np.asarray(shape_km)
        shape_k = np.asarray(shape_k)
        #TODO add a similar (to heads method) checking scheme for 'reasonable' values, maybe not?
        temp = shape_k/(2*re_delta_m)

        def lam_CD_low(shape_km: InputParam) -> InputParam:
            return (0.207 + 0.00205*(4 - shape_km)**5.5)

        def lam_CD_high(shape_km: InputParam) -> InputParam:
            return (0.207 - 0.003*((shape_km - 4)**2)/(1 + 0.02*shape_km**2))
        temp2 = np.piecewise(shape_km, [shape_km <= 4, shape_km > 4], [lam_CD_low, lam_CD_high])
        return temp*temp2

    @staticmethod
    def _dshape_k_dshape_km(shape_km: InputParam) -> npt.NDArray:
        'Add description here'
        shape_km = np.asarray(shape_km)
        #TODO add a similar (to heads method) checking scheme for 'reasonable' values
        def dshapek_low(shape_km: InputParam) -> InputParam:
            return 0.076*(-2*(4 - shape_km)/shape_km - (4 - shape_km)**2/shape_km**2)
        
        def dshapek_high(shape_km: InputParam) -> InputParam:
            return 0.04*(2*(shape_km - 4)/shape_km - (shape_km - 4)**2/shape_km**2)
        
        return np.piecewise(shape_km, [shape_km <= 4, shape_km > 4], [dshapek_low, dshapek_high]) #not sure what's wrong

    @staticmethod
    def _dshape_k_dx(delta_m: InputParam, shape_km: InputParam, u_e: InputParam, du_e: InputParam, m_e: InputParam, re_delta_m: InputParam) -> InputParam:
        # dH*/dxi, eq 11
        # requires delta_m, shape_ke, H**, u_e, du_e, CD, C_f, shape_d
        c_f = DrelaGilesLaminar._c_f_dg(shape_km, re_delta_m)
        shape_den = DrelaGilesLaminar._shape_den(shape_km, m_e)
        shape_k = DrelaGilesLaminar._shape_k(shape_km)
        c_D = DrelaGilesLaminar._c_D(shape_km,shape_k,re_delta_m)
        shape_d = DrelaGilesLaminar._shape_d(shape_k, m_e)
        temp1 = 2*c_D - 0.5*shape_k*c_f
        temp2 = (2*shape_den + shape_k*(1 - shape_d))*delta_m*du_e/u_e
        return (1/delta_m)*(temp1 - temp2)


class _DrelaGilesSeparationEvent(TermEvent):
    """
    Detects separation and will terminate integration when it occurs.

    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.

    Attributes
    ----------
        cf_crit: Displacement shape factor value that indicates separation
    """

    def __init__(self, cf_crit: float, u_e:Callable[[InputParam],npt.NDArray], nu: float) -> None:
        """
        Initialize separation criteria for Head's method.

        Parameters
        ----------
        shape_d_crit : float
            Critical displacement shape factor for separatation.
        """
        super().__init__()
        self._cf_crit = cf_crit
        self._u_e = u_e #TODO this idea legal? Idea is to input the u_e function directly and have the term event call it its own
        self._nu = nu

    @override
    def _call_impl(self, x: float, f: npt.NDArray) -> float:
        """
        Determine if Head method integrator should terminate.

        This will terminate once the displacement shape factor becomes greater
        than critical H_d.

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

        # f[0] is momentum thickness, f[1] is kinematic shape factor, f[2] is n_tilde
        shape_km = f[1]
        u_e = self._u_e(x) #TODO thus we could use the function here, since we have x
        re_delta_m = u_e*f[0]/self._nu
        current_cf = DrelaGilesLaminar._c_f_dg(shape_km,re_delta_m)
        return float(current_cf - self._cf_crit)

    @override
    def event_info(self) -> Tuple[TermReason, str]:
        return TermReason.SEPARATED, ""

class _DrelaGilesTransitionEvent(TermEvent):
    def __init__(self, n_tildae_crit: float) -> None:
        super().__init__()
        self.n_tildae_crit = n_tildae_crit

    @override
    def _call_impl(self, x: float, f: npt.NDArray) -> float:
        # Returns the difference between the critical n tildae value and the current n tildae value
        return self.n_tildae_crit - f[2] #kill event happens when sign changes

    # is n_tildae_crit properly declared, also default set to 9?
    @override
    def event_info(self) -> Tuple[TermReason, str]:
        # Is this the right way to do it?
        return TermReason.TRANSITIONED, ""