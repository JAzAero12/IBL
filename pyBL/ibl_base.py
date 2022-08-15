#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:17:53 2022

@author: ddmarsha
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.misc import derivative as fd
from abc import ABC, abstractmethod


TERMINATION_MESSAGES = {0: "Completed",
                        -1: "Separated",
                        1: "Transition",
                        -99: "Unknown Event"}


class IBLResult:
    """Bunch object representing the results of the IBL integration.
    
    Attributes
    ----------
        x_end: x-location of end of integration
        F_end: State value(s) at end of integration
        status: Reason integration terminated:
            * 0: Reached final distance
            * -1: Separation occured at x_end
            * 1: Transition occured at x_end
            * Other values can be used by specific implementations
        message: Description of termination reason
        success: True if solver successfully completed
    """
    def __init__(self, x_end = np.inf, F_end = np.inf,
                 status = -99, message = "Not Set", success = False):
        self.x_end = x_end
        self.F_end = F_end
        self.status = status
        self.message = message
        self.success = success


class IBLTermEventBase(ABC):
    """
    Base class for a termination event for IBL solver.
    
    The two abstract methods that have to be implemented are event_info and 
    _call_impl. Classes derived from this class can either be used within an 
    IBL implementation or as a parameter into the solve method.
    """
    def __init__(self):
        self.terminal = True
        
    def __call__(self, x, F):
        """
        ODE solver is going to call this method to determine if the integration 
        should terminate.
        
        Args
        ----
            x: Current x-location of the integration
            F: Current state value(s)
        
        Returns
        -------
            Floating point number that is zero when the solver should stop
        """
        return self._call_impl(x, F)
    
    @abstractmethod
    def event_info(self):
        """
        Method returns information about the purpose of this event. This is 
        used to provide feedback as to what caused the integration to terminate
        and any other helpful information.
        
        Returns
        -------
            2-tuple of event index and string providing any extra information.
            Event index should be -1 for separation and 1 for transition. Other 
            values may not be handled correctly.
        """
        pass
    
    @abstractmethod
    def _call_impl(self, x, F):
        """
        Information used to determine if IBL integrator should terminate.
        
        The return value is used in a root finder to find what x,F will result
        in the termination of the integrator. The function should return zero 
        when the integrator should terminate, and change signs around the
        termination state.
        
        Args
        ----
            x: Current x-location of the integration
            F: Current state value(s)
        """
        pass


class IBLBase(ABC):
    """
    The base class for integral boundary layer classes.
    
    This encapsulates the common features and needed parameters for all IBL 
    methods. At the very least it provides the inteface that is expected for all
    IBL classes.
    
    Attributes
    ----------
        _U_e: Function representing the edge velocity profile
        _dU_edx: Function representing the streamwise derivative of the edge 
                 velocity
        _d2U_edx2: Function representing the streamwise second derivative of the
                   edge velocity
        _kill_events: List of events that should be passed into ODE solver that
                      might cause the integration to terminate early
        _F: Piecewise polynomials representing the state variables from the
            ODE solution
    
    Raises
    ------
        ValueError: if configuration parameter is invalid (see message)
    """
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None):
        # set the velocity terms
        if U_e is None:
            if (dU_edx is not None):
                raise ValueError("Must specify U_e if specifying dU_edx")
            if (d2U_edx2 is not None):
                raise ValueError("Must specify U_e if specifying d2U_edx2")
            self._U_e = None
            self._dU_edx = None
            self._d2U_edx2 = None
        else:
            self.set_velocity(U_e, dU_edx, d2U_edx2)
        
        # initialize other parameters
        self._kill_events = None
        self._F = None

    def set_velocity(self, U_e, dU_edx = None, d2U_edx2 = None):
        """
        Set the edge velocity relations.
        
        A number of different ways can be used to set the velocity relation and
        its derivatives.
        * U_e can be a 2-tuple of xpoints and velocity values. In this case a
          cubic spline will be created and the derivative functionss will be
          taken from the cubic spline.
        * U_e and the derivatives can be callable objects.
            * If the first derivative object is provided but not the second 
              derivative object, then if the first derivative object has a
              method called 'derivative' then that method will be used to
              generate the second derivative object. Otherwise the second
              derivative will be approximated by finite differences of the first
              derivative.
            * If neither derivative objects are provided, then if U_e has a
              method called 'derivative' then that method will be used to
              generate both derivative objects. Otherwise the derivative
              objects will be created from finite difference approximations.
        
        Args
        ----
        U_e: Edge velocity
        dU_edx: First derivative of the edge velocity
        d2U_edx2: Second derivative of the edge velocity
        
        Raises
        ------
            ValueError: if configuration parameter is invalid (see message)
        """
        # check if U_e is callable
        if callable(U_e):
            self._U_e = U_e
            
            # if dU_edx not provided then use finite differences
            if dU_edx is None:
                if d2U_edx2 is not None:
                    raise ValueError("Can only pass second derivative if first "
                                     "derivative was specified")
                
                # if U_e has derivative method then use it
                if (hasattr(U_e, 'derivative') and
                        callable(getattr(U_e, 'derivative'))):
                    self._dU_edx = U_e.derivative()
                    self._d2U_edx2 = U_e.derivative(2)
                else:
                    self._dU_edx = lambda x: fd(self._U_e, x, 1e-4,
                                                n=1, order=3)
                    self._d2U_edx2 = lambda x: fd(self._U_e, x, 1e-4,
                                                  n=2, order=3)
            else:
                if not callable(dU_edx):
                    raise ValueError("Must pass in callable object for first "
                                     "derivative if callable U_e given")
                self._dU_edx = dU_edx
                
                # if d2U_edx2 not provied then use finite difference
                if d2U_edx2 is None:
                    # if dU_edx has derivative method then use it
                    if (hasattr(dU_edx, 'derivative') and
                            callable(getattr(dU_edx, 'derivative'))):
                        self._d2U_edx2 = dU_edx.derivative()
                    else:
                        self._d2U_edx2 = lambda x: fd(self._dU_edx, x, 1e-5,
                                                      n=1, order=3)
                else:
                    if not callable(dU_edx):
                        raise ValueError("Must pass in callable object for "
                                         "first derivative if callable U_e "
                                         "given")
                    
                    self._d2U_edx2 = d2U_edx2
        else:
            # if is 2-tuple the assume x, U_e pairs to build Cubic Spline
            if len(U_e) == 2:
                x_pts = np.asarray(U_e[0])
                U_e_pts = np.asarray(U_e[1])
                npts = x_pts.shape[0]
                # check to make sure have two vectors of same length suitable
                #   for building splines
                if (x_pts.ndim != 1):
                    raise ValueError("First element of U_e 2-tuple must be 1D "
                                     "vector of distances")
                if (U_e_pts.ndim != 1):
                    raise ValueError("Second element of U_e 2-tuple must be 1D "
                                     "vector of Velocities")
                if npts != U_e_pts.shape[0]:
                    raise ValueError("Vectors in U_e 2-tuple must be of same "
                                     "length")
                if npts < 2:
                    raise ValueError("Must pass at least two points for edge "
                                     "velocity")
                
                U_e_spline = CubicSpline(x_pts, U_e_pts)
                self.set_velocity(U_e_spline)
            else:
                # otherwise unknown velocity input
                raise ValueError("Don't know how to use {} to initialize "
                                 "velocity".format(U_e))
    
    def solve(self, xrange, y0i, rtol=1e-8, atol=1e-11, term_event = None):
        """
        Solve the ODEs to determine the boundary layer properties.
        
        Args
        ----
        xrange: 2-tuple with the start and end x-locations of integration
        y0i: Initial condition of the state vector for integration
        rtol: Relative tolerance for integration scheme
        atol: Absolute tolerance for integration scheme
        term_event: List of classes based on IBLTermEventBase, in addition to
                    any internal ones, to be used to determine if the 
                    integration should terminate before the end location. These
                    should mostly be for transition to turbulent boundary layer
                    or separation.
        
        Returns
        -------
            Bunch object (IBLResult) with information about the solution 
            process and termination.
        """
        ## setup the ODE solver
        xrange = np.asarray(xrange)
        y0 = np.asarray(y0i)
        if y0.ndim == 0:
            y0 = [y0i]
        
        kill_events = []
        if self._kill_events is not None:
            kill_events = kill_events + self._kill_events

        if term_event is None:
            if self._kill_events is None:
                kill_events = None
        else:
            if isinstance(term_event, list):
                kill_events = kill_events + term_event
            else:
                kill_events.append(term_event)

        rtn = solve_ivp(fun = self._ode_impl, t_span = xrange, y0 = y0,
                        method = 'RK45', dense_output = True,
                        events = kill_events, rtol = rtol, atol = atol)
        
        # if completed gather info
        self._solution = None
        x_end = xrange[0]
        F_end = y0
        status = -99
        message = rtn.message
        if rtn.success:
            self._solution = rtn.sol
            
            # if terminated on time or early figure out why
            if rtn.status == 0:
                x_end = rtn.t[-1]
                F_end = rtn.sol(x_end)
                status = 0
                message = ""
            elif rtn.status == 1:
                message = "Event not found."
                for i, xe in enumerate(rtn.t_events):
                    if xe.shape[0] > 0:
                        x_end = xe[0]
                        F_end = rtn.sol(x_end)
                        status, message = kill_events[i].event_info()
                        break
            else:
                status = -99

        if len(message)> 0:
            message = "{}: {}".format(TERMINATION_MESSAGES.get(status), message)
        else:
            message = TERMINATION_MESSAGES.get(status)
        return IBLResult(x_end = x_end, F_end = F_end, status = status,
                         message = message, success = rtn.success)
    
    def U_e(self, x):
        """
        Return the inviscid edge velocity at specified location(s)
        
        Args
        ----
            x: distance along surface
            
        Returns
        -------
            Inviscid edge velocity
        """
        if (self._U_e is not None):
            return self._U_e(x)
        else:
            raise ValueError("U_e was not set")
            return None
    
    def dU_edx(self, x):
        """
        Return the streamwise derivative of the inviscid edge velocity at 
        specified location(s)
        
        Args
        ----
            x: distance along surface
            
        Returns
        -------
            Derivative of inviscid edge velocity
        """
        if (self._dU_edx is not None):
            return self._dU_edx(x)
        else:
            raise ValueError("dU_edx was not set")
            return None
    
    def d2U_edx2(self, x):
        """
        Return the streamwise second derivative of the inviscid edge velocity at 
        specified location(s)
        
        Args
        ----
            x: distance along surface
            
        Returns
        -------
            Second derivative of inviscid edge velocity
        """
        if (self._d2U_edx2 is not None):
            return self._d2U_edx2(x)
        else:
            raise ValueError("d2U_edx2 was not set")
            return None
    
    @abstractmethod
    def U_n(self, x):
        """
        Calculate the transpiration velocity
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        pass
    
    @abstractmethod
    def delta_d(self, x):
        """
        Calculate the displacement thickness
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        pass
    
    @abstractmethod
    def delta_m(self, x):
        """
        Calcualte the momentum thickness
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        pass
    
    @abstractmethod
    def H(self, x):
        """
        Calculate the shape factor
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        pass
    
    @abstractmethod
    def tau_w(self, x):
        """
        Calculate the wall shear stress
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        pass
    
    def _add_kill_event(self, ke):
        if self._kill_events is None:
            self._kill_events = [ke]
        else:
            self._kill_events.append(ke)
    
    ## These need to be modified or removed
    def y(self,x):
        #returns m*n array, where m is len(x) and n is length(y)
        x_array = x #must be array
        #x_array = np.array([x])
        y_array = np.zeros([len(x),len(self._ode.y)])
        for i in range(len(x_array)):
            for j in range(len(self.dense_output_vec)): #-1
                if (x_array[i] >= self.x_vec[j]) & (x_array[i] <= self.x_vec[j+1]):
                    y_array[i,:] = self.dense_output_vec[j](x_array[i])
                    break
                 
        return y_array
    
    def yp(self,x):
        #Uses Dense Output construct to return derivative with polynomial
        x_array = x #must be array
        #x_array = np.array([x])
        yp_array = np.zeros([len(x),len(self._ode.y)])
        for i in range(len(x_array)):
            for j in range(len(self.dense_output_vec)): #-1
                if (x_array[i] >= self.x_vec[j]) & (x_array[i] <= self.x_vec[j+1]):
                    #y_array = np.append(y_array, [[self._piecewise_funs[j](x_array[i])]],axis=0)
                    #print(x_array[i])
                    #y_array[i,:] = self._piecewise_funs[j](x_array[i])
                    xdist = (x_array[i] - self.dense_output_vec[j].t_old) / self.dense_output_vec[j].h
                    if np.array(x_array[i]).ndim == 0:
                                #p = np.tile(x, testfit.order + 1)
                                p = np.tile(xdist, self.dense_output_vec[j].order + 1)
                                # TODO: This produces error when xdist=0 because p becomes vector of zeros
                                #       See issue #21
                                p = np.cumprod(p)/p
                    else:
                                p = np.tile(xdist, (self.dense_output_vec[j].order + 1, 1))
                                p = np.cumprod(p, axis=0)/p
                    #term1 = self.dense_output_vec[j].h h actually disappears
                    term2 = np.arange(1,self.dense_output_vec[j].order+2)
                    term3 = self.dense_output_vec[j].Q
                    term4 = p
                    yp_array[i,:] = np.dot(term2*term3, term4) 
                                        #yp_array[i,:] = self.dense_output_vec[j](x_array[i])
                    
                    break
        return yp_array        

