from typing import Tuple, cast, Optional, Any
from typing_extensions import override
from typing import Callable
import numpy as np
import numpy.typing as npt

from ibl.ibl_method import IBLMethod
from ibl.ibl_method import TermReason
from ibl.ibl_method import TermEvent
from ibl.initial_condition import ManualCondition
import matplotlib.pyplot as plt
from ibl.typing import InputParam
from ibl.transition_coupler import transition_coupler
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['figure.figsize'] = [13, 8]
plt.rcParams['legend.loc'] = 'lower center'
plt.rcParams['legend.borderaxespad'] = -5.5
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 3.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.default"] = "rm"
file_name = "Transition_Coupler_SimpleTests"
file_name = os.path.dirname(os.path.abspath(__file__))+'\\'+file_name
print(file_name)
if not os.path.exists(file_name):
    os.mkdir(file_name)

class Dummy_Model_Laminar(IBLMethod):

    def __init__(self, nu: float = 1.0, U_e: Optional[Any] = None,
                 dU_edx: Optional[Any] = None, d2U_edx2: Optional[Any] = None,
                 transition_checker: float = .5,  ic = None) -> None:
        super().__init__(nu=nu, u_e=U_e, du_e=dU_edx, ic = ic) 
        self.set_transition_checker_critical(transition_checker)
        self.transition_checker_init = 0


    def set_transition_checker_critical(self, transition_checker: float) -> None:
        self._set_kill_event(_TransitionEvent(transition_checker))

    @override
    def v_e(self, x: InputParam) -> npt.NDArray: #shouldn't need to change
        if self._solution is None:
            raise ValueError("No valid solution.")
        return np.zeros_like(x)

    @override
    def delta_d(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self._solution(x)[0]

    @override
    def delta_m(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self._solution(x)[1]

    @override
    def delta_k(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.") 
        return self._solution(x)[2]

    @override
    def shape_d(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self.delta_d(x)/self.delta_m(x)

    @override
    def shape_k(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self.delta_m(x)/self.delta_k(x)
    
    
    @override
    def tau_w(self, x: InputParam, rho: float) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return np.zeros_like(x)

    @override
    def dissipation(self, x: InputParam, rho: float) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return np.zeros_like(x)
    
    @override
    def _ode_setup(self) -> Tuple[npt.NDArray, float, float]:
       
        return np.array([self._ic.delta_d(),self._ic.delta_m(),self._ic.delta_k(),self.transition_checker_init]), 1e-8, 1e-11
    
    @override
    def _ode_impl(self, x: InputParam,
                  f: npt.NDArray) -> npt.NDArray:
        f_p = np.zeros_like(f)
        f_p[0] = .01
        f_p[1] = .01
        f_p[2] = .01
        return f_p
    

class _TransitionEvent(TermEvent):
    def __init__(self, transition_checker: float) -> None:
        super().__init__()
        self.transition_checker = transition_checker 
    @override
    def _call_impl(self, x: float, f: npt.NDArray) -> float:
        return self.transition_checker - x #kill event happens when sign changes

    @override
    def event_info(self) -> Tuple[TermReason, str]:
        return TermReason.TRANSITIONED, ""
    
class Dummy_Model_Turbulent(IBLMethod):

    def __init__(self, nu: float = 1.0, U_e: Optional[Any] = None,
                 dU_edx: Optional[Any] = None, d2U_edx2: Optional[Any] = None,
                 transition_checker: float = 9,  ic = None) -> None:
        super().__init__(nu=nu, u_e=U_e, du_e=dU_edx, ic = ic)
        self.set_transition_checker_critical(transition_checker)
        self.transition_checker_init = 0

    def set_transition_checker_critical(self, transition_checker: float) -> None:
        self._set_kill_event(_TransitionEvent(transition_checker))

    @override
    def v_e(self, x: InputParam) -> npt.NDArray: #shouldn't need to change
        if self._solution is None:
            raise ValueError("No valid solution.")
        return np.zeros_like(x)

    @override
    def delta_d(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self._solution(x)[0]

    @override
    def delta_m(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self._solution(x)[1]

    @override
    def delta_k(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.") 
        return self._solution(x)[2]

    @override
    def shape_d(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return self.delta_d(x)/self.delta_m(x)

    @override
    def shape_k(self, x: InputParam) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        shape_km = self._solution(x)[1]
        return self.delta_m(x)/self.delta_k(x)
    
    
    @override
    def tau_w(self, x: InputParam, rho: float) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return np.zeros_like(x)

    @override
    def dissipation(self, x: InputParam, rho: float) -> npt.NDArray:
        if self._solution is None:
            raise ValueError("No valid solution.")
        return np.zeros_like(x)
    
    @override
    def _ode_setup(self) -> Tuple[npt.NDArray, float, float]:
       
        return np.array([self._ic.delta_d(),self._ic.delta_m(),self._ic.delta_k(),self.transition_checker_init]), 1e-8, 1e-11
    
    @override
    def _ode_impl(self, x: InputParam,
                  f: npt.NDArray) -> npt.NDArray:
        f_p = np.zeros_like(f)
        f_p[0] = .1
        f_p[1] = .1
        f_p[2] = .1
        return f_p



# Running the transition coupler
def u_e_fun(x):
    return np.ones_like(x)
def due_dx(x):
    return np.zeros_like(x)
x = np.linspace(0,1,50)

#Dummy preprocessor function that does not do anything
def dummy_turb_ic_preprocessor(lam_end_del_d,lam_end_del_m,lam_end_del_k,solution_range_laminar_end,nu,U_e_end,dU_edx_end,d2U_edx2_end):
    return lam_end_del_d,lam_end_del_m,lam_end_del_k

#Dummy preprocessor function that does something
def dummy_turb_ic_preprocessor2(lam_end_del_d,lam_end_del_m,lam_end_del_k,solution_range_laminar_end,nu,U_e_end,dU_edx_end,d2U_edx2_end):
    return 2*lam_end_del_d,2*lam_end_del_m,2*lam_end_del_k

#All laminar case
lam_model = Dummy_Model_Laminar(U_e=u_e_fun,dU_edx=due_dx,transition_checker=10.,ic=ManualCondition(0.,.05,.1))

whole_model = transition_coupler(solution_range=np.array([x[0],x[-1]]),laminar_model=lam_model,turbulent_class=Dummy_Model_Turbulent,U_e=u_e_fun,dU_edx=due_dx,transition_loc=None,turb_ic_preprocessor=dummy_turb_ic_preprocessor)

delta_ds = whole_model.delta_d(x)
delta_ms = whole_model.delta_m(x)
delta_ks = whole_model.delta_k(x)
loc = whole_model.transition_point
fig, figure = plt.subplots(constrained_layout=True)
figure.plot(x,delta_ds,marker='*',label='variable 1',color='#154734')
figure.plot(x,delta_ms,marker='*',label='variable 2',color='#BD8B13')
figure.plot(x,delta_ks,marker='*',label='variable 3',color='#5CB8B2')
figure.plot([loc,loc],[0,.14],color='#54585A',linestyle='--')

figure.legend(ncol=3)
figure.set_xlabel('s')
figure.set_ylabel('Result Variables')
fig.tight_layout()
fig.savefig(file_name+'\\'+'all_lam.png')

#Natural transition case
lam_model = Dummy_Model_Laminar(U_e=u_e_fun,dU_edx=due_dx,transition_checker=x[30],ic=ManualCondition(0.,.05,.1))

whole_model = transition_coupler(solution_range=np.array([x[0],x[-1]]),laminar_model=lam_model,turbulent_class=Dummy_Model_Turbulent,U_e=u_e_fun,dU_edx=due_dx,transition_loc=None,turb_ic_preprocessor=dummy_turb_ic_preprocessor)

delta_ds = whole_model.delta_d(x)
delta_ms = whole_model.delta_m(x)
delta_ks = whole_model.delta_k(x)
loc = whole_model.transition_point
fig, figure = plt.subplots(constrained_layout=True)
figure.plot(x,delta_ds,marker='*',label='variable 1',color='#154734')
figure.plot(x,delta_ms,marker='*',label='variable 2',color='#BD8B13')
figure.plot(x,delta_ks,marker='*',label='variable 3',color='#5CB8B2')
figure.plot([loc,loc],[0,.14],color='#54585A',linestyle='--')
figure.legend(ncol=3)
figure.set_xlabel('s')
figure.set_ylabel('Result Variables')
fig.tight_layout()
fig.savefig(file_name+'\\'+'natrual_tran.png')

#Forced transition case
lam_model = Dummy_Model_Laminar(U_e=u_e_fun,dU_edx=due_dx,transition_checker=x[30],ic=ManualCondition(0.,.05,.1))

whole_model = transition_coupler(solution_range=np.array([x[0],x[-1]]),laminar_model=lam_model,turbulent_class=Dummy_Model_Turbulent,U_e=u_e_fun,dU_edx=due_dx,transition_loc=x[20],turb_ic_preprocessor=dummy_turb_ic_preprocessor)

delta_ds = whole_model.delta_d(x)
delta_ms = whole_model.delta_m(x)
delta_ks = whole_model.delta_k(x)
chek1 = 2*delta_ds[20]
chek2 = 2*delta_ms[20]
chek3 = 2*delta_ks[20]

loc = whole_model.transition_point
fig, figure = plt.subplots(constrained_layout=True)
figure.plot(x,delta_ds,marker='*',label='variable 1',color='#154734')
figure.plot(x,delta_ms,marker='*',label='variable 2',color='#BD8B13')
figure.plot(x,delta_ks,marker='*',label='variable 3',color='#5CB8B2')
figure.plot([loc,loc],[0,.14],color='#54585A',linestyle='--')

figure.legend(ncol=3)
figure.set_xlabel('s')
figure.set_ylabel('Result Variables')
fig.tight_layout()
fig.savefig(file_name+'\\'+'forced_tran.png')

#Turbulent Preprocessor Function
lam_model = Dummy_Model_Laminar(U_e=u_e_fun,dU_edx=due_dx,transition_checker=x[30],ic=ManualCondition(0.,.05,.1))

whole_model = transition_coupler(solution_range=np.array([x[0],x[-1]]),laminar_model=lam_model,turbulent_class=Dummy_Model_Turbulent,U_e=u_e_fun,dU_edx=due_dx,transition_loc=x[20],turb_ic_preprocessor=dummy_turb_ic_preprocessor2)

delta_ds = whole_model.delta_d(x)
delta_ms = whole_model.delta_m(x)
delta_ks = whole_model.delta_k(x)
loc = whole_model.transition_point
fig, figure = plt.subplots(constrained_layout=True)
figure.plot(x,delta_ds,marker='*',label='variable 1',color='#154734')
figure.plot(x,delta_ms,marker='*',label='variable 2',color='#BD8B13')
figure.plot(x,delta_ks,marker='*',label='variable 3',color='#5CB8B2')
figure.plot([loc,loc],[0,.25],color='#54585A',linestyle='--')
figure.plot(loc,chek1,label=r'2 $\times$ variable 1 at transition',marker='o',color='#154734')
figure.plot(loc,chek2,label=r'2 $\times$ variable 2 at transition',marker='o',color='#BD8B13')
figure.plot(loc,chek3,label=r'2 $\times$ variable 3 at transition',marker='o',color='#5CB8B2')

figure.legend(ncol=2,borderaxespad=-8)
figure.set_xlabel('s')
figure.set_ylabel('Result Variables')
fig.tight_layout()
fig.savefig(file_name+'\\'+'preproc_func.png')

plt.show()
pass