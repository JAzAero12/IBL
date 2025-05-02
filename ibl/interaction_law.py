import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def interaction_law(s_vec:npt.NDArray, u_inf:float, u_e_input:npt.NDArray, nu_inf:float, debug=False):

    #Function takes in inviscid u_e (as an array of values), and 
    # creates corrections to make the profile less susceptible to Goldstein's Singluarity

    #Preprocess the given u_e? TODO is this cheating?
    u_e = u_e_input.copy()
    u_e_fun = CubicSpline(s_vec,u_e)
    due_fun = u_e_fun.derivative()
    ddue_fun = u_e_fun.derivative(2)

    if debug:
        fig, ddue = plt.subplots()
        ddue.plot(s_vec,ddue_fun(s_vec),marker='o',markersize=4,color='#BD8B13')
        ddue.set_ylabel(r'$\frac{d^2u_e}{d\xi^2}$')
        ddue.set_xlabel(r'$\xi$')
        ddue.set_title('Debug Plot, interaction_law.py')
        ddue.set_ylim([-200,200])

    ramploc = -1
    te_u_e_scal = .99
    for i, s_val in enumerate(s_vec):
        if abs(ddue_fun(s_vec[int(-1*(i+1))]))<=50 and int(-1*(i+1)) >= -10 and u_e[int(-1*(i+1))] > te_u_e_scal*u_inf and due_fun(s_vec[int(-1*(i))])<0: #find a 'smooth' area
        #if int(-1*(i+1)) <= -5 and u_e[int(-1*(i+1))] > u_inf:
            ramploc = int(-1*(i+1))
            break

    if ramploc != -1:
        #Extend local velocity derivative of ramploc to the trailing edge
        ramploc_due = due_fun(s_vec[ramploc])

        def te_velramp(s):
            return u_e[ramploc] + ramploc_due*(s-s_vec[ramploc])
        
        u_e[ramploc+1:] = te_velramp(s_vec[ramploc+1:])

    #Create the wake approximation

    chord_len = abs(s_vec[-1] - s_vec[0])

    ReX = s_vec*u_e/nu_inf
    ReX[abs(ReX) < 1e-9] = 1e-9
    #u_e_len = len(u_e)
    delta_d_turb_fp = .046875 * s_vec / ReX**.2

    #Create a wake approximation, clustered
    wake_cluster = np.linspace(0,1,len(s_vec))**1. #clustering does not appear to provide any benefit
    wake_approx_s = s_vec[-1] + chord_len*wake_cluster
    wake_approx_s = wake_approx_s[1:]

    #The default wake velocity profile, and if TE velocity is faster than u_inf, just do linear ramp to u_inf
    wake_approx_ue = np.linspace(u_e[-1],u_inf,len(wake_approx_s))

    total_s_vec = np.concatenate((s_vec,wake_approx_s))
    total_ue_vec = np.concatenate((u_e,wake_approx_ue))

    if u_e[-1] < u_inf and ramploc != -1: #Fancier ramp if these condition is met
        #First make u_e first point on wake a shallower ramp of previous u_e points
        wake_fstpt_ue = u_e[-1] + .8*ramploc_due*(total_s_vec[len(u_e)]-s_vec[-1])
        midpt_u_e = u_e[-1] + 0.5*(u_inf - u_e[-1])    
        #Cubic ramp to u_inf
        ramp_end_idx = len(u_e) + 15
        ramp_svec = [s_vec[-1],total_s_vec[len(u_e)],total_s_vec[len(u_e)+int(0.5*(ramp_end_idx-len(u_e)))],total_s_vec[ramp_end_idx-1],total_s_vec[ramp_end_idx]]
        ramp_uvec = [u_e[-1],  wake_fstpt_ue,        midpt_u_e,                                    u_inf-0.01,                 u_inf]

        ramp_spline = CubicSpline(ramp_svec,ramp_uvec)

        total_ue_vec[len(u_e):ramp_end_idx+1] = ramp_spline(total_s_vec[len(u_e):ramp_end_idx+1])
        total_ue_vec[ramp_end_idx+1:] = u_inf
        #def vel_ramp(s_vals):
        #    l = u_inf - u_e[-1]
        #    midpt = .05*chord_len + s_vals[0]
        #    k = 80.
        #    return u_e[-1] + l/(1.+np.exp(-1*k*(s_vals - midpt)))
        #wake_approx_ue = vel_ramp(wake_approx_s)

    delta_d_wake = delta_d_turb_fp[-1] * np.exp(-(wake_approx_s - s_vec[-1])/(wake_approx_s[-1] - s_vec[-1]))
    total_delta_d = np.concatenate((delta_d_turb_fp,delta_d_wake))

    #total_s_vec = np.concatenate((s_vec,wake_approx_s))
    #total_ue_vec = np.concatenate((u_e,wake_approx_ue))
    #Plot for debugging
    if debug:
        fig, u_e_total = plt.subplots()
        u_e_total.plot(total_s_vec,total_ue_vec,marker='o',markersize=4,color='#154734')
        u_e_total.plot([s_vec[ramploc],s_vec[ramploc]],[min(u_e),max(u_e)],linestyle='--',color='black')
        u_e_total.plot([s_vec[-1],s_vec[-1]],[min(u_e),max(u_e)],linestyle='--',color='black')
        u_e_total.set_xlabel(r'$\xi$')
        u_e_total.set_ylabel(r'$u_e$')
        u_e_total.set_title('Debug Plot, interaction_law.py')

    #total_delta_d = np.concatenate((delta_d_turb_fp,delta_d_wake))

    d_vec = total_ue_vec*total_delta_d

    if d_vec[0] == np.nan:
        d_vec[0] = .5*d_vec[1] #To remove the nan at the start of the array

    #Use a cubic spline to smooth last few points?
    #end_pt    = total_s_vec[len(s_vec)] #Last point of spline
    #start_idx = -10
    #start_pt  = s_vec[start_idx]#First point of spline
    #second_pt = s_vec[start_idx+1]#Point immediately after to help shape spline
    #te_spline = CubicSpline([start_pt,second_pt,end_pt],
    #                        [d_vec[len(s_vec)+start_idx],d_vec[len(s_vec)+start_idx+1],d_vec[len(s_vec)]])
    #Use spline
    #for i in range(abs(start_idx)):
    #    d_vec[len(s_vec)-(i+1)] = te_spline(total_s_vec[len(s_vec)-(i+1)])

    #Plot for debugging
    if debug:
        fig, d_vec_plot = plt.subplots()
        d_vec_plot.plot(total_s_vec,d_vec,marker='o',markersize=4,color='#3A913F')
        d_vec_plot.plot([s_vec[-1],s_vec[-1]],[min(d_vec),max(d_vec)],linestyle='--',color='black') #The dip at TE from the fact that vel is forced at very last value
        d_vec_plot.set_xlabel(r'$\xi$')
        d_vec_plot.set_ylabel(r'$u_e\delta^*$')
        d_vec_plot.set_title('Debug Plot, interaction_law.py')

    c_mat = c_maker_func(total_s_vec)
    vel_corr = c_mat @ d_vec  
    #Every column value of c_mat and d_vec are multiplied to eachother, sum of that is for the specific row value
    #not working v 
    #vel_corr = np.zeros_like(total_s_vec)
    #for i in range(len(total_s_vec)):
    #    vel_corr[i] = sum(c_mat[:,i]*d_vec) - 2*c_mat[i,i]*d_vec[i] #eq 5.7.5?

    return vel_corr, d_vec, c_mat, total_s_vec, u_e


def c_maker_func(total_s_vec):
    c_mat = np.zeros([len(total_s_vec),len(total_s_vec)]) #The j values may exceed the square dimensions: pg 75
    e_mat = np.zeros([len(total_s_vec),len(total_s_vec)])
    #Creating E matrix
    #TODO when i = j its sus
    for i in range(0,len(total_s_vec)): #rows
        for j in range(0,len(total_s_vec)): #columns
            if j == 0:
                e_mat[i,j] = 0
            elif j == i:
                if i+1 == len(total_s_vec): #double check this portion of logic
                    #e_mat[i,j] = 0 #this appears to be a temporary fix for when i goes out of index during the loop
                    temp = (0 - total_s_vec[i])/(0 - total_s_vec[i-1])
                    temp2 = np.log(abs((total_s_vec[i] - total_s_vec[i-1])/(total_s_vec[i] - 0)))
                    e_mat[i,j] = (temp*temp2 + 2.)/(total_s_vec[i] - total_s_vec[i-1])
                else:
                    temp = (total_s_vec[i+1] - total_s_vec[i])/(total_s_vec[i+1] - total_s_vec[i-1])
                    temp2 = np.log(abs((total_s_vec[i] - total_s_vec[i-1])/(total_s_vec[i] - total_s_vec[i+1])))
                    e_mat[i,j] = (temp*temp2 + 2.)/(total_s_vec[i] - total_s_vec[i-1])
                #print([e_mat[i,j],e_mat[i-1,j]])
            elif j == i+1:
                temp = (total_s_vec[i] - total_s_vec[i-1])/(total_s_vec[i+1] - total_s_vec[i-1])
                temp2 = np.log(abs((total_s_vec[i] - total_s_vec[i-1])/(total_s_vec[i] - total_s_vec[i+1])))
                e_mat[i,j] = (temp*temp2 - 2.)/(total_s_vec[i+1] - total_s_vec[i])
            else: #i != j
                temp = np.log(abs((total_s_vec[i] - total_s_vec[j-1])/(total_s_vec[i] - total_s_vec[j])))
                e_mat[i,j] = temp/((total_s_vec[j] - total_s_vec[j-1]))
    #C matrix
    for i in range(1,len(total_s_vec)): #rows
        for j in range(0,len(total_s_vec)): #columns
            if j+1 ==  len(total_s_vec):
                e_val = 0
            else:
                e_val = e_mat[i,j+1]
            c_mat[i,j] = 1./np.pi * (e_mat[i,j] - e_val)

    return c_mat