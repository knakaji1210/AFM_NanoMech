import numpy as np
from numba import jit, types, float32,prange
from numba.typed import List
@jit("Tuple((f8[:],f8[:],f8[:]))(f4[:,:],f4[:,:])",nopython=True)
def new_calculate2(rtdelta,rtforce):
    o_deform=np.empty(65536,dtype=np.float64)
    rtadhesive_x=np.empty(65536,dtype=np.float64)
    rtadhesive_y=np.empty(65536,dtype=np.float64)
    for i in range(65536):
        select_rtforce=rtforce[i,:]
        rtminimum_p=np.nanmin (select_rtforce)
        rtminimum_pindex=np.argmin(select_rtforce)
        delta2minimum=rtdelta[i,rtminimum_pindex]
        max_mini_p=select_rtforce[rtminimum_pindex:]
        o_force_index=np.argmin(np.abs (max_mini_p))
        o_deformationlist=rtdelta[i,rtminimum_pindex:]
        o_deformation=o_deformationlist[o_force_index]
        o_deform[i]=o_deformation
        rtadhesive_x[i]=delta2minimum
        rtadhesive_y[i]=rtminimum_p
    return rtadhesive_x,rtadhesive_y,o_deform

# def new_calculate3(rtdelta,rtforce):
#     o_deform=np.empty(65536,dtype=np.float64)
#     rtadhesive_x=np.empty(65536,dtype=np.float64)
#     rtadhesive_y=np.empty(65536,dtype=np.float64)
#     for i in prange(65536):
#         select_rtforce=rtforce[i,:]
#         rtminimum_p=np.nanmin (select_rtforce)
#         rtminimum_pindex=np.argmin(select_rtforce)
#         delta2minimum=rtdelta[i,rtminimum_pindex]
#         max_mini_p=select_rtforce[rtminimum_pindex:]
#         o_force_index=np.argmin(np.abs (max_mini_p))
#         o_deformationlist=rtdelta[i,rtminimum_pindex:]
#         o_deformation=o_deformationlist[o_force_index]
#         o_deform[i]=o_deformation
#         rtadhesive_x[i]=delta2minimum
#         rtadhesive_y[i]=rtminimum_p
#     return rtadhesive_x,rtadhesive_y,o_deform

# @jit("Tuple((ListType(f8[::1]),ListType(f4[:]),ListType(f8)))(f4[:,:],f4[:,:],f4,f4,f4)",nopython=True)
# def linearfit(rtdelta,rtforce,R,startpoint,endpoint):
#     Maugis_deltalist=List()
#     range_deltalist=List()
#     w_list=List()
#     # Maugis_deltalist=np.empty(65536,type=np.float64)
#     # range_deltalist=np.empty(65536,dtype=np.float64)
#     # rtadhesive_y=np.empty(65536,dtype=np.float64)
#     for i in prange(65536):
#         rangeforce=rtforce[i,:]
#         rangedelta=rtdelta[i,:]
#         Fe=np.nanmin(rangeforce)
#         Fe_index=np.argmin (rangeforce)
#         rangeforce=rangeforce[Fe_index:]
#         rangedelta=rangedelta[Fe_index:]
#         Fmax=rangeforce[-1]
#         F_range=Fmax-Fe
#         F_start=F_range*startpoint
#         F_start=Fe+F_start
#         F_end=F_range*(1-endpoint)
#         F_end=Fmax-F_end
#         F_start_index=np.argmin(np.abs(rangeforce-F_start))
#         F_end_index=np.argmin (np.abs (rangeforce-F_end))
#         # F_start_index=np.interp(F_start,rangedelta,rangeforce)
#         # F_end_index=np.interp(F_end,rangedelta,rangeforce)
#         rangeforce=rangeforce[F_start_index:F_end_index+1]
#         rangedelta=rangedelta[F_start_index:F_end_index+1]
#         Maugis_F=-(3/2)*(rangeforce/Fe)
#         w=-2*Fe/(3*(np.pi*R))
#         Maugis_delta=(3*Maugis_F+3+np.sqrt (6*Maugis_F+9))/(3*(Maugis_F+3+np.sqrt (6*Maugis_F+9))**(1/3))
#         range_deltalist.append(rangedelta)
#         Maugis_deltalist.append(Maugis_delta)
#         w_list.append(w)
#     return Maugis_deltalist,range_deltalist,w_list
#
@jit("Tuple((f8[:],f8[:]))(f8[:,:],f8[:,:],f8,f8,f8,f8)",nopython=True)
def linearfit2(rtdelta,rtforce,poisson,R,startpoint,endpoint):
    modulus_list=np.empty(65536)
    w_list=np.empty(65536)
    for i in range(65536):
        rangeforce=rtforce[i,:]
        rangedelta=rtdelta[i,:]
        Fe=np.nanmin(rangeforce)
        Fe_index=np.argmin (rangeforce)
        rangeforce=rangeforce[Fe_index:]
        rangedelta=rangedelta[Fe_index:]
        Fmax=rangeforce[-1]
        F_range=Fmax-Fe
        F_start=F_range*startpoint
        F_start=Fe+F_start
        F_end=F_range*(1-endpoint)
        F_end=Fmax-F_end
        F_start_index=np.argmin(np.abs(rangeforce-F_start))
        F_end_index=np.argmin (np.abs (rangeforce-F_end))
        # F_start_index=np.interp(F_start,rangedelta,rangeforce)
        rangeforce=rangeforce[F_start_index:F_end_index+1]
        rangedelta=rangedelta[F_start_index:F_end_index+1]
        Maugis_F=-(3/2)*(rangeforce/Fe)
        w=-2*Fe/(3*(np.pi*R))
        Maugis_delta=(3*Maugis_F+3+np.sqrt (6*Maugis_F+9))/(3*(Maugis_F+3+np.sqrt (6*Maugis_F+9))**(1/3))
        A=np.vstack ((rangedelta,np.ones (len (rangedelta)))).T
        tmp_a,tmp_b=np.linalg.lstsq(A,Maugis_delta,rcond=-1)[0]
        K=((tmp_a**3)*(np.pi**2)*(w**2)*R)**(1/2)
        modulus=(3/4)*(1-(poisson**2))*K
        w_list[i]=w
        modulus_list[i]=modulus
    return modulus_list,w_list


