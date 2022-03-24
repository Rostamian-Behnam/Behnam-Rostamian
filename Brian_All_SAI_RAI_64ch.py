from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scis
from scipy import signal
import random
import os
import pandas as pd
from IPython import get_ipython
from sklearn import preprocessing
start_scope()
############################
defaultclock.dt = 0.1 * ms
data=np.array([])
press=np.array([])
# ### Izh neuron model parameters
sig = 30
a = 0.1 / ms
b = 0.02 / ms
c = -65 * mV
d = 2 * mV / ms
vth = 30 * mV
sigma = sig * mV
tau = 10*ms
# Suppress resolution conflict warnings
BrianLogger.suppress_name('brian2.groups.group.Group.resolve.resolution_conflict')
# # Suppress code generation messages on the console
BrianLogger.suppress_hierarchy('brian2.input.timedarray')
# # Suppress preference messages even in the log file
BrianLogger.suppress_hierarchy('brian2.devices.device',
                                filter_log_file=True)
BrianLogger.suppress_name('brian2.codegen.runtime.cython_rt.cython_rt.failed_compile_test')
# # Suppress code generation messages on the console
BrianLogger.suppress_name('brian2.groups.group.Group.resolve.resolution_conflict')
# # Suppress code generation messages on the console
# Suppress resolution conflict warnings
BrianLogger.suppress_name('neurongroup_2')
BrianLogger.suppress_name('neurongroup_1')

#############################################
eq_SA = '''dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u + D_SA*mV/ms : volt
         du/dt = a*(b*v-u) : volt/second
         D_SA=k1*I_SA(t,i)   :   1
         '''     
eq_RA = '''dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u + D_RA*mV/ms : volt
         du/dt = a*(b*v-u) : volt/second
         D_RA=k2*I_RA(t,i)   :   1
         '''            
eq_RAII = '''dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u + D_RAII*mV/ms : volt
du/dt = a*(b*v-u) : volt/second
D_RAII=k3*I_RAII(t,i)   :   1
'''    
SA = NeuronGroup(21, eq_SA, threshold='v>vth', reset='v = c;u += d', method='euler')
SA.v=c
RA = NeuronGroup(43, eq_RA, threshold='v>vth', reset='v = c;u += d', method='euler')
RA.v=c
RAII = NeuronGroup(1, eq_RAII, threshold='v>vth', reset='v = c;u += d', method='euler')
RAII.v=c
i=0
j=0
k1 = 50
k2 = 100
k3 = 10
statemon_SA = StateMonitor(SA, 'v', record=True)
spikemon_SA = SpikeMonitor(SA)
statemon_RA = StateMonitor(RA, 'v', record=True)
spikemon_RA = SpikeMonitor(RA)
statemon_RAII = StateMonitor(RAII, 'v', record=True)
spikemon_RAII = SpikeMonitor(RAII)
store()
#%%
for sp in [5400]:
    for Ro in [6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
        for tr in range(10): 
            press = np.load('C:\\Users\\Behnam\\Desktop\\Temp_data\\Final_Filtered_Data\\pressure_filtered_Ro{}_tr{}_sp{}.npz'.format(Ro,tr,sp))
            # press = press.to_numpy(dtype=None, copy=False)

            # data_raw1 = np.load('C:\\Users\\Behnam\\Desktop\\Temp_data\\Final_Filtered_Data\\pressure_filtered_Ro{}_tr{}_sp{}.npz'.format(Ro,tr,sp))
            press = press['press']
            vib = np.load('C:\\Users\\Behnam\\Desktop\\Temp_data\\Final_Filtered_Data_Vibration\\vibration_fil_Ro{}_tr{}_sp{}.npy'.format(Ro,tr,sp))
            # vib = vib.reshape(-1,1)
            duration = (70*60/sp)

            # vib = preprocessing.normalize(vib)
            # vib = vib.reshape(-1,1)

            f2 = len(vib)/duration
            t2 = 1000/f2
            # press = preprocessing.normalize(press)
            f = len(press[0])/duration
            t = 1000/f
            press0 = press[0]
            Ra0 = np.zeros([])
            for i in range(len(press0)):
                v1 = press0[i]
                v2 = press0[i-1]
                VR = abs(v2 - v1)/t                
                Ra0 = np.append(Ra0,VR)
######################################  
            data0=Ra0
            press1 = press[1]
            Ra1 = np.zeros([])
            for i in range(len(press1)):
                v1 = press1[i]
                v2 = press1[i-1]
                VR = abs(v2 - v1)/t                
                Ra1 = np.append(Ra1,VR)
            # Ra1 = Ra1.reshape(1,-1)
######################################  
            data1=Ra1
            press2 = press[2]
            Ra2 = np.zeros([])
            for i in range(len(press2)):
                v1 = press2[i]
                v2 = press2[i-1]
                VR = abs(v2 - v1)/t                
                Ra2 = np.append(Ra2,VR)
            # Ra2 = Ra2.reshape(1,-1)
######################################  
            data2=Ra2
            press3 = press[3]
            Ra3 = np.zeros([])
            for i in range(len(press3)):
                v1 = press3[i]
                v2 = press3[i-1]
                VR = abs(v2 - v1)/t                
                Ra3 = np.append(Ra3,VR)
            # Ra3 = Ra3.reshape(1,-1)
######################################  
            data3=Ra3
            press4 = press[4]
            Ra4 = np.zeros([])
            for i in range(len(press4)):
                v1 = press4[i]
                v2 = press4[i-1]
                VR = abs(v2 - v1)/t                
                Ra4 = np.append(Ra4,VR)
            # Ra4 = Ra4.reshape(1,-1)
######################################  
            data4=Ra4
            press5 = press[5]
            Ra5 = np.zeros([])
            for i in range(len(press5)):
                v1 = press5[i]
                v2 = press5[i-1]
                VR = abs(v2 - v1)/t                
                Ra5 = np.append(Ra5,VR)
            # Ra5 = Ra5.reshape(1,-1)
######################################  
            data5=Ra5
            press6 = press[6]
            Ra6 = np.zeros([])
            for i in range(len(press6)):
                v1 = press6[i]
                v2 = press6[i-1]
                VR = abs(v2 - v1)/t                
                Ra6 = np.append(Ra6,VR)
            # Ra6 = Ra6.reshape(1,-1)
######################################  
            data6=Ra6
            press7 = press[7]
            Ra7 = np.zeros([])
            for i in range(len(press7)):
                v1 = press7[i]
                v2 = press7[i-1]
                VR = abs(v2 - v1)/t                
                Ra7 = np.append(Ra7,VR)
            # Ra7 = Ra7.reshape(1,-1)
######################################  
            data7=Ra7
            press8 = press[8]
            Ra8 = np.zeros([])
            for i in range(len(press8)):
                v1 = press8[i]
                v2 = press8[i-1]
                VR = abs(v2 - v1)/t                
                Ra8 = np.append(Ra8,VR)
            # Ra8 = Ra8.reshape(1,-1)
######################################  
            data8=Ra8            
            press9 = press[9]
            Ra9 = np.zeros([])
            for i in range(len(press9)):
                v1 = press9[i]
                v2 = press9[i-1]
                VR = abs(v2 - v1)/t                
                Ra9 = np.append(Ra9,VR)
            # Ra9 = Ra9.reshape(1,-1)
######################################  
            data9=Ra9
            press10 = press[10]
            Ra10 = np.zeros([])
            for i in range(len(press10)):
                v1 = press10[i]
                v2 = press10[i-1]
                VR = abs(v2 - v1)/t                
                Ra10 = np.append(Ra10,VR)
            # Ra10 = Ra10.reshape(1,-1)
######################################  
            data10=Ra10
            press11 = press[11]
            Ra11 = np.zeros([])
            for i in range(len(press11)):
                v1 = press11[i]
                v2 = press11[i-1]
                VR = abs(v2 - v1)/t                
                Ra11 = np.append(Ra11,VR)
            # Ra11 = Ra11.reshape(1,-1)
######################################  
            data11=Ra11   
            press12 = press[12]
            Ra12 = np.zeros([])
            for i in range(len(press12)):
                v1 = press12[i]
                v2 = press12[i-1]
                VR = abs(v2 - v1)/t                
                Ra12 = np.append(Ra12,VR)
   ###################################### 
            data12=Ra12
            press13 = press[13]
            Ra13 = np.zeros([])
            for i in range(len(press13)):
                v1 = press13[i]
                v2 = press13[i-1]
                VR = abs(v2 - v1)/t                
                Ra13 = np.append(Ra13,VR)
   ###################################### 
            data13=Ra13
            press14 = press[14]
            Ra14 = np.zeros([])
            for i in range(len(press14)):
                v1 = press14[i]
                v2 = press14[i-1]
                VR = abs(v2 - v1)/t                
                Ra14 = np.append(Ra14,VR)
   ###################################### 
            data14=Ra14
            press15 = press[15]
            Ra15 = np.zeros([])
            for i in range(len(press15)):
                v1 = press15[i]
                v2 = press15[i-1]
                VR = abs(v2 - v1)/t                
                Ra15 = np.append(Ra15,VR)
   ###################################### 
            data15=Ra15            
            press16 = press[16]
            Ra16 = np.zeros([])
            for i in range(len(press16)):
                v1 = press16[i]
                v2 = press16[i-1]
                VR = abs(v2 - v1)/t                
                Ra16 = np.append(Ra16,VR)
   ###################################### 
            data16=Ra16            
            press17 = press[17]
            Ra17 = np.zeros([])
            for i in range(len(press17)):
                v1 = press17[i]
                v2 = press17[i-1]
                VR = abs(v2 - v1)/t                
                Ra17 = np.append(Ra17,VR)
   ###################################### 
            data17=Ra17            
            
            press18 = press[18]
            Ra18 = np.zeros([])
            for i in range(len(press18)):
                v1 = press18[i]
                v2 = press18[i-1]
                VR = abs(v2 - v1)/t                
                Ra18 = np.append(Ra18,VR)
   ###################################### 
            data18=Ra18            
            
            press19 = press[19]
            Ra19 = np.zeros([])
            for i in range(len(press19)):
                v1 = press19[i]
                v2 = press19[i-1]
                VR = abs(v2 - v1)/t                
                Ra19 = np.append(Ra19,VR)
   ###################################### 
            data19=Ra19            
            press20 = press[20]
            Ra20 = np.zeros([])
            for i in range(len(press20)):
                v1 = press20[i]
                v2 = press20[i-1]
                VR = abs(v2 - v1)/t                
                Ra20 = np.append(Ra20,VR)
   ###################################### 
            data20=Ra20            
            press21 = press[21]
            Ra21 = np.zeros([])
            for i in range(len(press21)):
                v1 = press21[i]
                v2 = press21[i-1]
                VR = abs(v2 - v1)/t                
                Ra21 = np.append(Ra21,VR)
   ###################################### 
            data21=Ra21            
            press22 = press[22]
            Ra22 = np.zeros([])
            for i in range(len(press22)):
                v1 = press22[i]
                v2 = press22[i-1]
                VR = abs(v2 - v1)/t                
                Ra22 = np.append(Ra22,VR)
   ###################################### 
            data22=Ra22
            press23 = press[23]

            Ra23 = np.zeros([])
            for i in range(len(press23)):
                v1 = press23[i]
                v2 = press23[i-1]
                VR = abs(v2 - v1)/t                
                Ra23 = np.append(Ra23,VR)
   ###################################### 
            data23=Ra23
            press24 = press[24]

            Ra24 = np.zeros([])
            for i in range(len(press24)):
                v1 = press24[i]
                v2 = press24[i-1]
                VR = abs(v2 - v1)/t                
                Ra24 = np.append(Ra24,VR)
######################################  
            data24=Ra24
            press25 = press[25]

            Ra25 = np.zeros([])
            for i in range(len(press25)):
                v1 = press25[i]
                v2 = press25[i-1]
                VR = abs(v2 - v1)/t                
                Ra25 = np.append(Ra25,VR)
######################################  
            data25=Ra25
            press26 = press[26]

            Ra26 = np.zeros([])
            for i in range(len(press26)):
                v1 = press26[i]
                v2 = press26[i-1]
                VR = abs(v2 - v1)/t                
                Ra26 = np.append(Ra26,VR)
######################################  
            data26=Ra26
            press27 = press[27]
            
            Ra27 = np.zeros([])
            for i in range(len(press27)):
                v1 = press27[i]
                v2 = press27[i-1]
                VR = abs(v2 - v1)/t                
                Ra27 = np.append(Ra27,VR)
######################################  
            data27=Ra27
            press28 = press[28]

            Ra28 = np.zeros([])
            for i in range(len(press28)):
                v1 = press28[i]
                v2 = press28[i-1]
                VR = abs(v2 - v1)/t                
                Ra28 = np.append(Ra28,VR)
######################################  
            data28=Ra28
            press29 = press[29]

            Ra29 = np.zeros([])
            for i in range(len(press29)):
                v1 = press29[i]
                v2 = press29[i-1]
                VR = abs(v2 - v1)/t                
                Ra29 = np.append(Ra29,VR)
######################################  
            data29=Ra29
            press30 = press[30]

            Ra30 = np.zeros([])
            for i in range(len(press30)):
                v1 = press30[i]
                v2 = press30[i-1]
                VR = abs(v2 - v1)/t                
                Ra30 = np.append(Ra30,VR)
######################################  
            data30=Ra30
            press31 = press[31]
            Ra31 = np.zeros([])
            for i in range(len(press31)):
                v1 = press31[i]
                v2 = press31[i-1]
                VR = abs(v2 - v1)/t                
                Ra31 = np.append(Ra31,VR)
######################################  
            data31=Ra31
            press32 = press[32]
            Ra32 = np.zeros([])
            for i in range(len(press32)):
                v1 = press32[i]
                v2 = press32[i-1]
                VR = abs(v2 - v1)/t                
                Ra32 = np.append(Ra32,VR)
######################################  
            data32=Ra32
            press33 = press[33]
            Ra33 = np.zeros([])
            for i in range(len(press33)):
                v1 = press33[i]
                v2 = press33[i-1]
                VR = abs(v2 - v1)/t                
                Ra33 = np.append(Ra33,VR)
######################################  
            data33=Ra33
            press34 = press[34]
            Ra34 = np.zeros([])
            for i in range(len(press34)):
                v1 = press34[i]
                v2 = press34[i-1]
                VR = abs(v2 - v1)/t                
                Ra34 = np.append(Ra34,VR)
######################################  
            data34=Ra34
            press35 = press[35]
            Ra35 = np.zeros([])
            for i in range(len(press35)):
                v1 = press35[i]
                v2 = press35[i-1]
                VR = abs(v2 - v1)/t                
                Ra35 = np.append(Ra35,VR)
######################################  
            data35=Ra35
            press36 = press[36]
            Ra36 = np.zeros([])
            for i in range(len(press36)):
                v1 = press36[i]
                v2 = press36[i-1]
                VR = abs(v2 - v1)/t                
                Ra36 = np.append(Ra36,VR)
######################################  
            data36=Ra36
            press37 = press[37]
            Ra37 = np.zeros([])
            for i in range(len(press37)):
                v1 = press37[i]
                v2 = press37[i-1]
                VR = abs(v2 - v1)/t                
                Ra37 = np.append(Ra37,VR)
######################################  
            data37=Ra37
            press38 = press[38]
            Ra38 = np.zeros([])
            for i in range(len(press38)):
                v1 = press38[i]
                v2 = press38[i-1]
                VR = abs(v2 - v1)/t                
                Ra38 = np.append(Ra38,VR)
######################################  
            data38=Ra38
            press39 = press[39]
            Ra39 = np.zeros([])
            for i in range(len(press39)):
                v1 = press39[i]
                v2 = press39[i-1]
                VR = abs(v2 - v1)/t                
                Ra39 = np.append(Ra39,VR)
######################################  
            data39=Ra39
            press40 = press[40]
            Ra40 = np.zeros([])
            for i in range(len(press40)):
                v1 = press40[i]
                v2 = press40[i-1]
                VR = abs(v2 - v1)/t                
                Ra40 = np.append(Ra40,VR)
######################################  
            data40=Ra40
            press41 = press[41]
            Ra41 = np.zeros([])
            for i in range(len(press41)):
                v1 = press41[i]
                v2 = press41[i-1]
                VR = abs(v2 - v1)/t                
                Ra41 = np.append(Ra41,VR)
######################################  
            data41=Ra41
            press42 = press[42]
            Ra42 = np.zeros([])
            for i in range(len(press42)):
                v1 = press42[i]
                v2 = press42[i-1]
                VR = abs(v2 - v1)/t                
                Ra42 = np.append(Ra42,VR)
######################################  
            data42=Ra42

########################################### RA-II ######################################################
            Ra43 = np.zeros([])
            RaII43 = np.zeros([])
            for i in range(len(vib)):
                v1 = vib[i]
                v2 = vib[i-1]
                VR1 = abs(v2 - v1)/t2
                Ra43 = np.append(Ra43,VR1)
            for i in range(len(Ra43)):
                v1 = Ra43[i]
                v2 = Ra43[i-1]
                VR2 = abs(v2 - v1)/t2 
                RaII43 = np.append(RaII43,VR2)
   ###################################### 
            data43=RaII43   
            
########################################### SA-I ######################################################

            press44=press[43]
            press45=press[44]
            press46=press[45]
            press47=press[46]
            press48=press[47]
            press49=press[48]
            press50=press[49]
            press51=press[50]
            press52=press[51]
            press53=press[52]
            press54=press[53]
            press55=press[54]
            press56=press[55]
            press57=press[56]
            press58=press[57]
            press59=press[58]
            press60=press[59]
            press61=press[60]
            press62=press[61]
            press63=press[62]                       
            press64=press[63]                       
            
            data0=reshape(Ra0,(1,-1))
            data1=reshape(Ra1,(1,-1))
            data2=reshape(Ra2,(1,-1))
            data3=reshape(Ra3,(1,-1))
            data4=reshape(Ra4,(1,-1))
            data5=reshape(Ra5,(1,-1))
            data6=reshape(Ra6,(1,-1))
            data7=reshape(Ra7,(1,-1))
            data8=reshape(Ra8,(1,-1))
            data9=reshape(Ra9,(1,-1))
            data10=reshape(Ra10,(1,-1))
            data11=reshape(Ra11,(1,-1))
            data12=reshape(Ra12,(1,-1))
            data13=reshape(Ra13,(1,-1))
            data14=reshape(Ra14,(1,-1))
            data15=reshape(Ra15,(1,-1))
            data16=reshape(Ra16,(1,-1))
            data17=reshape(Ra17,(1,-1))
            data18=reshape(Ra18,(1,-1))
            data19=reshape(Ra19,(1,-1))
            data20=reshape(Ra20,(1,-1))
            data21=reshape(Ra21,(1,-1))
            data22=reshape(Ra22,(1,-1))
            data23=reshape(Ra23,(1,-1))
            data24=reshape(Ra24,(1,-1))
            data25=reshape(Ra25,(1,-1))
            data26=reshape(Ra26,(1,-1))
            data27=reshape(Ra27,(1,-1))
            data28=reshape(Ra28,(1,-1))
            data29=reshape(Ra29,(1,-1))
            data30=reshape(Ra30,(1,-1))
            data31=reshape(Ra31,(1,-1))
            data32=reshape(Ra32,(1,-1))
            data33=reshape(Ra33,(1,-1))
            data34=reshape(Ra34,(1,-1))
            data35=reshape(Ra35,(1,-1))
            data36=reshape(Ra36,(1,-1))
            data37=reshape(Ra37,(1,-1))
            data38=reshape(Ra38,(1,-1))
            data39=reshape(Ra39,(1,-1))
            data40=reshape(Ra40,(1,-1))
            data41=reshape(Ra41,(1,-1))
            data42=reshape(Ra42,(1,-1))
            data43=reshape(press44,(1,-1))
            data44=reshape(press45,(1,-1))
            data45=reshape(press46,(1,-1))
            data46=reshape(press47,(1,-1))
            data47=reshape(press48,(1,-1))
            data48=reshape(press49,(1,-1))
            data49=reshape(press50,(1,-1))
            data50=reshape(press51,(1,-1))
            data51=reshape(press52,(1,-1))
            data52=reshape(press53,(1,-1))
            data53=reshape(press54,(1,-1))
            data54=reshape(press55,(1,-1))
            data55=reshape(press56,(1,-1))
            data56=reshape(press57,(1,-1))
            data57=reshape(press58,(1,-1))
            data58=reshape(press59,(1,-1))
            data59=reshape(press60,(1,-1))
            data60=reshape(press61,(1,-1))
            data61=reshape(press62,(1,-1))
            data62=reshape(press63,(1,-1))
            data63=reshape(press64,(1,-1))
            data64=reshape(RaII43,(1,-1))
            I_SA_data = np.vstack((data43,data44,data45,data46,data47,data48,data49,data50,data51,data52,data53,data54,data55,data56,data57,data58,data59,data60,data61,data62,data63))
            I_RA_data = np.vstack((data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,data29,data30,data31,data32,data33,data34,data35,data36,data37,data38,data39,data40,data41,data42))
            I_RAII_data = data64

            I_SA = TimedArray(np.transpose(I_SA_data), dt = t * ms)
            I_RA = TimedArray(np.transpose(I_RA_data), dt = t * ms)
            I_RAII = TimedArray(np.transpose(I_RAII_data), dt=t2 * ms)

            restore()
    
            run(duration*1000*ms)
            # get_ipython().run_line_magic('matplotlib', 'qt')

            # plot(statemon.t/ms, statemon.v[3])
            # plot(spikemon_SA.t/ms, spikemon_SA.i, '.k')
            # plot(spikemon_RA.t/ms, spikemon_RA.i, '.b')
            # plot(spikemon_RAII.t/ms, spikemon_RAII.i, '.r')
            
            print(j,'Done!')
            # plot(duration, spikemon.count / duration)
            j+=1
            spikeCount_SA = spikemon_SA.i
            spikeCount_SA = np.array(spikeCount_SA)
            spikeCount_df_SA = pd.DataFrame(spikeCount_SA, columns = ['Spike_INC_SA'])
            # print('SA = ', spikeCount_SA/duration)
            spikeTime_SA = spikemon_SA.t/ms
            spikeTime_SA = np.array(spikeTime_SA)
            spikeTime_df_SA = pd.DataFrame(spikeTime_SA, columns=['Spike_Time_SA'])
            
            firingRate_SA = spikemon_SA.count/duration
            firingRate_SA = np.array(firingRate_SA)
            firingRate_df_SA = pd.DataFrame(firingRate_SA, columns=['Firing_Rate_SA'])
            
         
            spikeCount_RA = spikemon_RA.i
            spikeCount_RA = np.array(spikeCount_RA)
            spikeCount_df_RA = pd.DataFrame(spikeCount_RA, columns = ['Spike_INC_RA'])
            
            spikeTime_RA = spikemon_RA.t/ms
            spikeTime_RA = np.array(spikeTime_RA)
            spikeTime_df_RA = pd.DataFrame(spikeTime_RA, columns=['Spike_Time_RA'])
            
            firingRate_RA = spikemon_RA.count/duration
            firingRate_RA = np.array(firingRate_RA)
            firingRate_df_RA = pd.DataFrame(firingRate_RA, columns=['Firing_Rate_RA'])
            # print('RA-I = ', spikemon_RA.count/duration)

            
            spikeCount_RAII = spikemon_RAII.i
            spikeCount_RAII = np.array(spikeCount_RAII)
            spikeCount_df_RAII = pd.DataFrame(spikeCount_RAII, columns = ['Spike_INC_RAII'])
            
            spikeTime_RAII = spikemon_RAII.t/ms
            spikeTime_RAII = np.array(spikeTime_RAII)
            spikeTime_df_RAII = pd.DataFrame(spikeTime_RAII, columns=['Spike_Time_RAII'])
            
            firingRate_RAII = spikemon_RAII.count/duration
            firingRate_RAII = np.array(firingRate_RAII)
            firingRate_df_RAII = pd.DataFrame(firingRate_RAII, columns=['Firing_Rate_RAII'])
            
            # print('RA-II = ', spikemon_RAII.count/duration)
        
            
            
            brian = pd.concat([spikeTime_df_SA, spikeCount_df_SA, firingRate_df_SA, spikeTime_df_RA, spikeCount_df_RA, firingRate_df_RA, spikeTime_df_RAII, spikeCount_df_RAII, firingRate_df_RAII], axis=1)            
            brian.to_csv('C:\\Users\\Behnam\\Desktop\\Temp_data\\Brian\\Final_All three\\64ch\\5400\\Brian_64ch_SAI_RAI_RAII_Ro{}_tr{}_sp{}.csv'.format(Ro,tr,sp))
#%%
get_ipython().run_line_magic('matplotlib', 'qt')
# plot(statemon.t/ms, statemon.v[3])
plt.rcParams['font.size'] = '10'
subplot(3,1,1)
xlabel('Time (ms)')
ylabel('Neuron index')
# xlim(0,duration*1000)
plot(spikemon_SA.t/ms, spikemon_SA.i, '.k')
subplot(3,1,2)
xlabel('Time (ms)')
ylabel('Neuron index')
# xlim(0,duration*1000)
plot(spikemon_RA.t/ms, spikemon_RA.i, '.b')
subplot(3,1,3)
xlabel('Time (ms)')
ylabel('Neuron index')
# xlim(0,duration*1000)
ylim(-0.8,)
plot(spikemon_RAII.t/ms, spikemon_RAII.i, '.r')
