# signal_generator_802_11n class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 31.10.2017 12:02
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time

from refptr import *
from thesdk import *
from rtl import *
import modem as mdm #Function definitions

#Simple buffer template

ofdm64dict_noguardband={ 'framelen':64,'data_loc': np.r_[1:11+1, 13:25+1, 
    27:39+1, 41:53+1, 55:64+1]-1, 'pilot_loc' : np.r_[-21, -7, 7, 21] + 32, 'CPlen':16}

ofdm64dict_withguardband={ 'framelen':64,'data_loc': np.r_[7:11+1, 13:25+1, 
    27:32+1, 34:39+1, 41:53+1, 55:59+1]-1, 'pilot_loc' : np.r_[-21, -7, 7, 21] + 32, 'CPlen':16}

#constant pilot sequences from the standard
PLPCsyn_short=np.sqrt(16/6)*np.r_[0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0]

PLPCsyn_long=np.r_[1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1]

bbsigdict_ofdm_sinusoid3={ 'mode':'ofdm_sinusoid', 'freqs':[1.0e6 , 13e6, 17e6 ], 'length':2**14, 'BBRs':20e6 };

#-----Data signals
bbsigdict_randombitstream_QAM4_OFDM={ 'mode':'ofdm_random_qam', 'QAM':4, 'length':2**14, 'BBRs': 20e6 };


class signal_generator_802_11n(rtl,thesdk):

    #802.11 OFDM Dictionaries define the structure of the OFDM frames
    def __init__(self,*arg): 
        self.proplist = [ 'Rs', 'bbsigdict', 'Txantennas', 'Users'];    #properties that can be propagated from parent
        self.ofdmdict =ofdm64dict_withguardband
        self.bbsigdict=bbsigdict_randombitstream_QAM4_OFDM
        self.Rs = self.bbsigdict['BBRs']         # Default system sampling frequency
        self.Users=2
        self.Txantennas=4
        self.iptr_A = refptr();
        self.model='py';                         #can be set externally, but is not propagated
        self._Z = refptr();
        self._classfile=__file__

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
    def init(self):
        pass

    def ofdm_sinusoid(self):
        ofdmdict=self.ofdmdict

        bbsigdict=self.bbsigdict
        framelen=ofdmdict['framelen']
        length=bbsigdict['length']
        frames=np.floor(length/framelen)

        CPlen=ofdmdict['CPlen']
        #QAM=bbsigdict['QAM']
        BBRs=bbsigdict['BBRs']
        signalindexes=np.round(np.array(self.bbsigdict['freqs'])/self.Rs*framelen).astype(int)
        frame=np.zeros((frames,framelen))
        frame[:,signalindexes]=1
        datasymbols=frame[:,ofdmdict['data_loc']]
        pilotsymbols=frame[:,ofdmdict['pilot_loc']]
        interpolated=mdm.ofdmMod(ofdmdict,datasymbols,pilotsymbols)
        interpolated=interpolated.reshape((-1,(framelen+CPlen)))
        interpolated=self.interpolate_at_ofdm({'signal':interpolated})
        length=interpolated.shape[1]
        duration=(framelen+CPlen)*self.Rs/BBRs
        overlap=(length-duration) #Now this is even, but necessarily. Handle later
        win=window({'Tr':100e-9, 'length':length, 'fs':self.Rs, 'duration': duration})
        interpolated=interpolated*win
        chained=interpolated[0,:]
        for k in range(1,interpolated.shape[0]):
            #print(np.zeros((int(length-overlap))))
            a=np.r_[chained, np.zeros((int(length-overlap)))]
            b=np.r_[np.zeros((int(chained.shape[0]-overlap))), interpolated[k,:] ]
            a.shape=(1,-1)
            b.shape=(1,-1)
            chained=np.sum(np.r_['0',a,b],axis=0)

        chained.shape=(1,-1)
        usersig=np.ones((self.Txantennas,1))*chained.T
        
        out=np.zeros((self.Users,usersig.shape[0],usersig.shape[1]),dtype='complex')
        for i in range(self.Users):
            out[i,:,:]=usersig

        self._Z.Value=out 
    
    def ofdm_random_qam(self):
            #Local vars just to clear to code
            ofdmdict=self.ofdmdict
            bbsigdict=self.bbsigdict
            framelen=ofdmdict['framelen']
            length=bbsigdict['length']
            CPlen=ofdmdict['CPlen']
            QAM=bbsigdict['QAM']
            BBRs=bbsigdict['BBRs']
            #The length is approx this many frames
            frames=np.floor(length/(framelen+CPlen))
            bitspersymbol=np.log2(QAM).astype(int)
            
            #Generate random bitstreams per user
            #bitstream(user,time,antenna)
            bitstream=np.random.randint(2,size=(self.Users,int(frames*bitspersymbol*framelen)))

            #Init the qam signal, frame and out
            #qamsignal is different for the users, but initially identical for the TXantennas 
            qamsignal=np.zeros((self.Users,int(frames*framelen)),dtype='complex')
            frame=np.zeros((int(frames),int(framelen)),dtype='complex')

            #for i in range(self.Txantennas):
            for i in range(self.Users):
                wordstream, qamsignal[i]= mdm.qamModulateBitStream(bitstream[i], QAM) #Modulated signal per user
                qamsignal[i]=qamsignal[i].reshape((1,qamsignal.shape[1]))
                frame= qamsignal[i].reshape((-1,framelen)) #The OFDM frames
                datasymbols=frame[:,ofdmdict['data_loc']]   #Set the data
                pilotsymbols=frame[:,ofdmdict['pilot_loc']] #In this case, also pilot carry bits
                if (factor({'n':self.Rs/BBRs}))[0] != 2:
                    print("SHIT ALERT: %s: The first interpolation factor for 802.11n is more than 2 \n and I'am too lazy to implement dedicated filter mask" %(self.__class__.__name__))
                    quit()

                interpolated=mdm.ofdmMod(ofdmdict,datasymbols,pilotsymbols) #Variable for interpolation
                interpolated=interpolated.reshape((-1,(framelen+CPlen)))
                interpolated=self.interpolate_at_ofdm({'signal':interpolated})
                length=interpolated.shape[1]
                duration=(framelen+CPlen)*self.Rs/BBRs
                overlap=(length-duration) #Now this is even, but not necessarily. Handle later
                win=window({'Tr':100e-9, 'length':length, 'fs':self.Rs, 'duration': duration})
                interpolated=interpolated*win

                #Initialize chaining of the symbols
                chained=np.array(interpolated[0,:],ndmin=2)
                #print(chained.shape)

                #Loop through all symbols on rows
                for k in range(1,interpolated.shape[0]):
                    #print(np.zeros((int(length-overlap))))
                    a=np.r_['1', chained, np.zeros((1,int(length-overlap)))]
                    b=np.r_['1',np.zeros((1,int(chained.shape[1]-overlap))), np.reshape(interpolated[k,:],(1,-1)) ]
                    a.shape=(1,-1)
                    b.shape=(1,-1)
                    chained=np.sum(np.r_['0',a,b],axis=0)
                    chained.shape=(1,-1)
                    #OFDM symbols chained
                #If we are handling the first user, we sill initialize the usersig
                if i==0:
                    usersig=np.zeros((self.Users, chained.shape[1], self.Txantennas),dtype='complex')
                    usersig[i,:,:]=np.transpose(np.ones((self.Txantennas,1))@chained)
                else:
                    usersig[i,:,:]=np.transpose(np.ones((self.Txantennas,1))@chained)
                
            self._Z.Value=usersig 

    def interpolate_at_ofdm(self,argdict={'signal':[]}):
        ratio=self.Rs/self.bbsigdict['BBRs']
        signal=argdict['signal']
        #Currently fixeed interpolation. check the fucntion definitions for details
        factors=factor({'n':ratio})
        filterlist=self.generate_interpolation_filterlist({'interp_factor':ratio})
        msg="Signal length is now %i" %(signal.shape[1])
        self.print_log({'type':'I', 'msg': msg}) 
        #This is to enable growth of the signal length that better mimics the hardware
        #sig.resample_poly is more effective, but does not allow growth.
        for symbol in range(signal.shape[0]):
            t=signal[symbol,:]
            for i in range(factors.shape[0]):
                #signali=sig.resample_poly(signal[user,:,antenna], fact, 1, window=i)
                t2=np.zeros(int(t.shape[0]*factors[i]),dtype='complex')
                t2[0::int(factors[i])]=t
                t=sig.convolve(t2, filterlist[i],mode='full')
            if symbol==0:
                signali=np.zeros((signal.shape[0],t.shape[0]),dtype='complex')
                signali[symbol,:]=t
            else:
                signali[symbol,:]=t
        print("Signal length is now %i" %(signali.shape[1]))
        self._filterlist=filterlist
        #print(filterlist)
        return signali

    def generate_interpolation_filterlist(self,argdict={'interp_factor':1}):
        #Use argument dictionary. Makes modifications easier.
        interp_factor=argdict['interp_factor']
        
        attenuation=70 #Desired attenuation in decibels
        factors=factor({'n':interp_factor})
        #print(factors)
        fsample=1
        BW=0.45
        numtaps=65     # TAps for the first filterThis should be somehow verified
        #Harris rule. This is to control stability of Rmez
        #numtaps= int(np.ceil(attenuation*fsample*factors[0]/(fsample/2-BW)))    # TAps for the first filterThis should be somehow verified
        desired=np.array([ 1, 10**(-attenuation/10)] )
        #check the mask specs from standard 
        #mask=np.array([ 1, 10**(-28/10)    ] )
        filterlist=list()
        if interp_factor >1:
            for i in factors:
                fact=i
                #print(fsample)
                if  fsample/(0.5)<= 8: #FIR is needed
                    msg= "BW to sample rate ratio is now %s" %(fsample/0.5)
                    self.print_log({'type': 'I', 'msg':msg })
                    msg="Interpolation by %i" %(fact)
                    self.print_log({'type': 'I', 'msg':msg })
                    bands=np.array([0, BW, (fsample*fact/2-BW), fact*fsample/2])
                    filterlist.append(sig.remez(numtaps, bands, desired, Hz=fact*fsample))
                    fsample=fsample*fact #increase the sample frequency
                    numtaps=np.amax([3, int(np.floor(numtaps/fact)) + int((np.floor(numtaps/fact)%2-1))]) 
                else:
                    print("BW to sample rate ratio is now %s" %(fsample/0.5))
                    fact=fnc.reduce(lambda x,y:x*y,factors)/fsample
                    print("Interpolation with 3-stage CIC-filter by %i" %(fact))
                    fircoeffs=np.ones(int(fact))/(fact) #do the rest of the interpolation with 3-stage CIC-filter
                    fircoeffs=fnc.reduce(lambda x,y: np.convolve(x,y),list([fircoeffs, fircoeffs, fircoeffs]))
                    filterlist.append(fircoeffs)
                    #print(filterlist)
                    fsample=fsample*fact #increase the sample frequency
                    print("BW to sample rate ratio is now %s" %(fsample/0.5))
                    break
        else:
            print("Interpolation ratio is 1. Generated unit coefficient")
            filterlist.append([1.0]) #Ensure correct operation in unexpected situations.
        return filterlist

    #def generate_802_11n_plpc(self):
    #    ofdmdict=self.ofdmdict
    #    framelen=ofdmdict['framelen']
    #    CPlen=ofdmdict['CPlen']
    #    QAMshort=4
    #    QAMlong=2
    #    BBRs=bbsigdict['BBRs']
    #    #The length is approx this many frames
    #    
    #    #Generate random bitstreams per user
    #    #bitstream(user,time,antenna)
    #    #sequence_short=[] 
    #    #for in in range(10)
    #    symbol_short=np.r[np.zeros((1,6)), PLPCsyn_short, np.zeros((1,6)))] #These are already 4-QAM modulated
    #    seq_short=np.ones((10,1))@symbol_short #10 short symbols

    #    symbol_long=np.r[np.zeros((1,6)), PLPCsyn_long, np.zeros((1,6)))]   #These are 2-QAM modulated
    #    seq_long=np.ones((2,1))@symbol_long                                 #2 long symbols
    #    
    #    #Of course windowing consists anomalies.Performed after IFFT anyway
    #    symbols=np.r('0',seq_short,seq_long)
    #    datasymbols=symbols[:,ofdmdict['data_loc']]   #Set the data
    #    pilotsymbols=symbols[:,ofdmdict['pilot_loc']] #Set the pilots
    #    modulated=mdm.ofdmMod(ofdmdict,datasymbols,pilotsymbols) #Variable for interpolation


    #    #Init the qam signal, frame and out
    #    #qamsignal is different for the users, but initially identical for the TXantennas 
    #    frame=np.zeros((int(frames),int(framelen)),dtype='complex')

#Window to taper OFDM symbols
def window(argdict={'Tr':100e-9, 'length':478, 'fs':80e6, 'duration':240 }):
    #Tr is the window rise/fall time 100ns is from the specification
    Tr=argdict['Tr']
    fs=argdict['fs']
    length=argdict['length']
    duration=argdict['duration']
    T=argdict['duration']/fs
    prefix=(length-duration)/2
    t=(np.arange(length)-prefix)/fs # Time: T is the duration of the symbol without tails
    window=np.zeros(length)
    for i in range(len(t)):
     if t[i]>= -Tr/2 and t[i]<=Tr/2:
         window[i]=np.sin(np.pi/2*(0.5+t[i]/Tr))**2
     elif t[i]>Tr/2 and t[i]<T-Tr/2:
         window[i]=1
     elif t[i]>= T-Tr/2 and t[i] <= T+Tr/2:
         window[i]=np.sin(np.pi/2*(0.5-(t[i]-T)/Tr))**2
    return window

    def run(self,*arg):
        if len(arg)>0:
            par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            par=False

        if self.model=='py':
            out=np.array(self.iptr_A.Value)
            if par:
                queue.put(out)
            self._Z.Value=out

def factor(argdict={'n':1}):
    #This is a function to calculate factors of an integer as in Matlab
    # "Everything in Matlab is available in Python" =False.
    reminder=argdict['n']
    factors=np.array([])
    while reminder >1:
        notfound=True
        for i in range(2,int(np.sqrt(reminder)+2)):
            if reminder % i == 0:
                factors=np.r_[ factors, i]
                reminder=reminder/i
                notfound=False
                break
        if notfound==True:
                factors=np.r_[ factors, reminder]
                reminder=0
    return factors

