# signal_generator_802_11n class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 03.11.2017 10:56
import sys
sys.path.append ('/home/projects/fader/TheSDK/Entities/refptr/py')
sys.path.append ('/home/projects/fader/TheSDK/Entities/thesdk/py')
sys.path.append ('/home/projects/fader/TheSDK/Entities/modem/py')
sys.path.append ('/home/projects/fader/TheSDK/Entities/multirate/py')
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time

from refptr import *
from thesdk import *

import modem as mdm #Function definitions
import multirate as mr

#+1 added because the ranges do not include the endpoint
ofdm64dict_withguardband={ 'framelen':64,'data_loc': np.r_[-26:-22+1, -20:-8+1, 
    -6:-1+1, 1:6+1, 8:20+1, 22:26+1 ], 'pilot_loc' : np.r_[-21, -7, 7, 21], 'CPlen':16}

ofdm64dict_noguardband={ 'framelen':64,'data_loc': np.r_[-32:-22+1, -20:-8+1, 
    -6:-1+1, 1:6+1, 8:20+1, 22:31+1 ], 'pilot_loc' : np.r_[-21, -7, 7, 21], 'CPlen':16}

#constant pilot sequences from the standard
#Standard presentation: neq freq=0  #-32               -24        -20         -16         -12         -8         -4                  4           8         12         16          20         24              
PLPCsyn_short=np.sqrt(13/6)*np.array([0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,0,0,0,0],ndmin=2).T

#Standard presentation-32              -24      -20       -16     -12       -8        -4                 4         8         12         16        20        24              
PLPCsyn_long=np.array([0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0],ndmin=2).T
#Convert to "Math" IFFT/FTT format: to_fft=symbol_in_802_standard_format[Freqmap], maps the negative frequencies to the upper half

#These are identical, lower works only in python
#Freqmap=np.r_[np.arange(32,64), np.arange(0,32)] #Maps negative frequencies to the end of the array
Freqmap=np.r_[np.arange(-32,0), np.arange(0,32)] #Maps negative frequencies to the end of the array


bbsigdict_ofdm_sinusoid3={ 'mode':'ofdm_sinusoid', 'freqs':[1.0e6 , 13e6, 17e6 ], 'length':2**14, 'BBRs':20e6 };

#-----Data signals
bbsigdict_randombitstream_QAM4_OFDM={ 'mode':'ofdm_random_qam', 'QAM':4, 'length':2**14, 'BBRs': 20e6 };


class signal_generator_802_11n(thesdk):

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
        self.DEBUG= True

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
    def init(self):
        pass

    def ofdm_sinusoid(self):
        ofdmdict=ofdm64dict_noguardband
        bbsigdict=self.bbsigdict
        framelen=ofdmdict['framelen']
        length=bbsigdict['length']
        frames=np.floor(length/framelen)

        CPlen=ofdmdict['CPlen']
        #QAM=bbsigdict['QAM']
        BBRs=bbsigdict['BBRs']
        
        #This defines locations as in 802 standard index 0 equals frequency -32=-fs/2
        signalfrequencies=np.round(np.array(self.bbsigdict['freqs'])/(BBRs/2)*framelen/2).astype(int)
        print(signalfrequencies)
        frame=np.zeros((frames,framelen))
        frame[:,signalfrequencies+32]=1
        print(frame[0,:])
        #Need to map onece in order to get remapped right during modulation
        datastream=frame[:,ofdmdict['data_loc']+32]
        pilotstream=frame[:,ofdmdict['pilot_loc']+32]
        
        print(datastream[0,:])
        print(pilotstream[0,:])

        #Modulator takes care of mapping the negative frequencises to the end of the array
        interpolated=mdm.ofdmMod(ofdmdict,datastream,pilotstream)
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
                dataframe=frame[:,ofdmdict['data_loc']+32]   #Set the data
                pilotframe=frame[:,ofdmdict['pilot_loc']+32] #In this case, also pilot carry bits
                if (mr.factor({'n':self.Rs/BBRs}))[0] != 2:
                    print("SHIT ALERT: %s: The first interpolation factor for 802.11n is more than 2 \n and I'am too lazy to implement dedicated filter mask" %(self.__class__.__name__))
                    quit()

                interpolated=mdm.ofdmMod(ofdmdict,dataframe,pilotframe) #Variable for interpolation
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
    
    def gen_plpc_preamble_field(self):
        #Need to generate the IFFT according to stardard
        #t=np.arange(64)
        #t.shape=(-1,1)
        #k=np.arange(-32,32)
        #k.shape=(-1,1)
        #IFFTmat=1/64*np.exp(2j*np.pi*t@k.T/64)
        #seq_short=IFFTmat@PLPCsyn_short
        
        #Equivalent method using the zero  indexing FFT
        #Python sucks. Be careful with dimensions
        #shift=-32  #Negative frequency shift by Fs/2
        #fshift=np.r_[PLPCsyn_short[shift::], PLPCsyn_short[0:shift]]
        #print(Freqmap)
        #fshift=PLPCsyn_short[Freqmap]
        #seq_short2=np.fft.ifft(PLPCsyn_short[Freqmap2],axis=0)
        #print(seq_short-seq_short2)

        #Sequence in time domain
        seq_short=np.fft.ifft(PLPCsyn_short[Freqmap],axis=0)
        #extend to 161 samples
        seq_short_extended=np.array([], ndmin=2,dtype='complex')
        for i in range(4):
            seq_short_extended=np.r_['1', seq_short_extended, seq_short.T]

        ##windowing
        win=np.r_[0.5, np.ones((159)), 0.5]
        self.PLPCseq_short=seq_short_extended[0,0:161]*win
        self.PLPCseq_short.shape=(-1,1)
        #print(np.fft.fft(seq_short,axis=0))

        ##Generate long sequence
        seq_long=np.fft.ifft(PLPCsyn_long[Freqmap],axis=0)
        seq_long_extended=np.array([], ndmin=2,dtype='complex')
        print(seq_long)
        for i in range(4):
            seq_long_extended=np.r_['1', seq_long_extended, seq_long.T]

        print(seq_long_extended)
        self.PLPCseq_long=seq_long_extended[0,0:161]*win
        self.PLPCseq_long.shape=(-1,1)

        a=np.r_['0',self.PLPCseq_short, np.zeros((self.PLPCseq_long.shape[0]-1,1))] 
        b=np.r_['0', np.zeros((self.PLPCseq_short.shape[0]-1,1)), self.PLPCseq_long]
        self.PLPCseq=a+b 
        print(self.PLPCseq)
 
    def gen_plcp_header_field(self):
        pass

    def interpolate_at_ofdm(self,argdict={'signal':[]}):
        ratio=self.Rs/self.bbsigdict['BBRs']
        signal=argdict['signal']
        #Currently fixeed interpolation. check the fucntion definitions for details
        factors=mr.factor({'n':ratio})
        filterlist=mr.generate_interpolation_filterlist({'interp_factor':ratio})
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
        msg="Signal length is now %i" %(signal.shape[1])
        self.print_log({'type':'I', 'msg': msg}) 
        #print(filterlist)
        return signali



#Window to taper OFDM symbols
def window(argdict={'Tr':100e-9, 'length':478, 'fs':80e6, 'duration':240 }):
    #Length is the length of the signal. Duration is the duration of the payload data
    #Tails are due to interpolation
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

#    def run(self,*arg):
#        if len(arg)>0:
#            par=True      #flag for parallel processing
#            queue=arg[0]  #multiprocessing.Queue as the first argument
#        else:
#            par=False
#
#        if self.model=='py':
#            out=np.array(self.iptr_A.Value)
#            if par:
#                queue.put(out)
#            self._Z.Value=out


if __name__=="__main__":
    import scipy as sci
    import numpy as np
    import matplotlib.pyplot as plt
    from refptr import *
    from signal_generator_802_11n import *
    t=signal_generator_802_11n()
    t.gen_plpc_preamble_field()


