import pandas as pd
import numpy as np
from scipy import signal as sp
from scipy.fft import fft, fftfreq, next_fast_len
import plotly.graph_objects as go
import os

class THzSpec:
        def __init__(self,spec_path,sample_type, plotter='plotly', name=None):
            # normalize time and convert to ps
            df = pd.read_csv(spec_path, sep='\s+', header=None, usecols=[0, 1], names=['Time', 'Amp'])
            # normalize time and convert to ps
            df['Time'] = (df['Time'] - df['Time'].iloc[0])*1E-12
            if name is None:
                self.name=os.path.basename(spec_path)
            else:
                self.name= name
            self.last_processing=None
            self.plotter = plotter
            self.sample_type=sample_type
            self.time = df['Time'].to_numpy()
            self.amp = df['Amp'].to_numpy()
            self.windowed_signal= self.amp #incase no window required
            self.idxmax = np.argmax(self.amp)
            self.sampling_rate = 1 / (self.time[1] - self.time[0])
            print("Read in file {}, {} points at sampling rate of {:.1e} Hz".format(spec_path,len(self.amp),self.sampling_rate))

        def __getitem__(self, item):
            return self.__dict__[item]
        
        def apply_window(self,start,end,curve,shift=0):
             # Apply the Tukey window
            window = np.zeros_like(self.amp)
            try:
                # offset window if necessary
                start = start + shift
                end = end + shift
                # generate window function
                length = end - start
                window[start:end] = sp.windows.tukey(length, alpha=curve)
                #window[start:end] = tukey(end - start, alpha=curve)
            except ValueError as e:
                raise ValueError("Your window is larger than the data:", e)
            # apply window
            self.window_function=window
            self.windowed_signal = self.amp * window
            self.windowed_signal = self.windowed_signal[:end]

        
        def fill_transform(self,zero_padding):
            x= next_fast_len(len(self.windowed_signal))
            if  x < 2 ** zero_padding:
                x = 2 ** zero_padding
            # calculate the padded signal for display only
            # self.padded_signal = self.windowed_signal.copy()
            # self.padded_signal.resize(x)
            self.padded_signal = np.append(self.windowed_signal,[0,0])
            filtered_time= self.time[:len(self.windowed_signal)]
            self.padded_time= np.append(filtered_time,[(len(filtered_time)+1)/self.sampling_rate, x/self.sampling_rate])
            # perfrom the FFT
            self.fft_result = fft(self.windowed_signal, x)
            self.fft_amp=np.abs(self.fft_result)
            self.fft_freq = fftfreq(len(self.fft_result), 1 / self.sampling_rate)
            self.wrapped_phase= np.angle(self.fft_result[:len(self.fft_result)//2])
            self.unwrapped_phase = np.unwrap(self.wrapped_phase)


        def process_signal(self,start,end,curve, zero_padding, shift=0):
            self.last_processing={'start':start,'end':end,'curve':curve, 'zero_padding':zero_padding}
            self.apply_window(start, end, curve, shift)
            self.fill_transform(zero_padding) 
        
        def _curve(self,x,y, label):
            if self.plotter == 'plotly':
                curve = go.Scatter(x=x,y=y,name=label)
            # if self.plotter == 'hv':
            #     curve = hv.Curve((x,y))
            return curve


        def window_curve(self):
            return self._curve(self.time, self.window_function, self.sample_type +'_window')
        def raw_curve(self):
            return self._curve(self.time, self.amp, (self.sample_type + '_amplitude'))
        def processed_signal_curve(self):
            # return self._curve(np.arange(len(self.padded_signal)) / self.sampling_rate, self.padded_signal, self.sample_type +'_processed_signal')
            return self._curve(self.padded_time, self.padded_signal, self.sample_type +'_processed_signal')
        def spectral_curve(self):
            return self._curve(self.fft_freq[:len(self.fft_freq)//2],self.fft_amp[:len(self.fft_amp)//2],self.sample_type +'_spectrum')
        def unwrapped_phase_curve(self):
            return self._curve(self.fft_freq[:len(self.fft_freq)//2], self.unwrapped_phase, self.sample_type +'_unwrapped_phase')
        def wrapped_phase_curve(self):
            return self._curve(self.fft_freq[:len(self.fft_freq)//2], self.wrapped_phase, self.sample_type +'_wrapped_phase')


class THzSpecSet:
    def __init__(self,sample_data, ref_data):
        if 'THzSpec' in str(type(sample_data)):
            self.sample= sample_data
        else:
            self.sample=THzSpec(sample_data, 'sample')
        if 'THzSpec' in str(type(sample_data)):
            self.ref= ref_data
        else:
            self.ref=THzSpec(ref_data, 'ref')
        self.set = (self.sample, self.ref)
        self.shift = self.sample.idxmax-self.ref.idxmax

    def __getitem__(self,i):
        return self.set[i]
    
    def apply_window(self,start,end,curve, use_shift=True):
        for i in range(2):
            if use_shift:
                self.set[i].apply_window(start,end,curve,self.shift)
            else:
                self.set[i].apply_window(start,end,curve)
    
    def fill_transform(self, zero_padding):
        for i in range(2):
            self.set[i].fill_transform(zero_padding)

    def process_signals(self,start,end,curve,zero_padding, use_shift=True):
        if use_shift:
            self.set[0].process_signal(start,end,curve,zero_padding,self.shift)
            self.set[1].process_signal(start,end,curve,zero_padding)
                
        else:
            for i in range(2):    
                self.set[i].process_signal(start,end,curve,zero_padding)
    
    def window_curves(self):
        return [self.set[x].window_curve() for x in [0,1]]
    def raw_curves(self):
        return [self.set[x].raw_curve() for x in [0,1]]
    def processed_signal_curves(self):
        return [self.set[x].processed_signal_curve() for x in [0,1]]
    def spectral_curves(self):
        return [self.set[x].spectral_curve() for x in [0,1]]
    def unwrapped_phase_curves(self):
        return [self.set[x].unwrapped_phase_curve() for x in [0,1]]
    def wrapped_phase_curves(self):
        return [self.set[x].wrapped_phase_curve() for x in [0,1]]
    
       
                
class THzData():
    def __init__(self):
        self.i=0
        self.df = pd.DataFrame()
    

    def add_data(self,sample_data,ref_data,name, **kwargs):
        data_obj= THzSpecSet(sample_data,ref_data)
        temp_df= pd.DataFrame({'name':name,'sample_file':data_obj.sample.name, 'reference_file':data_obj.ref.name,'data':data_obj}|kwargs, index=[self.i])
        self.df=pd.concat([self.df,temp_df])
        self.i+=1
    def process_all(self,start,end,curve,zero_padding, use_shift=True):
        for index,row in self.df.iterrows():
                row['data'].process_signals(start,end,curve,zero_padding, use_shift=True)