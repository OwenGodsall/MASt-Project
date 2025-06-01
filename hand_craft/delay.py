import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import matplotlib
import seaborn as sns
import librosa.display
import scienceplots
plt.style.use(['science', 'no-latex'])


def sta_fun(np_data):

    if np_data is None:
        raise ValueError("Input array cannot be None")

    dat_min = np.min(np_data)
    dat_max = np.max(np_data)
    dat_mean = np.mean(np_data)
    dat_std = np.std(np_data)
    dat_range = dat_max - dat_min

    rel_pos_min = np.argmin(np_data) / (len(np_data) - 1)
    rel_pos_max = np.argmax(np_data) / (len(np_data) - 1)

    x = np.arange(len(np_data))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, offset = np.linalg.lstsq(A, np_data, rcond=None)[0]
    fit = slope * x + offset
    linreg_error = np.mean((np_data - fit) ** 2)

    s = pd.Series(np_data)
    dat_skew = s.skew()
    dat_kurt = s.kurt()

    return np.array([
        dat_mean,
        dat_std,
        dat_kurt,
        dat_skew,
        dat_min,
        dat_max,
        rel_pos_min,
        rel_pos_max,
        dat_range,
        slope,
        offset,
        linreg_error
    ])

def plot(spectrogram1, spectrogram2, title, loc, file_name, sample_rate, hop_length):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    img1 = librosa.display.specshow(spectrogram1, x_axis='time', y_axis='linear', sr=sample_rate, ax=ax1, hop_length=hop_length)
    cbar1 = fig.colorbar(img1, ax=ax1, format='%+2.2f dB')
    cbar1.ax.tick_params(labelsize=20)
    # ax1.set_title('Spectrogram 1', fontsize=16)
    ax1.set_ylabel('Frequency (Hz)', fontsize=22)
    ax1.set_xlabel('Time (s)', fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    if spectrogram2.shape[1]>1600:
        spectrogram2 = spectrogram2[:, :1600]

    img2 = librosa.display.specshow(spectrogram2, x_axis='time', y_axis='linear', sr=sample_rate, ax=ax2, hop_length=hop_length)
    cbar2 = fig.colorbar(img2, ax=ax2, format='%+2.2f dB')
    cbar2.ax.tick_params(labelsize=20)
    # ax2.set_title('Spectrogram 2', fontsize=16)
    ax2.set_xlabel('Delay Time (s)', fontsize=22)
    ax2.set_ylabel('Frequency (Hz)', fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    os.makedirs(loc, exist_ok=True)

    plt.savefig(os.path.join(loc, file_name + '.png'), dpi=300)

    plt.close(fig)

    print('figure saved')



def delay_img(spectrogram, hop_length, SR, ID, show_function = False, norm = False, delay_seconds = 5, device = 'cuda'):
    

    spectrogram = torch.tensor(spectrogram, dtype=torch.float32, device=device)
    
    freq_bins, length = spectrogram.shape
    num_samples = min(int(delay_seconds * SR / hop_length), length - 1)

    delay_t = torch.zeros((num_samples, freq_bins), device=device)

    for tau in range(num_samples):
        A = spectrogram[:, tau:]    
        B = spectrogram[:, :length - tau]
        delay_t[tau] = torch.mean(torch.abs(A - B), dim=1)

    delay_t = delay_t.cpu().numpy()
    img = np.array(delay_t)
    spectrogram = np.array(spectrogram.cpu().numpy())
    if norm:
        if img.max() != img.min():
            delay_img = (img - img.min()) / (img.max() - img.min())
        else:
            delay_img = img
            print("warning in producing spectrogram!")
    else:
        delay_img = img
    
    if show_function:
        output_dir = "fig/delay_images"
        os.makedirs(output_dir, exist_ok=True)

        
        output_path = os.path.join(output_dir, "delay_image" + ID + ".png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1

        plot(spectrogram, delay_img.T, "Delay Image", output_dir, os.path.splitext(os.path.basename(output_path))[0], sample_rate=SR, hop_length=hop_length)

    return delay_img.T


def exponential_func(x, a, b, c):
    return a * (1 - np.exp(-x / b)) ** c


def timescale(spectrogram, hop_length, sample_rate, show_peaks=False, show_bad_curves=False, show_sample_curves=False, show_time_scale_scatter=False):
    
    dt = hop_length / sample_rate
    time_scale = []
    multiplier = []
    power = []
    quiet_zone = 0
    bad_index = 0
    end_indexes = []
    bad_curves = []
    bad_curves_fit = []
    low_time_scale = []
    high_time_scale = []
    sample_curves = []
    sample_curves_fit = []
    threshold = 0.05

    freq_bins = int((spectrogram.shape[0]-1)* 7/8)
    gradient_ = np.gradient(spectrogram,axis=1)

    for i in range(freq_bins):
        
        smoothed = spectrogram[i]
        gradient = gradient_[i]

        
        if len(end_indexes) == 0:
            cap = len(gradient) - 1
        else:
            cap = last_end_index + 200
        end_indices = np.where(np.abs(gradient[:cap]) < threshold)[0]
        if len(end_indices) == 0:
            end_index = np.argmin(np.abs(gradient[:cap]))
            # print("no end index")
        else:
            end_index = end_indices[0]
            
        last_end_index  = end_index
        
        end_indexes.append(end_index)
        if end_index <= 1:
            # time_scale.append(0)
            # multiplier.append(0)
            # print("end index too small")
            bad_index += 1
            continue
        elif end_index >= len(smoothed) - 1:
            # time_scale.append(0)
            # multiplier.append(0)
            quiet_zone += 1
            continue
            
        x_data = np.arange(end_index)
        y_data = smoothed[:end_index]

        def exponential_func(x, a, c):
            return a * (1 - np.exp(-x / end_index)) ** c

        # Fit the exponential function to the data
        try:
            popt, pcov = curve_fit(exponential_func, x_data, y_data, p0=[20, 0.5], maxfev=10000)
         
        except RuntimeError:
            bad_index += 1
            continue

        a, c = popt
        
        time_scale.append(end_index * dt)
        multiplier.append(a)
        power.append(c)

        fit = exponential_func(x_data, a, c)

        if show_bad_curves:
            bad_curves.append((i, smoothed))
            bad_curves_fit.append((i, fit))

        if i in [45, 120, 240]:
            sample_curves.append((i, smoothed))
            sample_curves_fit.append((i, fit))

    print("bad index", bad_index) 
    print("quiet index", quiet_zone)   
    if show_peaks:
        output_dir = "fig/peak_hist"
        os.makedirs(output_dir, exist_ok=True)

       
        plt.figure(figsize=(10, 8))

        
        plt.subplot(3, 1, 1)
        plt.hist([ts for ts in time_scale if ts > 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f"Histogram of Time Scales (All Values) (Bad Index: {bad_index}, Quiet Zone: {quiet_zone})")
        plt.xlabel("Time Scale (s)")
        plt.ylabel("Frequency")
        plt.tight_layout(pad=3.0)

        
        plt.subplot(3, 1, 2)
        plt.hist(multiplier, bins=20, color='green', alpha=0.7, edgecolor='black')
        plt.title("Histogram of Multipliers")
        plt.xlabel("Multiplier")
        plt.xscale("log")
        plt.ylabel("Frequency")
        plt.tight_layout(pad=3.0)

        plt.subplot(3, 1, 3)
        plt.hist(power, bins=20, color='red', alpha=0.7, edgecolor='black')
        plt.title("Histogram of Powers")
        plt.xlabel("Power")
        plt.xscale("log")
        plt.ylabel("Frequency")
        plt.tight_layout(pad=3.0)

       
        output_path = os.path.join(output_dir, "combined_histograms.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        plt.savefig(output_path)
        plt.close()

    if show_bad_curves and bad_curves:
        output_dir = "fig/512_Plots"
        os.makedirs(output_dir, exist_ok=True)

       
        plt.figure(figsize=(12, 16)) 
        num_subplots = 4
        subset_size = 10 
        total_curves = len(bad_curves)
        curves_per_subplot = min(subset_size, total_curves // num_subplots)

        for subplot_idx in range(num_subplots):
            start_idx = subplot_idx * curves_per_subplot * 5
            end_idx = start_idx + curves_per_subplot
            plt.subplot(2, 2, subplot_idx + 1)  

            for (idx, curve), (_, fit) in zip(bad_curves[start_idx:end_idx], bad_curves_fit[start_idx:end_idx]):
                color = plt.cm.tab10(idx % 10) 
                plt.plot(np.array(range(len(curve))) * dt, curve, label=f"Curve {idx}", color=color)
                plt.plot(np.array(range(len(fit))) * dt, fit, label=f"Fit {idx}", linestyle='--', color=color)
                end_index = end_indexes[idx]
                plt.axvline(x=end_index * dt, color='red', linestyle='--', label=f"End Index {idx}")

            plt.title(f"Bad Index Curves (Set {subplot_idx + 1})")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.legend(fontsize='small', loc='upper right') 
            plt.tight_layout(pad=3.0)  

        
        output_path = os.path.join(output_dir, "bad_curves.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        plt.savefig(output_path)
        plt.close()

    if show_sample_curves and sample_curves:
        output_dir = "fig/sample"
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.style.use(['science', 'no-latex'])
        
        fig, ax = plt.subplots(figsize=(10, 6),constrained_layout=True)


        sns.set_style("ticks")

        
        i = 0
        colors = sns.color_palette("Set1", 3) 
        for (idx, curve), (_, fit) in zip(sample_curves, sample_curves_fit):
            color = colors[i] 
            i+=1 
            ax.plot(np.arange(len(curve)) * dt, curve, label=f"{int(_*16.66666) + 1} Hz", color=color)
           
        
        ax.set_xlabel("Delay Time (s)" ,fontsize=22)
        ax.set_ylabel("$\Delta$ Amplitude (dB)",  fontsize=22)
        lims = plt.ylim()
        xlims = plt.xlim()
        ax.set_ylim(0, 50)
        ax.set_xlim(0,2)
        ax.tick_params(axis='both', which='major', labelsize=20)  # Increase tick size
        ax.legend(fontsize='20', loc='upper right') 

        
        axins = inset_axes(ax,
                width="10%",  
                height="50%", 
            bbox_transform=ax.transAxes,
            bbox_to_anchor=(0.15, -0.05, 1, 1), 
            loc='upper left',
            borderpad=0  
            )  
        
        
        fit_lengths = []
        amplitudes = []
        i=0
        for (idx, curve), (_, fit) in zip(sample_curves, sample_curves_fit):
            color = colors[i]  
            i+=1
            axins.plot(np.arange(len(curve)) * dt, curve, color=color)
            axins.plot(np.arange(len(fit)) * dt, fit, linestyle='--', color='red', alpha=0.7)
            fit_lengths.append(len(fit)*dt)
            amplitudes.append(fit[-1])
        xlim = np.max(fit_lengths)
        ylim = np.max(amplitudes)

        axins.set_xlim(0, xlim)
        axins.set_ylim(0, ylim)
        
        axins.tick_params(direction='in', labelbottom=False, labelleft=False, bottom=False, left=False)



       
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        

       
        output_path = os.path.join(output_dir, "sample_curves.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        
        plt.savefig(output_path, dpi = 300, bbox_inches='tight')
        plt.close()

    if show_time_scale_scatter:
        output_dir = "fig/time_scale"
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("ticks")
        plt.style.use(['science','no-latex'])
       
        plt.figure(figsize=(12, 8), dpi=300)
        x_val = np.arange(len(time_scale)) * 15.625

        time_scale = np.array(time_scale)/np.max(time_scale)
        multiplier = np.array(multiplier)/np.max(multiplier)
        power = np.array(power)/np.max(power)

        plt.scatter(x_val, time_scale, alpha=0.7, color=sns.color_palette("Set1")[0], label="Time Scale", marker='o')
        plt.scatter(x_val, multiplier, alpha=0.7, color=sns.color_palette("Set1")[1], label="Multiplier", marker='^')
        plt.scatter(x_val, power, alpha=0.7, color=sns.color_palette("Set1")[2], label="Power", marker='s')
        plt.xlabel("Frequency (Hz)", fontsize=22)
        plt.ylabel("Normalised Value", fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)

       
        '''plt.subplot(3, 1, 1)
        sns.scatterplot(x=x_val, y=time_scale, alpha=0.7, color=sns.color_palette("pastel")[0])
        
        plt.xlabel("Frequency (Hz)", fontsize=22)
        plt.ylabel("Time Scale (s)", fontsize=22)
        plt.tight_layout(pad=3.0)

        
        plt.subplot(3, 1, 2)
        sns.scatterplot(x=x_val, y=multiplier, alpha=0.7, color=sns.color_palette("pastel")[1])
       
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Multiplier")
        plt.tight_layout(pad=3.0)

       
        plt.subplot(3, 1, 3)
        sns.scatterplot(x=x_val, y=power, alpha=0.7, color=sns.color_palette("pastel")[2])
        
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.tight_layout(pad=3.0)'''

       
        output_path = os.path.join(output_dir, "combined_scatterplots.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        plt.savefig(output_path, dpi  = 300)
        plt.close()

    return time_scale, multiplier, power


    
def extract_delay_timescales(signal, signal_sr, ID, HOP, FRAME_LEN):
    
    
    stft = librosa.stft(
        y=signal, n_fft=int(FRAME_LEN), hop_length=int(HOP), win_length=int(FRAME_LEN), window='hann'
    )
    
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    

    print("spectrogram shape", spectrogram.shape)

    
    delay_spectrogram = delay_img(spectrogram, hop_length = int(HOP), SR=signal_sr, show_function=False, norm=False, ID=ID, delay_seconds=2, device='cuda')

    timescales, multipliers, powers = timescale(delay_spectrogram, hop_length=int(HOP), sample_rate=signal_sr, show_peaks=False, show_bad_curves=False, show_sample_curves=False, show_time_scale_scatter=False)

    return np.stack([timescales, multipliers, powers], axis=0)

def timescale_fix(spectrogram, hop_length, sample_rate, time_scale= 0.05, show_peaks=False, show_bad_curves=False, show_sample_curves=False, show_time_scale_scatter=False):
    
    dt = hop_length / sample_rate
    index = sample_rate / hop_length
    multiplier = []
    decay = []
    quiet_zone = 0
    bad_index = 0
    end_indexes = []
    bad_curves = []
    bad_curves_fit = []
    low_time_scale = []
    high_time_scale = []
    sample_curves = []
    sample_curves_fit = []
    threshold = 0.05
    

    freq_bins = int((spectrogram.shape[0]-1)* 7/8)
    initial_bin = int(50 * spectrogram.shape[0] / 8000)  

    for i in range(initial_bin, freq_bins):
        
        smoothed = spectrogram[i]
        end_index = int(time_scale * index)

        x_data = np.arange(end_index)
        y_data = smoothed[:end_index]

        def exponential_func(x, a, c):
            return a * (1 - np.exp(-x / end_index)) ** c

        
        try:
            popt, pcov = curve_fit(exponential_func, x_data, y_data, p0=[10, 1], maxfev=10000)
           
        except RuntimeError:
            bad_index += 1
            continue

        a, c = popt

        
        
        multiplier.append(a)
        decay.append(c)

        fit = exponential_func(x_data, a, c)

        if show_bad_curves:
            bad_curves.append((i, smoothed))
            bad_curves_fit.append((i, fit))

        if i in [6,7,8, 9, 10, 11, 12,13, 14]:
            sample_curves.append((i, smoothed))
            sample_curves_fit.append((i, fit))

    print("bad index", bad_index)    
    if show_peaks:
        output_dir = "fig/peak_hist"
        os.makedirs(output_dir, exist_ok=True)

        
        plt.figure(figsize=(10, 8))

        
        plt.subplot(3, 1, 1)
        plt.hist([ts for ts in time_scale if ts > 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f"Histogram of Time Scales (All Values) (Bad Index: {bad_index}, Quiet Zone: {quiet_zone})")
        plt.xlabel("Time Scale (s)")
        plt.ylabel("Frequency")
        plt.tight_layout(pad=3.0)

        
        plt.subplot(3, 1, 2)
        plt.hist(multiplier, bins=20, color='green', alpha=0.7, edgecolor='black')
        plt.title("Histogram of Multipliers")
        plt.xlabel("Multiplier")
        plt.xscale("log")
        plt.ylabel("Frequency")
        plt.tight_layout(pad=3.0)

        plt.subplot(3, 1, 3)
        plt.hist(power, bins=20, color='red', alpha=0.7, edgecolor='black')
        plt.title("Histogram of Powers")
        plt.xlabel("Power")
        plt.xscale("log")
        plt.ylabel("Frequency")
        plt.tight_layout(pad=3.0)

     
        output_path = os.path.join(output_dir, "combined_histograms.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        plt.savefig(output_path)
        plt.close()

    if show_bad_curves and bad_curves:
        output_dir = "fig/512_Plots"
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 16)) 
        num_subplots = 4
        subset_size = 10  
        total_curves = len(bad_curves)
        curves_per_subplot = min(subset_size, total_curves // num_subplots)

        for subplot_idx in range(num_subplots):
            start_idx = subplot_idx * curves_per_subplot * 5
            end_idx = start_idx + curves_per_subplot
            plt.subplot(2, 2, subplot_idx + 1) 

            for (idx, curve), (_, fit) in zip(bad_curves[start_idx:end_idx], bad_curves_fit[start_idx:end_idx]):
                color = plt.cm.tab10(idx % 10) 
                plt.plot(np.array(range(len(curve))) * dt, curve, label=f"Curve {idx}", color=color)
                plt.plot(np.array(range(len(fit))) * dt, fit, label=f"Fit {idx}", linestyle='--', color=color)
        
               

            plt.title(f"Bad Index Curves (Set {subplot_idx + 1})")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.legend(fontsize='small', loc='upper right') 
            plt.tight_layout(pad=3.0) 

        
        output_path = os.path.join(output_dir, "bad_curves.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        plt.savefig(output_path)
        plt.close()

    if show_sample_curves and sample_curves:
        output_dir = "fig/sample"
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
       
        fig, ax = plt.subplots(figsize=(12, 8),constrained_layout=True)

       

        sns.set_palette("pastel")
        sns.set_style("ticks")

        

        for (idx, curve), (_, fit) in zip(sample_curves, sample_curves_fit):
            ax.plot(np.arange(len(curve)) * dt, curve, label=f"{_*15.625} Hz")
           
           
        ax.set_xlabel("Delay Time (s)" ,fontsize=22)
        ax.set_ylabel("$\Delta$ Amplitude (dB)",  fontsize=22)
        lims = plt.ylim()
        xlims = plt.xlim()
        ax.set_ylim(0, 50)
        ax.set_xlim(0,2)
        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase tick size
        ax.legend(fontsize='20', loc='upper right') 

      
        axins = inset_axes(ax,
                   width="10%",  
                   height="50%", 
                    bbox_transform=ax.transAxes,
                    bbox_to_anchor=(0.15, -0.05, 1, 1), 
                    loc='upper left',
                    borderpad=0 
                    ) 
        
        
        
       
        
        fit_lengths = []
        amplitudes = []
        for (idx, curve), (_, fit) in zip(sample_curves, sample_curves_fit):
            axins.plot(np.arange(len(curve)) * dt, curve)
            axins.plot(np.arange(len(fit)) * dt, fit, linestyle='--', color='red', alpha=0.7)
            fit_lengths.append(len(fit)*dt)
            amplitudes.append(fit[-1])
        
        xlim = np.max(fit_lengths)
        ylim = np.max(amplitudes)

        axins.set_xlim(0, xlim)
        axins.set_ylim(0, ylim)
        
        axins.tick_params(direction='in', labelbottom=False, labelleft=False, bottom=False, left=False)



       
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        

     
        output_path = os.path.join(output_dir, "sample_curves.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
       
        plt.savefig(output_path, dpi = 300, bbox_inches='tight')
        plt.close()

    if show_time_scale_scatter:
        output_dir = "fig/time_scale_scatter"
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("ticks")
   
        plt.figure(figsize=(14, 10), dpi=300)
        x_val = np.arange(len(multiplier)) * 8000 / len(spectrogram)

     
        multiplier_view = np.array(multiplier)/np.max(multiplier)
        decay_view = np.array(decay)/np.max(decay)

      
        plt.scatter(x_val, multiplier_view, alpha=0.7, color=sns.color_palette("Set1")[1], label="Multiplier", marker='^')
        plt.scatter(x_val, decay_view, alpha=0.7, color=sns.color_palette("Set1")[2], label="Power", marker='s')
        plt.xlabel("Frequency (Hz)", fontsize=22)
        plt.ylabel("Normalised Value", fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)

  
        output_path = os.path.join(output_dir, "combined_scatterplots.png")
        base, ext = os.path.splitext(output_path)
        existing_files = os.listdir(output_dir)
        counter = 1
        while os.path.basename(output_path) in existing_files:
            output_path = f"{base}({counter}){ext}"
            counter += 1
        plt.savefig(output_path, dpi  = 300)
        plt.close()

    return multiplier, decay
