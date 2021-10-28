import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from nn_fac.nmf import nmf as nmf_hals
from tensorly.tenalg.proximal import fista
import soundfile as sf
from scipy import signal
'''
 W is a dictionary of 88 columns and 4097 frequency bins. Each column was obtained by performing a rank-one NMF (todo correct) on the recording of a single note in the MAPS database, on a Yamaha Disklavier with close microphones, the note was played mezzo forte and the loss was beta-divergence with beta=1.

 By performing matrix NNLS from the Power STFT Y of 30s of a (relatively simple) song in MAPS, recorded with the same piano in similar conditions and processes in the same way, we expect that $Y \approx WH$, where H are the activations of each note in the recording. A good loss to measure discrepencies here is the beta-divergence with beta=1.

 For the purpose of this toy experiment, only one song from MAPS is selected. We then perform NMF, and look at the activations as a piano roll.

 For the NMF part, we simply discard the provided templates, and estimate both the templates and the activations. Again it is best to use KL divergence. We can initialize with the provided template to get a initial dictionary.

 Input parameters:
 rank : int (default 88) number of notes to estimate in the music excerpt.
'''

#-------------------------------------------------------------------------
# Modeling

# Importing a good dictionnary for the NNLS part
W0 = np.load('./attack_dict_piano_ENSTDkCl_beta_1_stftAD_True_intensity_M.npy')

# Importing data and computing STFT using the Attack-Decay paper settings
# Read the song (you can use your own!)
the_signal, sampling_rate_local = sf.read('./MAPS_MUS-bk_xmas1_ENSTDkCl.wav')
# Using the settings of the Attack-Decay transcription paper
the_signal = the_signal[:,0] + the_signal[:,1] # summing left and right channels
frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
#time_atoms = time_atoms # ds scale
# Taking the power spectrogram
Y = np.abs(Y)**2
# adding some constant noise for avoiding zeros
#Y = Y+1e-8
# Cutting silence, end song and high frequencies (>10600 Hz)
cutf = 2000
cutt_in = 0 # song beginning
cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]
# Also cutting the dictionary
W0 = W0[:cutf,:]

# get sizes
m, n = Y.shape


#------------------------------------------------------------------
# NNLS with fixed dictionary

H_nnls = fista(W0.T@Y, W0.T@W0, tol=1e-16, n_iter_max=1000)

#------------------------------------------------------------------
# Computing the NMF to try and recover activations and templates
rank = 88
beta = 2 # beta=1 data is big :(

out = nmf_hals(Y, rank, init='custom', U_0 = W0, V_0 = H_nnls, return_costs=True, n_iter_max=1000, tol=1e-16, beta=beta, update_rule='hals')

#-----------------------------------------------------------------
# Results post-processing

# Normalize output
W = out[0]
H = out[1]
normsW = np.sum(W,axis=0)
W = W/normsW
#H = np.diag(1/np.max(H,1))@H
H = np.diag(normsW)@H

# Printing W and H
plt.figure()
plt.subplot(121)
plt.imshow(W[:200, :], aspect='auto')
ticks = np.trunc(frequencies[0:200:10])
plt.yticks(range(0,200,10), ticks.astype(int))
plt.ylabel('Hz')
plt.title('W learnt')
plt.subplot(122)
plt.imshow(W0[:200, :], aspect='auto')
ticks = np.trunc(frequencies[0:200:10])
plt.yticks(range(0,200,10), ticks.astype(int))
plt.title('Provided pre-trained W')
plt.ylabel('Hz')
#plt.xticks(range(8),notes)

# Plotting H
#plt.figure()
#for i in range(rank):
#    plt.subplot(rank,1,i+1)
#    plt.plot(H[i,:])
#    plt.xticks([])
#    plt.yticks([])
#    if i==rank-1:
#        hop = 100
#        ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
#        ticks_number = ticks.shape[0]
#        plt.xticks(range(0,ticks_number*hop,hop), ticks)
#    #plt.ylabel(notes[i])

# Plotting H version 2
# Thresholding the H values for activation detection, and plotting bitmap
thres = 1e-3 # todo handfix
H_plot = np.copy(H)
H_plot[H_plot<thres]=0
H_plot[H_plot>=thres]=1
H_nnls_plot = np.copy(H_nnls)
H_nnls_plot[H_nnls_plot<thres]=0
H_nnls_plot[H_nnls_plot>=thres]=1
hop = 100
plt.figure()
plt.subplot(121)
plt.imshow(H_plot, aspect='auto', interpolation='none')
plt.xticks([])
plt.yticks([])
ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
ticks_number = ticks.shape[0]
plt.xticks(range(0,ticks_number*hop,hop), ticks)
plt.ylabel('notes (not labeled)')
plt.title('H with NMF')
plt.subplot(122)
plt.imshow(H_nnls_plot, aspect='auto', interpolation='none')
plt.xticks([])
plt.yticks([])
ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
ticks_number = ticks.shape[0]
plt.xticks(range(0,ticks_number*hop,hop), ticks)
plt.title('H with nnls')
plt.ylabel('notes in order')

# Printing Y
plt.figure()
plt.subplot(211)
plt.imshow(Y[:200,:])
yticks = np.trunc(frequencies[0:200:20])
plt.yticks(range(0,200,20), yticks.astype(int))
plt.ylabel('Hz')
hop = 100
xticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
ticks_number = xticks.shape[0]
plt.xticks(range(0,ticks_number*hop,hop), xticks)
plt.xlabel('time (s)')
plt.title('Y')
plt.subplot(212)
plt.imshow(np.sqrt(Y[:200,:]))
plt.yticks(range(0,200,20), yticks.astype(int))
plt.ylabel('Hz')
plt.xticks(range(0,ticks_number*hop,hop), xticks)
plt.xlabel('time (s)')
plt.title('sqrt(Y)')


plt.show()
