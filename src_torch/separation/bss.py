"""
Offline Blind Source Separation example

Demonstrate the performance of different blind source separation (BSS) algorithms:

1) Independent Vector Analysis based on auxiliary function (AuxIVA)
The method implemented is described in the following publication

    N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

2) Independent Low-Rank Matrix Analysis (ILRMA)
The method implemented is described in the following publications

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization*, IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, September 2016

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined Blind
    Source Separation with Independent Low-Rank Matrix Analysis*, in Audio Source Separation,
    S. Makino, Ed. Springer, 2018, pp.  125-156.

3) Sparse Independent Vector Analysis based on auxiliary function (SparseAuxIVA)
The method implemented is described in the following publication

    J. Jansky, Z. Koldovsky, and N. Ono *A computationally cheaper method for blind speech
    separation based on AuxIVA and incomplete demixing transform*, Proc. IEEE, IWAENC, 2016.

4) Fast Multichannel Nonnegative Matrix Factorization (FastMNMF)
The method implemented is described in the following publication

    K. Sekiguchi, A. A. Nugraha, Y. Bando, K. Yoshii, *Fast Multichannel Source 
    Separation Based on Jointly Diagonalizable Spatial Covariance Matrices*, EUSIPCO, 2019.

5) Fast Multichannel Nonnegative Matrix Factorization 2 (FastMNMF2)
The method implemented is described in the following publication

    K. Sekiguchi, Y. Bando, A. A. Nugraha, K. Yoshii, T. Kawahara, *Fast Multichannel Nonnegative
    Matrix Factorization With Directivity-Aware Jointly-Diagonalizable Spatial
    Covariance Matrices for Blind Source Separation*, IEEE/ACM TASLP, 2020.

All the algorithms work in the STFT domain. The test files were extracted from the
`CMU ARCTIC <http://www.festvox.org/cmu_arctic/>`_ corpus.


Depending on the input arguments running this script will do these actions:.

1. Separate the sources.
2. Show a plot of the clean and separated spectrograms
3. Show a plot of the SDR and SIR as a function of the number of iterations.
4. Create a `play(ch)` function that can be used to play the `ch` source (if you are in ipython say).
5. Save the separated sources as .wav files
6. Show a GUI where a mixed signals and the separated sources can be played

This script requires the `mir_eval` to run, and `tkinter` and `sounddevice` packages for the GUI option.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from FastMNMF2 import FastMNMF2, EPS, MIC_INDEX, MultiSTFT
from Base import MultiISTFT

import torch
import torchaudio
import numpy as np
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile

wav_file = "/n/work1/juchen/BSS/recordings/466c0307.46oc0209..wav"
fs, data = wavfile.read(wav_file)
data = data[:,1:5]
data = np.array(data)

# ref_file = "/n/work1/juchen/BSS/recordings/01wo0316.47zc0303..wav"
# ref_data = wavfile.read(ref_file)[1]
# ref_data = np.array(ref_data).astype(np.float64).T

time_offset_sec = 0.3

if __name__ == "__main__":
    choices = ["fastmnmf2"]

    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstration of blind source separation using "
        "IVA, ILRMA, sparse IVA, FastMNMF, or FastMNMF2 ."
    )
    parser.add_argument("-b", "--block", type=int, default=2048, help="STFT block size")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default=choices[0],
        choices=choices,
        help="Chooses BSS method to run",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Saves the output of the separation to wav files",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_source", type=int, default=2, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=64, help="number of basis")
    parser.add_argument("--n_iter_init", type=int, default=50, help="nujmber of iteration used in twostep init")
    parser.add_argument(
        "--init_SCM",
        type=str,
        default="twostep",
        help="circular, obs (only for enhancement), twostep",
    )
    parser.add_argument("--n_iter", type=int, default=150, help="number of iteration")
    parser.add_argument("--g_eps", type=float, default=5e-2, help="minumum value used for initializing G_NM")
    parser.add_argument("--n_mic", type=int, default=4, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--method", type=str, default="IP", help="the method for updating Q")
    args = parser.parse_args()

    import pyroomacoustics as pra

    ## Prepare one-shot STFT
    L = args.block
    hop = L // 2
    win_a = pra.hann(L)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)


    ## STFT ANALYSIS
    X = pra.transform.stft.analysis(data, L, hop, win=win_a)
    wav = torch.Tensor(data)

    t_begin = time.perf_counter()

    ## START BSS
    bss_type = args.algo
    if bss_type == "fastmnmf2":
        print("Run FastMNMF2")

        #wav /= torch.abs(wav).max() * 1.2
        spec_FTM = MultiSTFT(wav[:, :], n_fft=args.block)
        
        if args.gpu < 0:
            device = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            device = "cuda"

        separater = FastMNMF2(
            n_source=args.n_source,
            n_basis=args.n_basis,
            device=device,
            init_SCM=args.init_SCM,
            n_bit=args.n_bit,
            algo=args.method,
            n_iter_init=args.n_iter_init,
            g_eps=args.g_eps,
        )

        separater.file_id = None
        separater.load_spectrogram(spec_FTM, fs)
        separater.solve(
            n_iter=args.n_iter,
            save_dir="./",
            save_likelihood=False,
            save_param=False,
            save_wav=False,
            interval_save=5,
        )

        Y = separater.separated_spec

        # Y = pra.bss.fastmnmf2(
        #     X, n_iter=100, n_components=16, n_src=2, callback=convergence_callback
        # )

    t_end = time.perf_counter()
    print("Time for BSS: {:.2f} s".format(t_end - t_begin))

    ## STFT Synthesis
    # y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
    assert not torch.isnan(Y).any(), "spec includes NaN"
    y = MultiISTFT(Y, shape="MFT").to(torch.float32)
    y = y.numpy()
    y = y[:, int(time_offset_sec*fs):]

    ## Compare SDR and SIR
    # y = y[:, L - hop:]
    # print(y.shape)
    # m = np.minimum(y.shape[1], y.shape[1])

    # sdr, sir, sar, perm = bss_eval_sources(y[:, :m], y[:, :m], compute_permutation=True)
    # print("SDR:", sdr)
    # f = open('SDR_result.txt', 'a')
    # f.write(str(sdr[0])+'\n'+str(sdr[1])+'\n')
    # f.close


    ## MVDR Beamforming
    # room.mic_array = mics
    # room.compute_rir()
    # room.simulate()
    
    # m = min(mics_signals.shape[1], y.shape[1])
    # for i, signal in enumerate(room.mic_array.signals):
    #     signal[:m] = mics_signals[i][:m]-y[1][:m]

    # sigma2_n = 5e-7
    # mics.rake_mvdr_filters(
    #     room.sources[0][0:1],
    #     room.sources[1][0:1],
    #     sigma2_n * np.eye(mics.Lg * mics.M),
    #     delay=0.05,
    # )
    # output = mics.process()
    # input_mic = pra.normalize(pra.highpass(mics.signals[0], room.fs))
    # wavfile.write("bf.wav", room.fs, input_mic)
    # m = min(ref.shape[1], input_mic.shape[0])
    # sdr_, sir_, sar_, perm_ = bss_eval_sources(ref[0, :m, 1], input_mic[:m])
    # print("SDR after MVDR beamforming", sdr_)
    

    ## PLOT RESULTS
    plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.specgram(ref_data[0], NFFT=1024, Fs=fs)
    # plt.title("Source 0 (clean)")
    # plt.colorbar()

    # plt.subplot(2, 2, 2)
    # plt.specgram(ref_data[1], NFFT=1024, Fs=fs)
    # plt.title("Source 1 (clean)")
    # plt.colorbar()

    plt.subplot(3, 1, 1)
    plt.specgram(data[:, 1], NFFT=1024, Fs=fs)
    plt.title("Mixture")
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.specgram(y[0, :], NFFT=1024, Fs=fs)
    plt.title("Source 0 (separated)")
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.specgram(y[1, :], NFFT=1024, Fs=fs)
    plt.title("Source 1 (separated)")
    plt.colorbar()

    plt.tight_layout(pad=0.5)

    plt.savefig("specs.png")

    plt.figure()
    # a = np.array(SDR)
    # print(len(SDR))
    # b = np.array(SIR)
    # plt.plot(
    #     np.arange(a.shape[0]) *10, a[:, 0], label="SDR Source 0", c="r", marker="*"
    # )
    # plt.plot(
    #     np.arange(a.shape[0]) *10, a[:, 1], label="SDR Source 1", c="r", marker="o"
    # )
    # plt.plot(
    #     np.arange(b.shape[0]) *10, b[:, 0], label="SIR Source 0", c="b", marker="*"
    # )
    # plt.plot(
    #     np.arange(b.shape[0]) *10, b[:, 1], label="SIR Source 1", c="b", marker="o"
    # )
    # plt.legend()

    # plt.tight_layout(pad=0.5)

    # plt.savefig("result.png")

    if args.save:
        from scipy.io import wavfile

        for i, sig in enumerate(y):
            print(y.shape)
            wavfile.write(
                "bss_source{}.wav".format(i + 1),
                fs,
                pra.normalize(sig, bits=16).astype(np.int16),
            )
        # for i, sig in enumerate(ref_data):
        #     wavfile.write(
        #         "bss_source{}_target.wav".format(i + 1),
        #         fs,
        #         pra.normalize(sig, bits=16).astype(np.int16),
        #     )