import os
from os import system

names = [
    "synthetic_comparisons_Frobenius_nls.py",
    "synthetic_comparisons_Frobenius.py",
    "synthetic_comparisons_KL_nls.py",
    "synthetic_comparisons_KL.py",
    "synthetic_Frobenius_delta_choice.py",
    "synthetic_KL_delta_choice.py",
    "test_audio_nls.py",
    "test_audio.py",
    "test_hyperspectral_nls.py",
    "test_hyperspectral.py"
]

for name in names:
    system("python "+name)
