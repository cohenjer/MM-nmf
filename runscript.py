import os
from os import system

nbseeds=0

names = [
    "synthetic_comparisons_Frobenius_nls.py",
    "synthetic_comparisons_Frobenius.py",
    "synthetic_comparisons_KL_nls.py",
    "synthetic_comparisons_KL.py",
    "synthetic_Frobenius_delta_choice.py",
    "synthetic_KL_delta_choice.py",
    "test_audio_nls.py",
    "test_audio_nmf.py",
    "test_hyperspectral_nls.py",
    "test_hyperspectral.py"
]
# run audio nmf by hand
for name in names:
    print(name+" running\n")
    system("python "+name+" "+str(nbseeds))
