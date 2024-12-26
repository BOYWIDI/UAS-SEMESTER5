import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Parameter input
kecepatan_pelayanan = np.arange(0, 11, 1)  # 0-10
kualitas_makanan = np.arange(0, 11, 1)      # 0-10
suasana_restoran = np.arange(0, 11, 1)       # 0-10

# Output: Tingkat Kebahagiaan
tingkat_kebahagiaan = np.arange(0, 11, 1)    # 0-10

# Fuzzy sets untuk kecepatan pelayanan
kecepatan_pelayanan_buruk = fuzz.trapmf(kecepatan_pelayanan, [0, 0, 3, 5])
kecepatan_pelayanan_cukup = fuzz.trimf(kecepatan_pelayanan, [3, 5, 7])
kecepatan_pelayanan_bagus = fuzz.trapmf(kecepatan_pelayanan, [5, 7, 10, 10])

# Fuzzy sets untuk kualitas makanan
kualitas_makanan_buruk = fuzz.trapmf(kualitas_makanan, [0, 0, 3, 5])
kualitas_makanan_cukup = fuzz.trimf(kualitas_makanan, [3, 5, 7])
kualitas_makanan_bagus = fuzz.trapmf(kualitas_makanan, [5, 7, 10, 10])

# Fuzzy sets untuk suasana restoran
suasana_buruk = fuzz.trapmf(suasana_restoran, [0, 0, 3, 5])
suasana_cukup = fuzz.trimf(suasana_restoran, [3, 5, 7])
suasana_bagus = fuzz.trapmf(suasana_restoran, [5, 7, 10, 10])

# Fuzzy sets untuk tingkat kebahagiaan
bahagia = fuzz.trapmf(tingkat_kebahagiaan, [0, 0, 3, 5])
cukup_bahagia = fuzz.trimf(tingkat_kebahagiaan, [3, 5, 7])
tidak_bahagia = fuzz.trapmf(tingkat_kebahagiaan, [5, 7, 10, 10])

# Input dari pengguna
kecepatan_input = 7  # Contoh input
kualitas_input = 8   # Contoh input
suasana_input = 6    # Contoh input

# Fuzzifikasi
kecepatan_buruk_level = fuzz.interp_membership(kecepatan_pelayanan, kecepatan_pelayanan_buruk, kecepatan_input)
kecepatan_cukup_level = fuzz.interp_membership(kecepatan_pelayanan, kecepatan_pelayanan_cukup, kecepatan_input)
kecepatan_bagus_level = fuzz.interp_membership(kecepatan_pelayanan, kecepatan_pelayanan_bagus, kecepatan_input)

kualitas_buruk_level = fuzz.interp_membership(kualitas_makanan, kualitas_makanan_buruk, kualitas_input)
kualitas_cukup_level = fuzz.interp_membership(kualitas_makanan, kualitas_makanan_cukup, kualitas_input)
kualitas_bagus_level = fuzz.interp_membership(kualitas_makanan, kualitas_makanan_bagus, kualitas_input)

suasana_buruk_level = fuzz.interp_membership(suasana_restoran, suasana_buruk, suasana_input)
suasana_cukup_level = fuzz.interp_membership(suasana_restoran, suasana_cukup, suasana_input)
suasana_bagus_level = fuzz.interp_membership(suasana_restoran, suasana_bagus, suasana_input)

# Aturan fuzzy
rule1 = np.fmin(kecepatan_buruk_level, np.fmin(kualitas_buruk_level, suasana_buruk_level))
rule2 = np.fmin(kecepatan_cukup_level, np.fmin(kualitas_cukup_level, suasana_cukup_level))
rule3 = np.fmin