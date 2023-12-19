import pandas as pd
import matplotlib.pyplot as plt

# Data string
data_str1 = """
Epoch: 001, Loss: 1.9490, Train: 0.1786, Val: 0.1140, Test: 0.1120
Epoch: 002, Loss: 1.9324, Train: 0.2857, Val: 0.1360, Test: 0.1350
Epoch: 003, Loss: 1.8998, Train: 0.4857, Val: 0.3500, Test: 0.3200
Epoch: 004, Loss: 1.8409, Train: 0.5357, Val: 0.4620, Test: 0.4580
Epoch: 005, Loss: 1.7682, Train: 0.5643, Val: 0.5140, Test: 0.5150
Epoch: 006, Loss: 1.6548, Train: 0.5500, Val: 0.5240, Test: 0.5260
Epoch: 007, Loss: 1.5561, Train: 0.5571, Val: 0.5180, Test: 0.5310
Epoch: 008, Loss: 1.4872, Train: 0.5643, Val: 0.5220, Test: 0.5370
Epoch: 009, Loss: 1.3994, Train: 0.6000, Val: 0.5420, Test: 0.5530
Epoch: 010, Loss: 1.2918, Train: 0.6643, Val: 0.5720, Test: 0.5790
Epoch: 011, Loss: 1.1752, Train: 0.7714, Val: 0.6280, Test: 0.6180
Epoch: 012, Loss: 1.0875, Train: 0.8214, Val: 0.6520, Test: 0.6530
Epoch: 013, Loss: 1.0064, Train: 0.8786, Val: 0.6960, Test: 0.7050
Epoch: 014, Loss: 0.9350, Train: 0.9357, Val: 0.7180, Test: 0.7230
Epoch: 015, Loss: 0.8784, Train: 0.9643, Val: 0.7480, Test: 0.7430
Epoch: 016, Loss: 0.7940, Train: 0.9643, Val: 0.7680, Test: 0.7600
Epoch: 017, Loss: 0.7115, Train: 0.9714, Val: 0.7840, Test: 0.7720
Epoch: 018, Loss: 0.6202, Train: 0.9786, Val: 0.7920, Test: 0.7870
Epoch: 019, Loss: 0.5637, Train: 0.9786, Val: 0.7920, Test: 0.7940
Epoch: 020, Loss: 0.5229, Train: 0.9786, Val: 0.7960, Test: 0.7920
Epoch: 021, Loss: 0.4496, Train: 0.9786, Val: 0.7960, Test: 0.7910
Epoch: 022, Loss: 0.4068, Train: 0.9786, Val: 0.7920, Test: 0.7900
Epoch: 023, Loss: 0.3795, Train: 0.9786, Val: 0.7840, Test: 0.7860
Epoch: 024, Loss: 0.3389, Train: 0.9857, Val: 0.7800, Test: 0.7800
Epoch: 025, Loss: 0.3023, Train: 0.9857, Val: 0.7800, Test: 0.7820
Epoch: 026, Loss: 0.2586, Train: 0.9857, Val: 0.7780, Test: 0.7860
Epoch: 027, Loss: 0.2649, Train: 0.9929, Val: 0.7760, Test: 0.7850
Epoch: 028, Loss: 0.2599, Train: 0.9929, Val: 0.7780, Test: 0.7910
Epoch: 029, Loss: 0.1725, Train: 0.9929, Val: 0.7860, Test: 0.7900
Epoch: 030, Loss: 0.2260, Train: 0.9929, Val: 0.7840, Test: 0.7920
Epoch: 031, Loss: 0.1688, Train: 0.9929, Val: 0.7900, Test: 0.7940
Epoch: 032, Loss: 0.1882, Train: 0.9929, Val: 0.7860, Test: 0.7950
Epoch: 033, Loss: 0.1399, Train: 0.9929, Val: 0.7820, Test: 0.7920
Epoch: 034, Loss: 0.1207, Train: 0.9929, Val: 0.7840, Test: 0.7940
Epoch: 035, Loss: 0.0954, Train: 1.0000, Val: 0.7760, Test: 0.7880
Epoch: 036, Loss: 0.1331, Train: 1.0000, Val: 0.7740, Test: 0.7890
Epoch: 037, Loss: 0.1088, Train: 1.0000, Val: 0.7740, Test: 0.7850
Epoch: 038, Loss: 0.1125, Train: 1.0000, Val: 0.7700, Test: 0.7880
Epoch: 039, Loss: 0.1275, Train: 1.0000, Val: 0.7700, Test: 0.7860
Epoch: 040, Loss: 0.1213, Train: 1.0000, Val: 0.7680, Test: 0.7870
Epoch: 041, Loss: 0.0897, Train: 1.0000, Val: 0.7680, Test: 0.7880
Epoch: 042, Loss: 0.1014, Train: 1.0000, Val: 0.7660, Test: 0.7890
Epoch: 043, Loss: 0.0738, Train: 1.0000, Val: 0.7720, Test: 0.7900
Epoch: 044, Loss: 0.0962, Train: 1.0000, Val: 0.7680, Test: 0.7910
Epoch: 045, Loss: 0.1150, Train: 1.0000, Val: 0.7700, Test: 0.7900
Epoch: 046, Loss: 0.0897, Train: 1.0000, Val: 0.7720, Test: 0.7920
Epoch: 047, Loss: 0.0730, Train: 1.0000, Val: 0.7760, Test: 0.7940
Epoch: 048, Loss: 0.0671, Train: 1.0000, Val: 0.7800, Test: 0.7970
Epoch: 049, Loss: 0.0775, Train: 1.0000, Val: 0.7780, Test: 0.7960
Epoch: 050, Loss: 0.0773, Train: 1.0000, Val: 0.7820, Test: 0.7960
Epoch: 051, Loss: 0.0575, Train: 1.0000, Val: 0.7760, Test: 0.7970
Epoch: 052, Loss: 0.0792, Train: 1.0000, Val: 0.7760, Test: 0.7970
Epoch: 053, Loss: 0.0712, Train: 1.0000, Val: 0.7740, Test: 0.7970
Epoch: 054, Loss: 0.0621, Train: 1.0000, Val: 0.7720, Test: 0.7990
Epoch: 055, Loss: 0.0762, Train: 1.0000, Val: 0.7720, Test: 0.7970
Epoch: 056, Loss: 0.0510, Train: 1.0000, Val: 0.7760, Test: 0.7970
Epoch: 057, Loss: 0.0396, Train: 1.0000, Val: 0.7740, Test: 0.7970
Epoch: 058, Loss: 0.0524, Train: 1.0000, Val: 0.7740, Test: 0.7970
Epoch: 059, Loss: 0.0560, Train: 1.0000, Val: 0.7740, Test: 0.7990
Epoch: 060, Loss: 0.0536, Train: 1.0000, Val: 0.7720, Test: 0.7980
Epoch: 061, Loss: 0.0534, Train: 1.0000, Val: 0.7720, Test: 0.7980
Epoch: 062, Loss: 0.0770, Train: 1.0000, Val: 0.7680, Test: 0.8000
Epoch: 063, Loss: 0.0614, Train: 1.0000, Val: 0.7680, Test: 0.7990
Epoch: 064, Loss: 0.0635, Train: 1.0000, Val: 0.7680, Test: 0.8000
Epoch: 065, Loss: 0.0601, Train: 1.0000, Val: 0.7680, Test: 0.8010
Epoch: 066, Loss: 0.0483, Train: 1.0000, Val: 0.7700, Test: 0.7980
Epoch: 067, Loss: 0.0613, Train: 1.0000, Val: 0.7720, Test: 0.7990
Epoch: 068, Loss: 0.0514, Train: 1.0000, Val: 0.7740, Test: 0.7950
Epoch: 069, Loss: 0.0710, Train: 1.0000, Val: 0.7760, Test: 0.7970
Epoch: 070, Loss: 0.0425, Train: 1.0000, Val: 0.7760, Test: 0.7960
Epoch: 071, Loss: 0.0646, Train: 1.0000, Val: 0.7760, Test: 0.8000
Epoch: 072, Loss: 0.0661, Train: 1.0000, Val: 0.7740, Test: 0.8020
Epoch: 073, Loss: 0.0486, Train: 1.0000, Val: 0.7740, Test: 0.8030
Epoch: 074, Loss: 0.0529, Train: 1.0000, Val: 0.7740, Test: 0.8020
Epoch: 075, Loss: 0.0528, Train: 1.0000, Val: 0.7760, Test: 0.8030
Epoch: 076, Loss: 0.0696, Train: 1.0000, Val: 0.7760, Test: 0.8040
Epoch: 077, Loss: 0.0657, Train: 1.0000, Val: 0.7760, Test: 0.8020
Epoch: 078, Loss: 0.0587, Train: 1.0000, Val: 0.7740, Test: 0.8010
Epoch: 079, Loss: 0.0528, Train: 1.0000, Val: 0.7700, Test: 0.7980
Epoch: 080, Loss: 0.0451, Train: 1.0000, Val: 0.7680, Test: 0.7970
Epoch: 081, Loss: 0.0578, Train: 1.0000, Val: 0.7680, Test: 0.7970
Epoch: 082, Loss: 0.0559, Train: 1.0000, Val: 0.7720, Test: 0.7960
Epoch: 083, Loss: 0.0665, Train: 1.0000, Val: 0.7700, Test: 0.7970
Epoch: 084, Loss: 0.0531, Train: 1.0000, Val: 0.7700, Test: 0.7950
Epoch: 085, Loss: 0.0493, Train: 1.0000, Val: 0.7700, Test: 0.7950
Epoch: 086, Loss: 0.0612, Train: 1.0000, Val: 0.7700, Test: 0.7940
Epoch: 087, Loss: 0.0430, Train: 1.0000, Val: 0.7700, Test: 0.7920
Epoch: 088, Loss: 0.0401, Train: 1.0000, Val: 0.7700, Test: 0.7920
Epoch: 089, Loss: 0.0531, Train: 1.0000, Val: 0.7680, Test: 0.7930
Epoch: 090, Loss: 0.0457, Train: 1.0000, Val: 0.7680, Test: 0.7950
Epoch: 091, Loss: 0.0633, Train: 1.0000, Val: 0.7700, Test: 0.7970
Epoch: 092, Loss: 0.0465, Train: 1.0000, Val: 0.7700, Test: 0.7980
Epoch: 093, Loss: 0.0404, Train: 1.0000, Val: 0.7700, Test: 0.7990
Epoch: 094, Loss: 0.0456, Train: 1.0000, Val: 0.7700, Test: 0.7990
Epoch: 095, Loss: 0.0500, Train: 1.0000, Val: 0.7680, Test: 0.7990
Epoch: 096, Loss: 0.0447, Train: 1.0000, Val: 0.7660, Test: 0.7990
Epoch: 097, Loss: 0.0593, Train: 1.0000, Val: 0.7680, Test: 0.7980
Epoch: 098, Loss: 0.0455, Train: 1.0000, Val: 0.7680, Test: 0.7960
Epoch: 099, Loss: 0.0348, Train: 1.0000, Val: 0.7640, Test: 0.7960
Epoch: 100, Loss: 0.0474, Train: 1.0000, Val: 0.7680, Test: 0.7980
Epoch: 101, Loss: 0.0342, Train: 1.0000, Val: 0.7700, Test: 0.7970
Epoch: 102, Loss: 0.0637, Train: 1.0000, Val: 0.7700, Test: 0.8000
Epoch: 103, Loss: 0.0415, Train: 1.0000, Val: 0.7680, Test: 0.8020
Epoch: 104, Loss: 0.0567, Train: 1.0000, Val: 0.7660, Test: 0.8020
Epoch: 105, Loss: 0.0556, Train: 1.0000, Val: 0.7660, Test: 0.8040
Epoch: 106, Loss: 0.0450, Train: 1.0000, Val: 0.7680, Test: 0.8010
Epoch: 107, Loss: 0.0575, Train: 1.0000, Val: 0.7700, Test: 0.8000
Epoch: 108, Loss: 0.0294, Train: 1.0000, Val: 0.7720, Test: 0.8000
Epoch: 109, Loss: 0.0497, Train: 1.0000, Val: 0.7700, Test: 0.8000
Epoch: 110, Loss: 0.0423, Train: 1.0000, Val: 0.7680, Test: 0.8010
Epoch: 111, Loss: 0.0416, Train: 1.0000, Val: 0.7680, Test: 0.8010
Epoch: 112, Loss: 0.0479, Train: 1.0000, Val: 0.7680, Test: 0.8010
Epoch: 113, Loss: 0.0462, Train: 1.0000, Val: 0.7760, Test: 0.8030
Epoch: 114, Loss: 0.0446, Train: 1.0000, Val: 0.7760, Test: 0.8040
Epoch: 115, Loss: 0.0442, Train: 1.0000, Val: 0.7780, Test: 0.8030
Epoch: 116, Loss: 0.0653, Train: 1.0000, Val: 0.7780, Test: 0.8020
Epoch: 117, Loss: 0.0445, Train: 1.0000, Val: 0.7760, Test: 0.8000
Epoch: 118, Loss: 0.0718, Train: 1.0000, Val: 0.7780, Test: 0.8020
Epoch: 119, Loss: 0.0407, Train: 1.0000, Val: 0.7760, Test: 0.8040
Epoch: 120, Loss: 0.0577, Train: 1.0000, Val: 0.7780, Test: 0.8040
Epoch: 121, Loss: 0.0603, Train: 1.0000, Val: 0.7760, Test: 0.8020
Epoch: 122, Loss: 0.0516, Train: 1.0000, Val: 0.7740, Test: 0.8030
Epoch: 123, Loss: 0.0355, Train: 1.0000, Val: 0.7740, Test: 0.8000
Epoch: 124, Loss: 0.0261, Train: 1.0000, Val: 0.7720, Test: 0.7990
Epoch: 125, Loss: 0.0658, Train: 1.0000, Val: 0.7660, Test: 0.7990
Epoch: 126, Loss: 0.0511, Train: 1.0000, Val: 0.7620, Test: 0.7960
Epoch: 127, Loss: 0.0416, Train: 1.0000, Val: 0.7580, Test: 0.7950
Epoch: 128, Loss: 0.0550, Train: 1.0000, Val: 0.7580, Test: 0.7950
Epoch: 129, Loss: 0.0581, Train: 1.0000, Val: 0.7640, Test: 0.7940
Epoch: 130, Loss: 0.0513, Train: 1.0000, Val: 0.7640, Test: 0.7980
Epoch: 131, Loss: 0.0625, Train: 1.0000, Val: 0.7640, Test: 0.7990
Epoch: 132, Loss: 0.0510, Train: 1.0000, Val: 0.7640, Test: 0.7980
Epoch: 133, Loss: 0.0365, Train: 1.0000, Val: 0.7640, Test: 0.8010
Epoch: 134, Loss: 0.0398, Train: 1.0000, Val: 0.7640, Test: 0.7990
Epoch: 135, Loss: 0.0499, Train: 1.0000, Val: 0.7640, Test: 0.7980
Epoch: 136, Loss: 0.0428, Train: 1.0000, Val: 0.7620, Test: 0.7970
Epoch: 137, Loss: 0.0562, Train: 1.0000, Val: 0.7580, Test: 0.7970
Epoch: 138, Loss: 0.0457, Train: 1.0000, Val: 0.7600, Test: 0.7940
Epoch: 139, Loss: 0.0367, Train: 1.0000, Val: 0.7600, Test: 0.7930
Epoch: 140, Loss: 0.0328, Train: 1.0000, Val: 0.7640, Test: 0.7920
Epoch: 141, Loss: 0.0371, Train: 1.0000, Val: 0.7660, Test: 0.7920
Epoch: 142, Loss: 0.0398, Train: 1.0000, Val: 0.7660, Test: 0.7930
Epoch: 143, Loss: 0.0438, Train: 1.0000, Val: 0.7660, Test: 0.7920
Epoch: 144, Loss: 0.0420, Train: 1.0000, Val: 0.7680, Test: 0.7930
Epoch: 145, Loss: 0.0502, Train: 1.0000, Val: 0.7640, Test: 0.7950
Epoch: 146, Loss: 0.0495, Train: 1.0000, Val: 0.7680, Test: 0.7960
Epoch: 147, Loss: 0.0481, Train: 1.0000, Val: 0.7700, Test: 0.7960
Epoch: 148, Loss: 0.0328, Train: 1.0000, Val: 0.7700, Test: 0.7970
Epoch: 149, Loss: 0.0349, Train: 1.0000, Val: 0.7720, Test: 0.8000
Epoch: 150, Loss: 0.0457, Train: 1.0000, Val: 0.7720, Test: 0.8040
Epoch: 151, Loss: 0.0393, Train: 1.0000, Val: 0.7720, Test: 0.8030
Epoch: 152, Loss: 0.0392, Train: 1.0000, Val: 0.7740, Test: 0.8020
Epoch: 153, Loss: 0.0521, Train: 1.0000, Val: 0.7720, Test: 0.7990
Epoch: 154, Loss: 0.0433, Train: 1.0000, Val: 0.7700, Test: 0.8020
Epoch: 155, Loss: 0.0371, Train: 1.0000, Val: 0.7660, Test: 0.8050
Epoch: 156, Loss: 0.0454, Train: 1.0000, Val: 0.7640, Test: 0.8030
Epoch: 157, Loss: 0.0503, Train: 1.0000, Val: 0.7620, Test: 0.8030
Epoch: 158, Loss: 0.0493, Train: 1.0000, Val: 0.7620, Test: 0.8040
Epoch: 159, Loss: 0.0369, Train: 1.0000, Val: 0.7620, Test: 0.8030
Epoch: 160, Loss: 0.0479, Train: 1.0000, Val: 0.7620, Test: 0.8040
Epoch: 161, Loss: 0.0371, Train: 1.0000, Val: 0.7620, Test: 0.8030
Epoch: 162, Loss: 0.0412, Train: 1.0000, Val: 0.7640, Test: 0.8050
Epoch: 163, Loss: 0.0396, Train: 1.0000, Val: 0.7600, Test: 0.8060
Epoch: 164, Loss: 0.0425, Train: 1.0000, Val: 0.7640, Test: 0.8040
Epoch: 165, Loss: 0.0298, Train: 1.0000, Val: 0.7660, Test: 0.8020
Epoch: 166, Loss: 0.0506, Train: 1.0000, Val: 0.7660, Test: 0.8030
Epoch: 167, Loss: 0.0325, Train: 1.0000, Val: 0.7640, Test: 0.8030
Epoch: 168, Loss: 0.0304, Train: 1.0000, Val: 0.7640, Test: 0.8050
Epoch: 169, Loss: 0.0328, Train: 1.0000, Val: 0.7660, Test: 0.8040
Epoch: 170, Loss: 0.0328, Train: 1.0000, Val: 0.7660, Test: 0.8050
Epoch: 171, Loss: 0.0328, Train: 1.0000, Val: 0.7640, Test: 0.8060
Epoch: 172, Loss: 0.0467, Train: 1.0000, Val: 0.7700, Test: 0.8080
Epoch: 173, Loss: 0.0353, Train: 1.0000, Val: 0.7740, Test: 0.8080
Epoch: 174, Loss: 0.0326, Train: 1.0000, Val: 0.7740, Test: 0.8040
Epoch: 175, Loss: 0.0469, Train: 1.0000, Val: 0.7720, Test: 0.8020
Epoch: 176, Loss: 0.0486, Train: 1.0000, Val: 0.7700, Test: 0.8010
Epoch: 177, Loss: 0.0312, Train: 1.0000, Val: 0.7680, Test: 0.8020
Epoch: 178, Loss: 0.0262, Train: 1.0000, Val: 0.7680, Test: 0.8000
Epoch: 179, Loss: 0.0472, Train: 1.0000, Val: 0.7680, Test: 0.8000
Epoch: 180, Loss: 0.0456, Train: 1.0000, Val: 0.7680, Test: 0.8000
Epoch: 181, Loss: 0.0378, Train: 1.0000, Val: 0.7720, Test: 0.8010
Epoch: 182, Loss: 0.0284, Train: 1.0000, Val: 0.7740, Test: 0.8010
Epoch: 183, Loss: 0.0312, Train: 1.0000, Val: 0.7720, Test: 0.8010
Epoch: 184, Loss: 0.0479, Train: 1.0000, Val: 0.7740, Test: 0.8050
Epoch: 185, Loss: 0.0288, Train: 1.0000, Val: 0.7740, Test: 0.8070
Epoch: 186, Loss: 0.0364, Train: 1.0000, Val: 0.7780, Test: 0.8100
Epoch: 187, Loss: 0.0294, Train: 1.0000, Val: 0.7820, Test: 0.8110
Epoch: 188, Loss: 0.0613, Train: 1.0000, Val: 0.7840, Test: 0.8100
Epoch: 189, Loss: 0.0260, Train: 1.0000, Val: 0.7840, Test: 0.8110
Epoch: 190, Loss: 0.0308, Train: 1.0000, Val: 0.7820, Test: 0.8110
Epoch: 191, Loss: 0.0403, Train: 1.0000, Val: 0.7800, Test: 0.8080
Epoch: 192, Loss: 0.0281, Train: 1.0000, Val: 0.7760, Test: 0.8080
Epoch: 193, Loss: 0.0362, Train: 1.0000, Val: 0.7780, Test: 0.8080
Epoch: 194, Loss: 0.0360, Train: 1.0000, Val: 0.7800, Test: 0.8100
Epoch: 195, Loss: 0.0462, Train: 1.0000, Val: 0.7720, Test: 0.8110
Epoch: 196, Loss: 0.0321, Train: 1.0000, Val: 0.7680, Test: 0.8120
Epoch: 197, Loss: 0.0374, Train: 1.0000, Val: 0.7660, Test: 0.8060
Epoch: 198, Loss: 0.0376, Train: 1.0000, Val: 0.7660, Test: 0.8070
Epoch: 199, Loss: 0.0377, Train: 1.0000, Val: 0.7620, Test: 0.8070
Epoch: 200, Loss: 0.0380, Train: 1.0000, Val: 0.7620, Test: 0.8060
"""

# data_str2 = """
# Epoch: 001, Loss: 1.9463, Train: 0.1929, Val: 0.1620, Test: 0.1530
# Epoch: 002, Loss: 1.9440, Train: 0.2357, Val: 0.1740, Test: 0.1630
# Epoch: 003, Loss: 1.9407, Train: 0.2357, Val: 0.1780, Test: 0.1630
# Epoch: 004, Loss: 1.9365, Train: 0.2500, Val: 0.1920, Test: 0.1760
# Epoch: 005, Loss: 1.9316, Train: 0.2643, Val: 0.1980, Test: 0.1810
# Epoch: 006, Loss: 1.9255, Train: 0.2929, Val: 0.2060, Test: 0.1920
# Epoch: 007, Loss: 1.9211, Train: 0.5643, Val: 0.2840, Test: 0.2760
# Epoch: 008, Loss: 1.9157, Train: 0.7786, Val: 0.4020, Test: 0.4080
# Epoch: 009, Loss: 1.9094, Train: 0.8571, Val: 0.5060, Test: 0.5070
# Epoch: 010, Loss: 1.9006, Train: 0.9143, Val: 0.5960, Test: 0.5910
# Epoch: 011, Loss: 1.8943, Train: 0.9214, Val: 0.6360, Test: 0.6470
# Epoch: 012, Loss: 1.8849, Train: 0.9214, Val: 0.6500, Test: 0.6830
# Epoch: 013, Loss: 1.8745, Train: 0.9071, Val: 0.6860, Test: 0.7110
# Epoch: 014, Loss: 1.8643, Train: 0.9000, Val: 0.7100, Test: 0.7250
# Epoch: 015, Loss: 1.8559, Train: 0.9000, Val: 0.7280, Test: 0.7410
# Epoch: 016, Loss: 1.8463, Train: 0.9071, Val: 0.7320, Test: 0.7500
# Epoch: 017, Loss: 1.8327, Train: 0.9071, Val: 0.7460, Test: 0.7620
# Epoch: 018, Loss: 1.8191, Train: 0.9214, Val: 0.7540, Test: 0.7660
# Epoch: 019, Loss: 1.8114, Train: 0.9214, Val: 0.7540, Test: 0.7730
# Epoch: 020, Loss: 1.7950, Train: 0.9214, Val: 0.7520, Test: 0.7710
# Epoch: 021, Loss: 1.7908, Train: 0.9214, Val: 0.7500, Test: 0.7720
# Epoch: 022, Loss: 1.7712, Train: 0.9214, Val: 0.7540, Test: 0.7730
# Epoch: 023, Loss: 1.7473, Train: 0.9143, Val: 0.7500, Test: 0.7700
# Epoch: 024, Loss: 1.7305, Train: 0.9214, Val: 0.7520, Test: 0.7680
# Epoch: 025, Loss: 1.7222, Train: 0.9143, Val: 0.7520, Test: 0.7670
# Epoch: 026, Loss: 1.7061, Train: 0.9214, Val: 0.7560, Test: 0.7710
# Epoch: 027, Loss: 1.6972, Train: 0.9214, Val: 0.7520, Test: 0.7670
# Epoch: 028, Loss: 1.6652, Train: 0.9214, Val: 0.7520, Test: 0.7680
# Epoch: 029, Loss: 1.6357, Train: 0.9214, Val: 0.7500, Test: 0.7690
# Epoch: 030, Loss: 1.6348, Train: 0.9143, Val: 0.7540, Test: 0.7710
# Epoch: 031, Loss: 1.6113, Train: 0.9357, Val: 0.7520, Test: 0.7750
# Epoch: 032, Loss: 1.5916, Train: 0.9071, Val: 0.7460, Test: 0.7740
# Epoch: 033, Loss: 1.5856, Train: 0.9214, Val: 0.7480, Test: 0.7720
# Epoch: 034, Loss: 1.5529, Train: 0.9214, Val: 0.7480, Test: 0.7700
# Epoch: 035, Loss: 1.5244, Train: 0.9286, Val: 0.7520, Test: 0.7720
# Epoch: 036, Loss: 1.5299, Train: 0.9286, Val: 0.7540, Test: 0.7700
# Epoch: 037, Loss: 1.4914, Train: 0.9286, Val: 0.7540, Test: 0.7720
# Epoch: 038, Loss: 1.4690, Train: 0.9286, Val: 0.7560, Test: 0.7680
# Epoch: 039, Loss: 1.4409, Train: 0.9286, Val: 0.7500, Test: 0.7690
# Epoch: 040, Loss: 1.4178, Train: 0.9357, Val: 0.7480, Test: 0.7680
# Epoch: 041, Loss: 1.4097, Train: 0.9357, Val: 0.7540, Test: 0.7690
# Epoch: 042, Loss: 1.3763, Train: 0.9357, Val: 0.7520, Test: 0.7730
# Epoch: 043, Loss: 1.3462, Train: 0.9286, Val: 0.7520, Test: 0.7700
# Epoch: 044, Loss: 1.3276, Train: 0.9357, Val: 0.7520, Test: 0.7720
# Epoch: 045, Loss: 1.2998, Train: 0.9357, Val: 0.7500, Test: 0.7700
# Epoch: 046, Loss: 1.2926, Train: 0.9357, Val: 0.7460, Test: 0.7720
# Epoch: 047, Loss: 1.2437, Train: 0.9429, Val: 0.7500, Test: 0.7740
# Epoch: 048, Loss: 1.2428, Train: 0.9429, Val: 0.7480, Test: 0.7780
# Epoch: 049, Loss: 1.2088, Train: 0.9500, Val: 0.7440, Test: 0.7770
# Epoch: 050, Loss: 1.1826, Train: 0.9500, Val: 0.7500, Test: 0.7780
# Epoch: 051, Loss: 1.1886, Train: 0.9500, Val: 0.7460, Test: 0.7790
# Epoch: 052, Loss: 1.1052, Train: 0.9500, Val: 0.7480, Test: 0.7780
# Epoch: 053, Loss: 1.1092, Train: 0.9500, Val: 0.7520, Test: 0.7770
# Epoch: 054, Loss: 1.1163, Train: 0.9571, Val: 0.7540, Test: 0.7770
# Epoch: 055, Loss: 1.0813, Train: 0.9571, Val: 0.7540, Test: 0.7800
# Epoch: 056, Loss: 1.0929, Train: 0.9571, Val: 0.7560, Test: 0.7810
# Epoch: 057, Loss: 1.0357, Train: 0.9571, Val: 0.7560, Test: 0.7800
# Epoch: 058, Loss: 1.0003, Train: 0.9643, Val: 0.7600, Test: 0.7810
# Epoch: 059, Loss: 1.0337, Train: 0.9643, Val: 0.7660, Test: 0.7820
# Epoch: 060, Loss: 0.9583, Train: 0.9643, Val: 0.7680, Test: 0.7880
# Epoch: 061, Loss: 0.9452, Train: 0.9643, Val: 0.7680, Test: 0.7870
# Epoch: 062, Loss: 0.9078, Train: 0.9714, Val: 0.7700, Test: 0.7900
# Epoch: 063, Loss: 0.8896, Train: 0.9714, Val: 0.7720, Test: 0.7920
# Epoch: 064, Loss: 0.8908, Train: 0.9714, Val: 0.7720, Test: 0.7930
# Epoch: 065, Loss: 0.8624, Train: 0.9786, Val: 0.7760, Test: 0.7970
# Epoch: 066, Loss: 0.8384, Train: 0.9786, Val: 0.7700, Test: 0.7970
# Epoch: 067, Loss: 0.8195, Train: 0.9786, Val: 0.7700, Test: 0.7960
# Epoch: 068, Loss: 0.8433, Train: 0.9786, Val: 0.7700, Test: 0.7950
# Epoch: 069, Loss: 0.8160, Train: 0.9786, Val: 0.7740, Test: 0.7940
# Epoch: 070, Loss: 0.7907, Train: 0.9786, Val: 0.7740, Test: 0.7980
# Epoch: 071, Loss: 0.8121, Train: 0.9786, Val: 0.7740, Test: 0.7950
# Epoch: 072, Loss: 0.7538, Train: 0.9786, Val: 0.7700, Test: 0.7960
# Epoch: 073, Loss: 0.7914, Train: 0.9786, Val: 0.7700, Test: 0.7980
# Epoch: 074, Loss: 0.7756, Train: 0.9786, Val: 0.7700, Test: 0.8010
# Epoch: 075, Loss: 0.7349, Train: 0.9786, Val: 0.7740, Test: 0.8020
# Epoch: 076, Loss: 0.7415, Train: 0.9786, Val: 0.7720, Test: 0.8010
# Epoch: 077, Loss: 0.7513, Train: 0.9857, Val: 0.7720, Test: 0.8000
# Epoch: 078, Loss: 0.7318, Train: 0.9857, Val: 0.7720, Test: 0.7990
# Epoch: 079, Loss: 0.7108, Train: 0.9857, Val: 0.7680, Test: 0.7980
# Epoch: 080, Loss: 0.6983, Train: 0.9857, Val: 0.7620, Test: 0.8010
# Epoch: 081, Loss: 0.6899, Train: 0.9857, Val: 0.7680, Test: 0.8000
# Epoch: 082, Loss: 0.6199, Train: 0.9857, Val: 0.7700, Test: 0.8030
# Epoch: 083, Loss: 0.6632, Train: 0.9857, Val: 0.7720, Test: 0.8040
# Epoch: 084, Loss: 0.6443, Train: 0.9857, Val: 0.7720, Test: 0.8040
# Epoch: 085, Loss: 0.6132, Train: 0.9857, Val: 0.7740, Test: 0.8050
# Epoch: 086, Loss: 0.6467, Train: 0.9857, Val: 0.7720, Test: 0.8040
# Epoch: 087, Loss: 0.6704, Train: 0.9857, Val: 0.7740, Test: 0.8020
# Epoch: 088, Loss: 0.6256, Train: 0.9857, Val: 0.7720, Test: 0.8010
# Epoch: 089, Loss: 0.5616, Train: 0.9857, Val: 0.7700, Test: 0.8030
# Epoch: 090, Loss: 0.5856, Train: 0.9857, Val: 0.7700, Test: 0.8010
# Epoch: 091, Loss: 0.6199, Train: 0.9857, Val: 0.7700, Test: 0.8030
# Epoch: 092, Loss: 0.6251, Train: 0.9857, Val: 0.7720, Test: 0.8020
# Epoch: 093, Loss: 0.5904, Train: 0.9786, Val: 0.7700, Test: 0.8030
# Epoch: 094, Loss: 0.5283, Train: 0.9857, Val: 0.7720, Test: 0.8030
# Epoch: 095, Loss: 0.5358, Train: 0.9857, Val: 0.7720, Test: 0.8050
# Epoch: 096, Loss: 0.5700, Train: 0.9857, Val: 0.7760, Test: 0.8060
# Epoch: 097, Loss: 0.5588, Train: 0.9929, Val: 0.7760, Test: 0.8050
# Epoch: 098, Loss: 0.5915, Train: 0.9857, Val: 0.7760, Test: 0.8090
# Epoch: 099, Loss: 0.5312, Train: 0.9929, Val: 0.7760, Test: 0.8140
# Epoch: 100, Loss: 0.5671, Train: 0.9929, Val: 0.7760, Test: 0.8120
# Epoch: 101, Loss: 0.5290, Train: 0.9929, Val: 0.7740, Test: 0.8110
# Epoch: 102, Loss: 0.5528, Train: 0.9929, Val: 0.7740, Test: 0.8090
# Epoch: 103, Loss: 0.5320, Train: 0.9929, Val: 0.7740, Test: 0.8090
# Epoch: 104, Loss: 0.4919, Train: 0.9929, Val: 0.7740, Test: 0.8090
# Epoch: 105, Loss: 0.5370, Train: 0.9929, Val: 0.7740, Test: 0.8080
# Epoch: 106, Loss: 0.4916, Train: 0.9929, Val: 0.7700, Test: 0.8080
# Epoch: 107, Loss: 0.5212, Train: 0.9929, Val: 0.7740, Test: 0.8110
# Epoch: 108, Loss: 0.5208, Train: 0.9929, Val: 0.7720, Test: 0.8090
# Epoch: 109, Loss: 0.4952, Train: 0.9929, Val: 0.7680, Test: 0.8110
# Epoch: 110, Loss: 0.4721, Train: 0.9929, Val: 0.7720, Test: 0.8130
# Epoch: 111, Loss: 0.4914, Train: 0.9929, Val: 0.7720, Test: 0.8120
# Epoch: 112, Loss: 0.5093, Train: 0.9929, Val: 0.7740, Test: 0.8130
# Epoch: 113, Loss: 0.4784, Train: 0.9929, Val: 0.7740, Test: 0.8110
# Epoch: 114, Loss: 0.4521, Train: 0.9929, Val: 0.7780, Test: 0.8140
# Epoch: 115, Loss: 0.4619, Train: 0.9929, Val: 0.7840, Test: 0.8130
# Epoch: 116, Loss: 0.4684, Train: 0.9929, Val: 0.7780, Test: 0.8130
# Epoch: 117, Loss: 0.4853, Train: 0.9929, Val: 0.7780, Test: 0.8160
# Epoch: 118, Loss: 0.4369, Train: 0.9929, Val: 0.7740, Test: 0.8150
# Epoch: 119, Loss: 0.4733, Train: 0.9929, Val: 0.7740, Test: 0.8140
# Epoch: 120, Loss: 0.4587, Train: 0.9929, Val: 0.7740, Test: 0.8130
# Epoch: 121, Loss: 0.4906, Train: 0.9929, Val: 0.7740, Test: 0.8140
# Epoch: 122, Loss: 0.4591, Train: 0.9929, Val: 0.7800, Test: 0.8140
# Epoch: 123, Loss: 0.4561, Train: 0.9929, Val: 0.7780, Test: 0.8160
# Epoch: 124, Loss: 0.4384, Train: 0.9929, Val: 0.7780, Test: 0.8140
# Epoch: 125, Loss: 0.4276, Train: 0.9929, Val: 0.7800, Test: 0.8140
# Epoch: 126, Loss: 0.4211, Train: 0.9929, Val: 0.7800, Test: 0.8180
# Epoch: 127, Loss: 0.4482, Train: 0.9929, Val: 0.7800, Test: 0.8170
# Epoch: 128, Loss: 0.4679, Train: 0.9929, Val: 0.7780, Test: 0.8150
# Epoch: 129, Loss: 0.4519, Train: 0.9929, Val: 0.7780, Test: 0.8150
# Epoch: 130, Loss: 0.4411, Train: 0.9929, Val: 0.7780, Test: 0.8170
# Epoch: 131, Loss: 0.4462, Train: 0.9929, Val: 0.7860, Test: 0.8190
# Epoch: 132, Loss: 0.4163, Train: 0.9929, Val: 0.7840, Test: 0.8210
# Epoch: 133, Loss: 0.4154, Train: 0.9929, Val: 0.7840, Test: 0.8200
# Epoch: 134, Loss: 0.3948, Train: 0.9929, Val: 0.7840, Test: 0.8230
# Epoch: 135, Loss: 0.3991, Train: 0.9929, Val: 0.7840, Test: 0.8200
# Epoch: 136, Loss: 0.4192, Train: 0.9929, Val: 0.7840, Test: 0.8210
# Epoch: 137, Loss: 0.3847, Train: 1.0000, Val: 0.7820, Test: 0.8220
# Epoch: 138, Loss: 0.4190, Train: 1.0000, Val: 0.7840, Test: 0.8220
# Epoch: 139, Loss: 0.4740, Train: 1.0000, Val: 0.7840, Test: 0.8200
# Epoch: 140, Loss: 0.4186, Train: 1.0000, Val: 0.7820, Test: 0.8190
# Epoch: 141, Loss: 0.4051, Train: 0.9929, Val: 0.7840, Test: 0.8180
# Epoch: 142, Loss: 0.3906, Train: 1.0000, Val: 0.7780, Test: 0.8140
# Epoch: 143, Loss: 0.3981, Train: 0.9929, Val: 0.7760, Test: 0.8120
# Epoch: 144, Loss: 0.4450, Train: 0.9929, Val: 0.7760, Test: 0.8140
# Epoch: 145, Loss: 0.4068, Train: 0.9929, Val: 0.7780, Test: 0.8160
# Epoch: 146, Loss: 0.3585, Train: 0.9929, Val: 0.7780, Test: 0.8160
# Epoch: 147, Loss: 0.3681, Train: 0.9929, Val: 0.7820, Test: 0.8140
# Epoch: 148, Loss: 0.4031, Train: 0.9929, Val: 0.7800, Test: 0.8150
# Epoch: 149, Loss: 0.3797, Train: 0.9929, Val: 0.7780, Test: 0.8140
# Epoch: 150, Loss: 0.3920, Train: 0.9929, Val: 0.7820, Test: 0.8150
# Epoch: 151, Loss: 0.3784, Train: 0.9929, Val: 0.7840, Test: 0.8170
# Epoch: 152, Loss: 0.3887, Train: 0.9929, Val: 0.7840, Test: 0.8150
# Epoch: 153, Loss: 0.3596, Train: 0.9929, Val: 0.7840, Test: 0.8140
# Epoch: 154, Loss: 0.3714, Train: 0.9929, Val: 0.7780, Test: 0.8140
# Epoch: 155, Loss: 0.3496, Train: 1.0000, Val: 0.7780, Test: 0.8130
# Epoch: 156, Loss: 0.3982, Train: 1.0000, Val: 0.7820, Test: 0.8140
# Epoch: 157, Loss: 0.3700, Train: 1.0000, Val: 0.7780, Test: 0.8130
# Epoch: 158, Loss: 0.3608, Train: 1.0000, Val: 0.7780, Test: 0.8090
# Epoch: 159, Loss: 0.3698, Train: 1.0000, Val: 0.7800, Test: 0.8140
# Epoch: 160, Loss: 0.3914, Train: 1.0000, Val: 0.7800, Test: 0.8090
# Epoch: 161, Loss: 0.3757, Train: 1.0000, Val: 0.7800, Test: 0.8120
# Epoch: 162, Loss: 0.3805, Train: 1.0000, Val: 0.7780, Test: 0.8120
# Epoch: 163, Loss: 0.3597, Train: 0.9929, Val: 0.7800, Test: 0.8110
# Epoch: 164, Loss: 0.3747, Train: 1.0000, Val: 0.7760, Test: 0.8140
# Epoch: 165, Loss: 0.3525, Train: 1.0000, Val: 0.7760, Test: 0.8120
# Epoch: 166, Loss: 0.3738, Train: 1.0000, Val: 0.7800, Test: 0.8140
# Epoch: 167, Loss: 0.3589, Train: 1.0000, Val: 0.7800, Test: 0.8130
# Epoch: 168, Loss: 0.3423, Train: 1.0000, Val: 0.7800, Test: 0.8150
# Epoch: 169, Loss: 0.3313, Train: 1.0000, Val: 0.7800, Test: 0.8120
# Epoch: 170, Loss: 0.3451, Train: 1.0000, Val: 0.7780, Test: 0.8150
# Epoch: 171, Loss: 0.3497, Train: 1.0000, Val: 0.7760, Test: 0.8150
# Epoch: 172, Loss: 0.3431, Train: 1.0000, Val: 0.7800, Test: 0.8140
# Epoch: 173, Loss: 0.3653, Train: 1.0000, Val: 0.7800, Test: 0.8170
# Epoch: 174, Loss: 0.3558, Train: 1.0000, Val: 0.7800, Test: 0.8170
# Epoch: 175, Loss: 0.3279, Train: 1.0000, Val: 0.7800, Test: 0.8170
# Epoch: 176, Loss: 0.3488, Train: 1.0000, Val: 0.7800, Test: 0.8170
# Epoch: 177, Loss: 0.3593, Train: 1.0000, Val: 0.7800, Test: 0.8180
# Epoch: 178, Loss: 0.3497, Train: 1.0000, Val: 0.7820, Test: 0.8220
# Epoch: 179, Loss: 0.3288, Train: 1.0000, Val: 0.7800, Test: 0.8180
# Epoch: 180, Loss: 0.3461, Train: 1.0000, Val: 0.7820, Test: 0.8210
# Epoch: 181, Loss: 0.3759, Train: 1.0000, Val: 0.7820, Test: 0.8170
# Epoch: 182, Loss: 0.3698, Train: 1.0000, Val: 0.7820, Test: 0.8190
# Epoch: 183, Loss: 0.3340, Train: 1.0000, Val: 0.7820, Test: 0.8190
# Epoch: 184, Loss: 0.3254, Train: 1.0000, Val: 0.7820, Test: 0.8180
# Epoch: 185, Loss: 0.3583, Train: 1.0000, Val: 0.7820, Test: 0.8150
# Epoch: 186, Loss: 0.3179, Train: 1.0000, Val: 0.7780, Test: 0.8140
# Epoch: 187, Loss: 0.3043, Train: 1.0000, Val: 0.7780, Test: 0.8140
# Epoch: 188, Loss: 0.3393, Train: 1.0000, Val: 0.7780, Test: 0.8140
# Epoch: 189, Loss: 0.3055, Train: 1.0000, Val: 0.7780, Test: 0.8160
# Epoch: 190, Loss: 0.3002, Train: 1.0000, Val: 0.7760, Test: 0.8170
# Epoch: 191, Loss: 0.3060, Train: 1.0000, Val: 0.7760, Test: 0.8170
# Epoch: 192, Loss: 0.3500, Train: 1.0000, Val: 0.7800, Test: 0.8190
# Epoch: 193, Loss: 0.3083, Train: 1.0000, Val: 0.7800, Test: 0.8190
# Epoch: 194, Loss: 0.3455, Train: 1.0000, Val: 0.7820, Test: 0.8230
# Epoch: 195, Loss: 0.3209, Train: 1.0000, Val: 0.7820, Test: 0.8210
# Epoch: 196, Loss: 0.3328, Train: 1.0000, Val: 0.7820, Test: 0.8210
# Epoch: 197, Loss: 0.3240, Train: 1.0000, Val: 0.7820, Test: 0.8210
# Epoch: 198, Loss: 0.3189, Train: 1.0000, Val: 0.7800, Test: 0.8170
# Epoch: 199, Loss: 0.3397, Train: 1.0000, Val: 0.7800, Test: 0.8190
# Epoch: 200, Loss: 0.3231, Train: 1.0000, Val: 0.7820, Test: 0.8180
# """

data_str2 = """
Epoch: 001, Loss: 1.9991, Train: 0.3571, Val: 0.2560, Test: 0.2440
Epoch: 002, Loss: 1.7281, Train: 0.7357, Val: 0.5220, Test: 0.5530
Epoch: 003, Loss: 1.6365, Train: 0.8786, Val: 0.6680, Test: 0.6740
Epoch: 004, Loss: 1.5585, Train: 0.9643, Val: 0.7220, Test: 0.7390
Epoch: 005, Loss: 1.5184, Train: 0.9643, Val: 0.7320, Test: 0.7520
Epoch: 006, Loss: 1.4645, Train: 0.9786, Val: 0.7400, Test: 0.7570
Epoch: 007, Loss: 1.4451, Train: 0.9929, Val: 0.7640, Test: 0.7860
Epoch: 008, Loss: 1.4359, Train: 0.9929, Val: 0.7820, Test: 0.7920
Epoch: 009, Loss: 1.4154, Train: 0.9929, Val: 0.7840, Test: 0.8020
Epoch: 010, Loss: 1.3913, Train: 0.9929, Val: 0.7840, Test: 0.8000
Epoch: 011, Loss: 1.3748, Train: 0.9929, Val: 0.7860, Test: 0.7930
Epoch: 012, Loss: 1.3667, Train: 0.9929, Val: 0.7880, Test: 0.7880
Epoch: 013, Loss: 1.3913, Train: 0.9929, Val: 0.7780, Test: 0.7860
Epoch: 014, Loss: 1.3608, Train: 0.9929, Val: 0.7740, Test: 0.7840
Epoch: 015, Loss: 1.3493, Train: 0.9929, Val: 0.7740, Test: 0.7820
Epoch: 016, Loss: 1.3334, Train: 0.9929, Val: 0.7700, Test: 0.7810
Epoch: 017, Loss: 1.3301, Train: 0.9929, Val: 0.7660, Test: 0.7850
Epoch: 018, Loss: 1.3409, Train: 0.9929, Val: 0.7780, Test: 0.7840
Epoch: 019, Loss: 1.3382, Train: 0.9929, Val: 0.7720, Test: 0.7860
Epoch: 020, Loss: 1.3304, Train: 0.9929, Val: 0.7780, Test: 0.7860
Epoch: 021, Loss: 1.3177, Train: 0.9929, Val: 0.7760, Test: 0.7870
Epoch: 022, Loss: 1.3180, Train: 0.9929, Val: 0.7760, Test: 0.7800
Epoch: 023, Loss: 1.3053, Train: 0.9929, Val: 0.7720, Test: 0.7820
Epoch: 024, Loss: 1.3128, Train: 0.9929, Val: 0.7660, Test: 0.7770
Epoch: 025, Loss: 1.3083, Train: 0.9929, Val: 0.7700, Test: 0.7750
Epoch: 026, Loss: 1.3110, Train: 0.9929, Val: 0.7660, Test: 0.7730
Epoch: 027, Loss: 1.3150, Train: 0.9929, Val: 0.7660, Test: 0.7710
Epoch: 028, Loss: 1.3344, Train: 0.9929, Val: 0.7640, Test: 0.7620
Epoch: 029, Loss: 1.3135, Train: 1.0000, Val: 0.7620, Test: 0.7570
Epoch: 030, Loss: 1.3023, Train: 1.0000, Val: 0.7560, Test: 0.7580
Epoch: 031, Loss: 1.2897, Train: 1.0000, Val: 0.7480, Test: 0.7610
Epoch: 032, Loss: 1.2991, Train: 1.0000, Val: 0.7440, Test: 0.7590
Epoch: 033, Loss: 1.2929, Train: 1.0000, Val: 0.7400, Test: 0.7650
Epoch: 034, Loss: 1.2935, Train: 1.0000, Val: 0.7320, Test: 0.7610
Epoch: 035, Loss: 1.2880, Train: 1.0000, Val: 0.7340, Test: 0.7610
Epoch: 036, Loss: 1.2842, Train: 1.0000, Val: 0.7320, Test: 0.7620
Epoch: 037, Loss: 1.2917, Train: 1.0000, Val: 0.7300, Test: 0.7590
Epoch: 038, Loss: 1.2814, Train: 1.0000, Val: 0.7320, Test: 0.7560
Epoch: 039, Loss: 1.2813, Train: 1.0000, Val: 0.7240, Test: 0.7570
Epoch: 040, Loss: 1.2784, Train: 1.0000, Val: 0.7300, Test: 0.7520
Epoch: 041, Loss: 1.2842, Train: 1.0000, Val: 0.7280, Test: 0.7530
Epoch: 042, Loss: 1.2753, Train: 1.0000, Val: 0.7280, Test: 0.7550
Epoch: 043, Loss: 1.2902, Train: 1.0000, Val: 0.7280, Test: 0.7520
Epoch: 044, Loss: 1.2793, Train: 1.0000, Val: 0.7280, Test: 0.7500
Epoch: 045, Loss: 1.2863, Train: 1.0000, Val: 0.7280, Test: 0.7540
Epoch: 046, Loss: 1.2557, Train: 1.0000, Val: 0.7240, Test: 0.7540
Epoch: 047, Loss: 1.2736, Train: 1.0000, Val: 0.7240, Test: 0.7510
Epoch: 048, Loss: 1.2809, Train: 1.0000, Val: 0.7260, Test: 0.7510
Epoch: 049, Loss: 1.2740, Train: 1.0000, Val: 0.7240, Test: 0.7520
Epoch: 050, Loss: 1.2885, Train: 1.0000, Val: 0.7240, Test: 0.7510
Epoch: 051, Loss: 1.2830, Train: 1.0000, Val: 0.7200, Test: 0.7470
Epoch: 052, Loss: 1.2695, Train: 1.0000, Val: 0.7220, Test: 0.7510
Epoch: 053, Loss: 1.2887, Train: 1.0000, Val: 0.7200, Test: 0.7460
Epoch: 054, Loss: 1.2718, Train: 1.0000, Val: 0.7240, Test: 0.7460
Epoch: 055, Loss: 1.2776, Train: 1.0000, Val: 0.7120, Test: 0.7430
Epoch: 056, Loss: 1.2712, Train: 1.0000, Val: 0.7120, Test: 0.7410
Epoch: 057, Loss: 1.2762, Train: 1.0000, Val: 0.7100, Test: 0.7410
Epoch: 058, Loss: 1.2665, Train: 1.0000, Val: 0.7080, Test: 0.7380
Epoch: 059, Loss: 1.2790, Train: 1.0000, Val: 0.7100, Test: 0.7340
Epoch: 060, Loss: 1.2624, Train: 1.0000, Val: 0.7120, Test: 0.7350
Epoch: 061, Loss: 1.2677, Train: 1.0000, Val: 0.7180, Test: 0.7390
Epoch: 062, Loss: 1.2729, Train: 1.0000, Val: 0.7320, Test: 0.7470
Epoch: 063, Loss: 1.2864, Train: 1.0000, Val: 0.7320, Test: 0.7450
Epoch: 064, Loss: 1.2742, Train: 1.0000, Val: 0.7340, Test: 0.7440
Epoch: 065, Loss: 1.2745, Train: 1.0000, Val: 0.7360, Test: 0.7480
Epoch: 066, Loss: 1.2716, Train: 1.0000, Val: 0.7420, Test: 0.7450
Epoch: 067, Loss: 1.2762, Train: 1.0000, Val: 0.7380, Test: 0.7450
Epoch: 068, Loss: 1.2583, Train: 1.0000, Val: 0.7360, Test: 0.7430
Epoch: 069, Loss: 1.2846, Train: 1.0000, Val: 0.7280, Test: 0.7500
Epoch: 070, Loss: 1.2623, Train: 1.0000, Val: 0.7220, Test: 0.7410
Epoch: 071, Loss: 1.2621, Train: 1.0000, Val: 0.7160, Test: 0.7350
Epoch: 072, Loss: 1.2605, Train: 1.0000, Val: 0.7080, Test: 0.7320
Epoch: 073, Loss: 1.2492, Train: 1.0000, Val: 0.7100, Test: 0.7310
Epoch: 074, Loss: 1.2434, Train: 1.0000, Val: 0.7120, Test: 0.7330
Epoch: 075, Loss: 1.2456, Train: 1.0000, Val: 0.7100, Test: 0.7400
Epoch: 076, Loss: 1.2629, Train: 1.0000, Val: 0.7200, Test: 0.7470
Epoch: 077, Loss: 1.2612, Train: 1.0000, Val: 0.7120, Test: 0.7460
Epoch: 078, Loss: 1.2621, Train: 1.0000, Val: 0.7160, Test: 0.7490
Epoch: 079, Loss: 1.2645, Train: 1.0000, Val: 0.7180, Test: 0.7430
Epoch: 080, Loss: 1.2429, Train: 1.0000, Val: 0.7080, Test: 0.7380
Epoch: 081, Loss: 1.2562, Train: 1.0000, Val: 0.7020, Test: 0.7300
Epoch: 082, Loss: 1.2462, Train: 1.0000, Val: 0.7020, Test: 0.7290
Epoch: 083, Loss: 1.2473, Train: 1.0000, Val: 0.7060, Test: 0.7320
Epoch: 084, Loss: 1.2577, Train: 1.0000, Val: 0.7100, Test: 0.7390
Epoch: 085, Loss: 1.2537, Train: 1.0000, Val: 0.7140, Test: 0.7360
Epoch: 086, Loss: 1.2460, Train: 1.0000, Val: 0.7160, Test: 0.7380
Epoch: 087, Loss: 1.2510, Train: 1.0000, Val: 0.7160, Test: 0.7350
Epoch: 088, Loss: 1.2315, Train: 1.0000, Val: 0.7100, Test: 0.7250
Epoch: 089, Loss: 1.2491, Train: 1.0000, Val: 0.7120, Test: 0.7310
Epoch: 090, Loss: 1.2459, Train: 1.0000, Val: 0.7180, Test: 0.7380
Epoch: 091, Loss: 1.2523, Train: 1.0000, Val: 0.7200, Test: 0.7410
Epoch: 092, Loss: 1.2462, Train: 1.0000, Val: 0.7220, Test: 0.7480
Epoch: 093, Loss: 1.2611, Train: 1.0000, Val: 0.7200, Test: 0.7460
Epoch: 094, Loss: 1.2470, Train: 1.0000, Val: 0.7200, Test: 0.7490
Epoch: 095, Loss: 1.2579, Train: 1.0000, Val: 0.7220, Test: 0.7520
Epoch: 096, Loss: 1.2528, Train: 1.0000, Val: 0.7140, Test: 0.7500
Epoch: 097, Loss: 1.2458, Train: 1.0000, Val: 0.7200, Test: 0.7490
Epoch: 098, Loss: 1.2535, Train: 1.0000, Val: 0.7160, Test: 0.7490
Epoch: 099, Loss: 1.2480, Train: 1.0000, Val: 0.7200, Test: 0.7480
Epoch: 100, Loss: 1.2515, Train: 1.0000, Val: 0.7240, Test: 0.7430
Epoch: 101, Loss: 1.2528, Train: 1.0000, Val: 0.7340, Test: 0.7480
Epoch: 102, Loss: 1.2442, Train: 1.0000, Val: 0.7320, Test: 0.7520
Epoch: 103, Loss: 1.2446, Train: 1.0000, Val: 0.7220, Test: 0.7510
Epoch: 104, Loss: 1.2494, Train: 1.0000, Val: 0.7280, Test: 0.7540
Epoch: 105, Loss: 1.2335, Train: 1.0000, Val: 0.7240, Test: 0.7590
Epoch: 106, Loss: 1.2400, Train: 1.0000, Val: 0.7140, Test: 0.7600
Epoch: 107, Loss: 1.2255, Train: 1.0000, Val: 0.7100, Test: 0.7600
Epoch: 108, Loss: 1.2428, Train: 1.0000, Val: 0.7080, Test: 0.7590
Epoch: 109, Loss: 1.2398, Train: 1.0000, Val: 0.7100, Test: 0.7570
Epoch: 110, Loss: 1.2507, Train: 1.0000, Val: 0.7080, Test: 0.7630
Epoch: 111, Loss: 1.2542, Train: 1.0000, Val: 0.7120, Test: 0.7590
Epoch: 112, Loss: 1.2473, Train: 1.0000, Val: 0.7200, Test: 0.7620
Epoch: 113, Loss: 1.2400, Train: 1.0000, Val: 0.7200, Test: 0.7530
Epoch: 114, Loss: 1.2438, Train: 1.0000, Val: 0.7100, Test: 0.7480
Epoch: 115, Loss: 1.2572, Train: 1.0000, Val: 0.7180, Test: 0.7510
Epoch: 116, Loss: 1.2309, Train: 1.0000, Val: 0.7220, Test: 0.7460
Epoch: 117, Loss: 1.2310, Train: 1.0000, Val: 0.7260, Test: 0.7460
Epoch: 118, Loss: 1.2559, Train: 1.0000, Val: 0.7260, Test: 0.7470
Epoch: 119, Loss: 1.2471, Train: 1.0000, Val: 0.7260, Test: 0.7470
Epoch: 120, Loss: 1.2406, Train: 1.0000, Val: 0.7240, Test: 0.7400
Epoch: 121, Loss: 1.2462, Train: 1.0000, Val: 0.7280, Test: 0.7370
Epoch: 122, Loss: 1.2423, Train: 1.0000, Val: 0.7260, Test: 0.7360
Epoch: 123, Loss: 1.2426, Train: 1.0000, Val: 0.7240, Test: 0.7430
Epoch: 124, Loss: 1.2433, Train: 0.9929, Val: 0.7160, Test: 0.7410
Epoch: 125, Loss: 1.2601, Train: 0.9929, Val: 0.7120, Test: 0.7420
Epoch: 126, Loss: 1.2448, Train: 1.0000, Val: 0.7240, Test: 0.7430
Epoch: 127, Loss: 1.2269, Train: 1.0000, Val: 0.7220, Test: 0.7420
Epoch: 128, Loss: 1.2519, Train: 1.0000, Val: 0.7280, Test: 0.7440
Epoch: 129, Loss: 1.2354, Train: 1.0000, Val: 0.7360, Test: 0.7430
Epoch: 130, Loss: 1.2505, Train: 0.9929, Val: 0.7240, Test: 0.7470
Epoch: 131, Loss: 1.2544, Train: 0.9929, Val: 0.7280, Test: 0.7430
Epoch: 132, Loss: 1.2518, Train: 1.0000, Val: 0.7260, Test: 0.7510
Epoch: 133, Loss: 1.2563, Train: 1.0000, Val: 0.7240, Test: 0.7380
Epoch: 134, Loss: 1.2396, Train: 1.0000, Val: 0.7080, Test: 0.7310
Epoch: 135, Loss: 1.2532, Train: 1.0000, Val: 0.7020, Test: 0.7330
Epoch: 136, Loss: 1.2388, Train: 1.0000, Val: 0.7080, Test: 0.7300
Epoch: 137, Loss: 1.2446, Train: 0.9929, Val: 0.7200, Test: 0.7290
Epoch: 138, Loss: 1.2472, Train: 0.9929, Val: 0.7280, Test: 0.7330
Epoch: 139, Loss: 1.2514, Train: 0.9929, Val: 0.7300, Test: 0.7490
Epoch: 140, Loss: 1.2564, Train: 1.0000, Val: 0.7360, Test: 0.7480
Epoch: 141, Loss: 1.2655, Train: 1.0000, Val: 0.7440, Test: 0.7520
Epoch: 142, Loss: 1.2685, Train: 1.0000, Val: 0.7420, Test: 0.7490
Epoch: 143, Loss: 1.2327, Train: 1.0000, Val: 0.7420, Test: 0.7430
Epoch: 144, Loss: 1.2462, Train: 1.0000, Val: 0.7380, Test: 0.7500
Epoch: 145, Loss: 1.2525, Train: 1.0000, Val: 0.7340, Test: 0.7510
Epoch: 146, Loss: 1.2242, Train: 1.0000, Val: 0.7240, Test: 0.7480
Epoch: 147, Loss: 1.2717, Train: 1.0000, Val: 0.7320, Test: 0.7420
Epoch: 148, Loss: 1.2586, Train: 1.0000, Val: 0.7360, Test: 0.7410
Epoch: 149, Loss: 1.2342, Train: 1.0000, Val: 0.7300, Test: 0.7480
Epoch: 150, Loss: 1.2382, Train: 1.0000, Val: 0.7260, Test: 0.7530
Epoch: 151, Loss: 1.2524, Train: 0.9929, Val: 0.7240, Test: 0.7550
Epoch: 152, Loss: 1.2544, Train: 0.9929, Val: 0.7220, Test: 0.7580
Epoch: 153, Loss: 1.2604, Train: 1.0000, Val: 0.7260, Test: 0.7520
Epoch: 154, Loss: 1.2594, Train: 1.0000, Val: 0.7260, Test: 0.7520
Epoch: 155, Loss: 1.2446, Train: 1.0000, Val: 0.7320, Test: 0.7530
Epoch: 156, Loss: 1.2442, Train: 1.0000, Val: 0.7380, Test: 0.7520
Epoch: 157, Loss: 1.2518, Train: 0.9929, Val: 0.7420, Test: 0.7480
Epoch: 158, Loss: 1.2473, Train: 1.0000, Val: 0.7440, Test: 0.7550
Epoch: 159, Loss: 1.2500, Train: 1.0000, Val: 0.7400, Test: 0.7480
Epoch: 160, Loss: 1.2467, Train: 1.0000, Val: 0.7400, Test: 0.7530
Epoch: 161, Loss: 1.2480, Train: 1.0000, Val: 0.7380, Test: 0.7510
Epoch: 162, Loss: 1.2522, Train: 1.0000, Val: 0.7360, Test: 0.7520
Epoch: 163, Loss: 1.2514, Train: 1.0000, Val: 0.7380, Test: 0.7550
Epoch: 164, Loss: 1.2480, Train: 1.0000, Val: 0.7340, Test: 0.7530
Epoch: 165, Loss: 1.2331, Train: 1.0000, Val: 0.7420, Test: 0.7550
Epoch: 166, Loss: 1.2619, Train: 1.0000, Val: 0.7360, Test: 0.7550
Epoch: 167, Loss: 1.2427, Train: 1.0000, Val: 0.7360, Test: 0.7510
Epoch: 168, Loss: 1.2572, Train: 1.0000, Val: 0.7400, Test: 0.7510
Epoch: 169, Loss: 1.2669, Train: 1.0000, Val: 0.7300, Test: 0.7600
Epoch: 170, Loss: 1.2490, Train: 1.0000, Val: 0.7380, Test: 0.7640
Epoch: 171, Loss: 1.2544, Train: 1.0000, Val: 0.7380, Test: 0.7640
Epoch: 172, Loss: 1.2421, Train: 1.0000, Val: 0.7420, Test: 0.7650
Epoch: 173, Loss: 1.2368, Train: 0.9929, Val: 0.7520, Test: 0.7640
Epoch: 174, Loss: 1.2802, Train: 0.9929, Val: 0.7500, Test: 0.7610
Epoch: 175, Loss: 1.2445, Train: 1.0000, Val: 0.7460, Test: 0.7630
Epoch: 176, Loss: 1.2453, Train: 1.0000, Val: 0.7460, Test: 0.7570
Epoch: 177, Loss: 1.2220, Train: 1.0000, Val: 0.7440, Test: 0.7520
Epoch: 178, Loss: 1.2482, Train: 1.0000, Val: 0.7480, Test: 0.7470
Epoch: 179, Loss: 1.2483, Train: 1.0000, Val: 0.7460, Test: 0.7450
Epoch: 180, Loss: 1.2386, Train: 1.0000, Val: 0.7480, Test: 0.7470
Epoch: 181, Loss: 1.2381, Train: 1.0000, Val: 0.7480, Test: 0.7490
Epoch: 182, Loss: 1.2450, Train: 1.0000, Val: 0.7560, Test: 0.7560
Epoch: 183, Loss: 1.2366, Train: 1.0000, Val: 0.7520, Test: 0.7530
Epoch: 184, Loss: 1.2393, Train: 1.0000, Val: 0.7480, Test: 0.7540
Epoch: 185, Loss: 1.2476, Train: 1.0000, Val: 0.7440, Test: 0.7510
Epoch: 186, Loss: 1.2545, Train: 1.0000, Val: 0.7560, Test: 0.7460
Epoch: 187, Loss: 1.2287, Train: 1.0000, Val: 0.7560, Test: 0.7480
Epoch: 188, Loss: 1.2393, Train: 1.0000, Val: 0.7500, Test: 0.7530
Epoch: 189, Loss: 1.2355, Train: 1.0000, Val: 0.7520, Test: 0.7500
Epoch: 190, Loss: 1.2639, Train: 1.0000, Val: 0.7480, Test: 0.7470
Epoch: 191, Loss: 1.2554, Train: 1.0000, Val: 0.7480, Test: 0.7450
Epoch: 192, Loss: 1.2567, Train: 1.0000, Val: 0.7400, Test: 0.7450
Epoch: 193, Loss: 1.2235, Train: 1.0000, Val: 0.7280, Test: 0.7440
Epoch: 194, Loss: 1.2341, Train: 1.0000, Val: 0.7280, Test: 0.7420
Epoch: 195, Loss: 1.2280, Train: 1.0000, Val: 0.7280, Test: 0.7410
Epoch: 196, Loss: 1.2624, Train: 1.0000, Val: 0.7300, Test: 0.7400
Epoch: 197, Loss: 1.2522, Train: 1.0000, Val: 0.7300, Test: 0.7400
Epoch: 198, Loss: 1.2408, Train: 1.0000, Val: 0.7360, Test: 0.7430
Epoch: 199, Loss: 1.2456, Train: 1.0000, Val: 0.7280, Test: 0.7470
Epoch: 200, Loss: 1.2616, Train: 1.0000, Val: 0.7280, Test: 0.7400
"""

# Function to parse data string
def parse_data(data_str):
    data = []
    for line in data_str.strip().split('\n'):
        parts = line.split(',')
        epoch = int(parts[0].split(':')[1].strip())
        val = float(parts[3].split(':')[1].strip())
        test = float(parts[4].split(':')[1].strip())
        data.append([epoch, val, test])
    return pd.DataFrame(data, columns=['Epoch', 'Val', 'Test'])

# Parsing the data
df1 = parse_data(data_str1)
df2 = parse_data(data_str2)

# Plotting
plt.figure(figsize=(10, 6))

# Plot dataset 1
plt.plot(df1['Epoch'], df1['Val'], color='orange', label='Dataset 1 - Val')
#plt.plot(df1['Epoch'], df1['Test'], color='blue', label='Dataset 1 - Test')

# Plot dataset 2
plt.plot(df2['Epoch'], df2['Val'], color='green', label='Dataset 2 - Val')
#plt.plot(df2['Epoch'], df2['Test'], color='green', linestyle='dashed', label='Dataset 2 - Test')

plt.xlabel('Epoch')
plt.ylabel('Values')
plt.title('Comparison of Two Datasets')
plt.legend()
plt.show()
