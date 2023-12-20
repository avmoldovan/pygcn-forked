import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import matplotlib.pyplot as plt

def get_baseline_run():

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
    df1 = parse_data(data_str1)

    return df1

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

def draw_wireframe(G):
    pos = nx.spring_layout(G, seed=42)
    cent = nx.degree_centrality(G)
    node_size = list(map(lambda x: x * 500, cent.values()))
    cent_array = np.array(list(cent.values()))
    threshold = sorted(cent_array, reverse=True)[10]
    print("threshold", threshold)
    cent_bin = np.where(cent_array >= threshold, 1, 0.1)
    plt.figure(figsize=(12, 12))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size,
                                   cmap=plt.cm.plasma,
                                   node_color=cent_bin,
                                   nodelist=list(cent.keys()),
                                   alpha=cent_bin)
    edges = nx.draw_networkx_edges(G, pos, width=0.25, alpha=0.3)
    plt.show()
def draw_network_edges(G, data, label_dict):
    node_color = []
    nodelist = [[], [], [], [], [], [], []]
    colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
    labels = data.y
    for n, i in enumerate(labels):
        node_color.append(colorlist[i])
        nodelist[i].append(n)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 10))
    labellist = list(label_dict.values())
    for num, i in enumerate(zip(nodelist, labellist)):
        n, l = i[0], i[1]
        nx.draw_networkx_nodes(G, pos, nodelist=n, node_size=5, node_color=colorlist[num], label=l)
    nx.draw_networkx_edges(G, pos, width=0.25)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

def visualize_mesh(pos, face):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=data.face.t(), antialiased=False)
    plt.show()


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()