import numpy as np
from Utils import *

__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"

packets_cfd = PacketCDF()
packets_cfd.load_cdf_data('{}/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv'.format(__ns3_path))
print(packets_cfd.compute_conditional_probability(-1500, 3000))
# list =[
#         58726.0,
#         57869.0,
#         59760.0,
#         53101.0,
#         55998.0,
#         56295.0,
#         52586.0,
#         57323.0,
#         52778.0,
#         54067.0,
#         54025.0,
#         56667.0,
#         55818.0,
#         54449.0,
#         45940.0,
#         56718.0,
#         48120.0,
#         50947.0,
#         54076.0,
#         53534.0,
#         52471.0,
#         52639.0,
#         55321.0,
#         54849.0,
#         52172.0,
#         52726.0,
#         56785.0,
#         52430.0,
#         54954.0,
#         52512.0
#     ]
# print(np.average(list))