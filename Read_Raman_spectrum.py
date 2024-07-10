import numpy as np
from numpy import dtype
array = np.load('Raman_spectrum_M(1).npy', allow_pickle=True)



# 尝试获取帮助文档
print("\nObject help:")
spectral_container = array.item()
# print(help(spectral_container))



spectra_list = spectral_container.tolist()
print(len(spectra_list))  # 该文件下样本数量

# 打印样本的Raman_spectrum数据
print("Spectra list:")
for spectrum in spectra_list:
    print(spectrum.spectral_data)  ## 波峰数据
    print(spectrum.spectral_axis)  ## 波峰坐标


# # 使用 flat 属性
# flat_array = spectral_container.flat
# print("Flattened array shape:", flat_array.shape)
# print(flat_array)
#
# # 使用 mean 属性
# mean_spectrum = spectral_container.mean
# print("Mean spectrum:", mean_spectrum)
#
# # 使用 shape 属性
# spatial_shape = spectral_container.shape
# print("Spatial shape:", spatial_shape)
#
# # 使用 spectral_length 属性
# spectral_length = spectral_container.spectral_length
# print("Spectral length:", spectral_length)





