import numpy as np

#read data
filename = "../data/data.xls"
data = np.loadtxt(fname=filename, delimiter=',', skiprows=2, encoding='cp1252')
print(data)