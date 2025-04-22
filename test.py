import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('resultados.csv')

plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
plt.title("Histograma de Números Aleatorios (0 a 1)")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()
