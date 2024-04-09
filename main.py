import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])









regr = MLPRegressor(hidden_layer_sizes=(500, 400, 300, 150, 100),
                    max_iter=20000,
                    activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=10000)




print('Treinando RNA')
regr = regr.fit(x,y)

print('Preditor')
y_est = regr.predict(x)

# Calculando o erro
erro = y - y_est

# Calculando a média do erro
media_erro = np.mean(erro)

# Calculando o desvio padrão do erro
desvio_padrao_erro = np.std(erro)

for i in range(100):
    print("\n")
print("Media do erro:", media_erro)
print("Desvio padrao do erro:", desvio_padrao_erro)

plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)

plt.show()
