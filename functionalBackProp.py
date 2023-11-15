import numpy as np
import matplotlib.pyplot as plt

# Função identidade para previsão contínua
def identidade(x):
    return x

# Derivada da função identidade
def derivada_identidade(x):
    return 1

# Função sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

# Configurações da rede neural
tamanho_entrada = 2
tamanho_oculta = 4  # Ajuste o número de neurônios conforme necessário
tamanho_saida = 1
taxa_aprendizado = 0.5  # Ajuste a taxa de aprendizado conforme necessário
epocas = 1000  # Pode ser necessário aumentar o número de épocas

# Dados de entrada e saída (valores contínuos)
X = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])
y = np.array([[0.1], [0.9], [0.8], [0.2]])  # Exemplo de valores de preço de ações

# Inicialização dos pesos
pesos_entrada_oculta = np.random.uniform(size=(tamanho_entrada, tamanho_oculta))
pesos_oculta_saida = np.random.uniform(size=(tamanho_oculta, tamanho_saida))

# Lista para armazenar o erro ao longo das épocas
historico_erro = []

# TREINAMENTO

# Treinamento da rede neural
for epoca in range(epocas):
    # Feedforward
    entrada_oculta = np.dot(X, pesos_entrada_oculta)
    saida_oculta = sigmoid(entrada_oculta)

    entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
    saida_prevista = identidade(entrada_saida)

    # Cálculo do erro (Mean Squared Error)
    erro = y - saida_prevista
    erro_medio = np.mean(erro**2)
    historico_erro.append(erro_medio)

    # Backpropagation
    erro_saida = erro * derivada_identidade(saida_prevista)
    erro_oculta = erro_saida.dot(pesos_oculta_saida.T) * derivada_sigmoid(saida_oculta)

    # Atualização dos pesos
    pesos_oculta_saida += saida_oculta.T.dot(erro_saida) * taxa_aprendizado
    pesos_entrada_oculta += X.T.dot(erro_oculta) * taxa_aprendizado

# Plotar o gráfico do erro ao longo das épocas
plt.plot(range(1, epocas + 1), historico_erro)
plt.title('Evolução do Erro durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Erro Médio Quadrático')
plt.show()

# TESTE

# Teste da rede neural treinada
novos_dados = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

entrada_oculta = np.dot(novos_dados, pesos_entrada_oculta)
saida_oculta = sigmoid(entrada_oculta)

entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
saida_prevista = identidade(entrada_saida)

# Imprimir os valores previstos
print("\nValores Previstos:")
print(saida_prevista)