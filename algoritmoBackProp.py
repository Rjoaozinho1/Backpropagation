import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Função sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

# Configurações da rede neural
tamanho_entrada = 2
tamanho_oculta = 3
tamanho_saida = 1
taxa_aprendizado = 0.5
epocas = 5000

# Dados de entrada e saída
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

y = np.array([[0],[1],[1],[0]])

# Inicialização dos pesos
pesos_entrada_oculta = np.random.uniform(size=(tamanho_entrada, tamanho_oculta))
pesos_oculta_saida = np.random.uniform(size=(tamanho_oculta, tamanho_saida))

# Lista para armazenar o erro ao longo das épocas
historico_erro = []

# Treinamento da rede neural
for epoca in range(epocas):
    # Feedforward
    entrada_oculta = np.dot(X, pesos_entrada_oculta)
    saida_oculta = sigmoid(entrada_oculta)

    entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
    saida_prevista = sigmoid(entrada_saida)

    # Cálculo do erro
    erro = y - saida_prevista

    # Backpropagation
    erro_saida = erro * derivada_sigmoid(saida_prevista)
    erro_oculta = erro_saida.dot(pesos_oculta_saida.T) * derivada_sigmoid(saida_oculta)

    # Atualização dos pesos
    pesos_oculta_saida += saida_oculta.T.dot(erro_saida) * taxa_aprendizado
    pesos_entrada_oculta += X.T.dot(erro_oculta) * taxa_aprendizado

    # Armazena o erro para plotar o gráfico
    historico_erro.append(np.mean(np.abs(erro)))

# Plotar o gráfico do erro ao longo das épocas
plt.plot(range(1, epocas + 1), historico_erro)
plt.title('Evolução do Erro durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Erro Médio Absoluto')
plt.show()

# Teste da rede neural treinada
# novos_dados = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
novos_dados = np.array([[0, 0],[0, 1],[1, 1],[1, 0]])
# novos_dados = np.array([[1, 1],[0, 1],[0, 0],[1, 0]]) # teste nao muito bom

entrada_oculta = np.dot(novos_dados, pesos_entrada_oculta)
saida_oculta = sigmoid(entrada_oculta)

entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
saida_prevista = sigmoid(entrada_saida)

# Tabela com a saída prevista
tabela_saida_prevista = tabulate(saida_prevista, headers=["Saída Prevista"], tablefmt="pretty")
print("\nTabela com a Saída Prevista:")
print(tabela_saida_prevista)