import pandas as pd
import matplotlib.pyplot as plt
from ID3 import ArvoreDecisaoID3

# nomeando as colunas da base de dados
colunas = [
    'classe', 'resposta_1', 'resposta_2', 'resposta_3', 'resposta_4', 'resposta_5',
    'resposta_6', 'resposta_7', 'resposta_8', 'resposta_9', 'resposta_10',
    'resposta_11', 'resposta_12', 'resposta_13', 'resposta_14', 'resposta_15', 'resposta_16'
]

# Carregando os dados da base de dados:
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
df = pd.read_csv(url, names=colunas)

# Exibição das 5 primeiras linhas da base de dados:
df.head()

# Exibição das 5 ultimas linhas da base de dados:
df.tail()

# Contagem da quantidade de democratas e republicanos na base de dados
contagem_classes = df['classe'].value_counts()

print("Distribuição das classes na base de dados: ")
for classe, count in contagem_classes.items():
    print(f"- {classe}: {count}")

# Divisão da base de dados

# Embaralhando e redefinindo os índices dos dados na base de dados
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculando o tamanho dos conjuntos de treino e teste
tamanho_total = len(df)
tamanho_treino = int(0.7 * tamanho_total)

# Dividindo a base de dados para treino e teste
df_treino = df.iloc[:tamanho_treino]
df_teste = df.iloc[tamanho_treino:]

# Separação das características (X) e rótulos (y)
X_treino = df_treino.drop(columns=['classe'])
y_treino  = df_treino['classe']
X_teste = df_teste.drop(columns=['classe'])
y_teste = df_teste['classe']

# Exibição dos tamanhos dos conjuntos
print(f"Tamanho do conjunto de treinamento: {len(df_treino)} exemplos")
print(f"Tamanho do conjunto de teste: {len(df_teste)} exemplos")

# Conversão das variáveis categóricas em variáveis numéricas usando one-hot encoding
X_treino_encoded = pd.get_dummies(X_treino)
X_teste_encoded = pd.get_dummies(X_teste)

# Garantindo que ambas as bases (treino e teste) tenham as mesmas colunas após a conversão
X_treino_encoded, X_teste_encoded = X_treino_encoded.align(X_teste_encoded, join='left', axis=1)
X_teste_encoded = X_teste_encoded.fillna(0)  # Preenchendo possíveis valores NaN após alinhamento

# Criando um dicionário para mapear os valores categóricos para numéricos
mapeamento_classes = {"democrat": 0, "republican": 1}

# Convertendo os rótulos para valores numéricos, garantindo que não haja espaços extras
y_treino_encoded = y_treino.astype(str).str.strip().map(mapeamento_classes)
y_teste_encoded = y_teste.astype(str).str.strip().map(mapeamento_classes)

# Verificando se a conversão foi feita corretamente
print("Exemplo de rótulos antes da conversão:")
y_treino.head()

print("\nExemplo de rótulos após a conversão:")
y_treino_encoded.head()

arvore = ArvoreDecisaoID3()

historico_ganhos = {}     # Dicionário para armazenar os ganhos dos atributos e apresentar eles posterioremente num gráfico de barras

# Treinando o modelo e obtendo a acurácia de treino
acuracia_treino = arvore.treinar(X_treino, y_treino, historico_ganhos)

# Fazendo predições no conjunto de teste e calculando a acurácia de teste
previsoes_teste = arvore.prever(X_teste)
acuracia_teste = arvore.calcular_acuracia(previsoes_teste, y_teste)

# Acurácia de treino e acurácia de teste
acuracia_treino = acuracia_treino * 100  # Convertendo para porcentagem
acuracia_teste = acuracia_teste * 100    # Convertendo para porcentagem

# Criando o gráfico de barras horizontais (invertendo a ordem)
plt.barh(['Acurácia de Teste', 'Acurácia de Treino'], [acuracia_teste, acuracia_treino], color=['#4FA3F7', 'orange'])

# Adicionando título e rótulos
plt.title('Acurácia de Treino e Teste')
plt.xlabel('Acurácia (%)')
plt.xlim(0, 100)  # Limite do gráfico de 0 a 100%

# Adicionando os valores das acurácias dentro das barras
for i, acuracia in enumerate([acuracia_teste, acuracia_treino]):
    plt.text(acuracia + 2, i, f'{acuracia:.2f}%', va='center', ha='left', color='black', fontweight='bold')

# Exibindo o gráfico
plt.show()

# Dicionário para armazenar os ganhos dos atributos e apresentar eles posterioremente num gráfico de barras_teste * 100:.2f}%")

# Calcula a média do ganho de informação para cada atributo
importancia_atributos = {}

for atributo, ganhos in historico_ganhos.items():
    importancia_atributos[atributo] = sum(ganhos) / len(ganhos)

# Criando o gráfico de barras
plt.figure(figsize=(15, 7))
bars = plt.bar(importancia_atributos.keys(), importancia_atributos.values(), color='#FF2400')

# Adicionando os valores exatos em cima das barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

plt.xlabel("Atributos")
plt.ylabel("Ganho de Informação Médio")
plt.title("Importância dos Atributos na Árvore de Decisão")
plt.xticks(rotation=45)
plt.show()


# Criando a matriz de confusão
cm = pd.crosstab(y_teste, previsoes_teste, rownames=['Classe Real'], colnames=['Classe Prevista'])

# Plotando a matriz de confusão
plt.figure(figsize=(6, 5))  # Tamanho da figura
plt.imshow(cm, interpolation='nearest', cmap='Blues')  # Exibindo a imagem (matriz) com a paleta 'Blues'
plt.title('Matriz de Confusão', fontsize=16)
plt.colorbar()  # Barra de cores

# Adicionando os rótulos aos eixos e as anotações nas células
tick_marks = range(len(cm.columns))
plt.xticks(tick_marks, cm.columns, rotation=45, fontsize=12)
plt.yticks(tick_marks, cm.index, fontsize=12)

# Adicionando os valores nas células
for i in range(len(cm.columns)):
    for j in range(len(cm.index)):
        plt.text(j, i, str(cm.iloc[i, j]), ha="center", va="center", color="black", fontsize=14)

plt.xlabel('Classe Prevista', fontsize=12)
plt.ylabel('Classe Real', fontsize=12)
plt.tight_layout()  # Ajusta para não cortar nenhum elemento
plt.show()