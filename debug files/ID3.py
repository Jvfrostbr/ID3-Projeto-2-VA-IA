import pandas as pd
import matplotlib.pyplot as plt
import math

class ArvoreDecisaoID3:
    def __init__(self):
        self.arvore = {}  # Estrutura que armazenará a árvore de decisão

    def calcular_entropia(self, dados_alvo):
        qtd_ocorrencias = dados_alvo.value_counts()       # conta a quantidade de ocorrências de cada classe
        probabilidade = qtd_ocorrencias / len(dados_alvo)

        # Calcula a entropia utilizando a fórmula
        entropia = -sum(probabilidade * probabilidade.apply(lambda p: math.log2(p) if p > 0 else 0))

        return entropia

    def calcular_ganho_informacao(self, dados, dados_alvo, atributo):
        entropia_total = self.calcular_entropia(dados_alvo)

        # Obtém os valores únicos para o atributo
        valores_unicos = dados[atributo].unique()
        ganho = 0

        for valor in valores_unicos:
            # Filtra o conjunto de dados para incluir apenas as ocorrências onde o atributo tem o valor da interação
            dados_filtrados = dados[dados[atributo] == valor]
            dados_alvo_filtrados = dados.loc[dados[atributo] == valor, dados_alvo.name]

            probabilidade_valor = len(dados_filtrados) / len(dados)
            entropia_subconjunto = self.calcular_entropia(dados_alvo_filtrados)
            ganho += probabilidade_valor * entropia_subconjunto  # Calcula o ganho total

        ganho_informacao = entropia_total - ganho
        return ganho_informacao

    def encontrar_melhor_atributo(self, dados, atributos, dados_alvo, historico_ganhos):
      # Cria um dicionário onde a chave é o atributo e o valor é o ganho de informação obtida por esse atributo
        ganhos = {}

        for atributo in atributos:
            ganho = self.calcular_ganho_informacao(dados, dados_alvo, atributo)
            ganhos[atributo] = ganho

            # Armazena o ganho no histórico (guarda em lista para calcular média depois)
            if atributo in historico_ganhos:
                historico_ganhos[atributo].append(ganho)
            else:
                historico_ganhos[atributo] = [ganho]

        melhor_atributo = max(ganhos, key=ganhos.get)    # Encontra o atributo que possuir o maior ganho de informação
        return melhor_atributo

    def construir_arvore(self, dados, atributos, dados_alvo, profundidade, max_depth, historico_ganhos):
        valores_alvo = dados_alvo.unique()

        # Segue abaixo os critérios de parada para a construção da árvore:
        # Caso base 1: se todos os exemplos pertencem à mesma classe
        if len(valores_alvo) == 1:
            return valores_alvo[0]

        # Caso base 2: se não há mais atributos para dividir
        if len(atributos) == 0 or profundidade >= max_depth:
            return dados_alvo.mode()[0]  # Retorna a classe mais frequente

        # Caso base 3: se a profundidade já atingiu a profundidade máxima
        if profundidade >= max_depth:
            return valores_alvo[0]

        # Encontra o melhor atributo para dividir
        melhor_atributo = self.encontrar_melhor_atributo(dados, atributos, dados_alvo, historico_ganhos)
        arvore = {melhor_atributo: {}}  # Inicializa a árvore com o melhor atributo

        # Remove o melhor atributo da lista de atributos
        atributos_restantes = [atributo for atributo in atributos if atributo != melhor_atributo]

        # Divide os dados e chama recursivamente para construir subárvores
        for valor in dados[melhor_atributo].unique():
            subconjunto = dados[dados[
                                    melhor_atributo] == valor]  # retorna um subconjunto de dados com as linhas em que o valor da coluna melhor_atributo é igual a valor
            if len(subconjunto) == 0:
                arvore[melhor_atributo][valor] = dados_alvo.mode()[0]  # Caso não haja exemplos
            else:
                arvore[melhor_atributo][valor] = self.construir_arvore(subconjunto, atributos_restantes,
                                                                       subconjunto[dados_alvo.name], profundidade + 1,
                                                                       max_depth, historico_ganhos)

        return arvore

    def treinar(self, X_treino, y_treino, historico_ganhos):
        dados = X_treino.copy()
        dados['alvo'] = y_treino
        atributos = X_treino.columns.tolist()
        profundidade = 0               # Atribuindo a prondidade inicial da árvore (que recursivamente vai ser incrementada em construir_arvore)
        max_depth = 3
        self.arvore = self.construir_arvore(dados, atributos, dados['alvo'], profundidade, max_depth, historico_ganhos)

        # Predizendo para o conjunto de dados dados de treino e calculando a acurácia
        previsoes_treino = self.prever(X_treino)
        acuracia_treino = self.calcular_acuracia(previsoes_treino, y_treino)

        return acuracia_treino

    def prever_amostra(self, arvore, amostra):
        if not isinstance(arvore, dict):
            return arvore  # Retorna o valor da folha (classe prevista)

        atributo = list(arvore.keys())[0]  # Pega o primeiro atributo (nó de decisão)
        valor = amostra[atributo]  # Pega o valor da amostra para o atributo atual

        if valor not in arvore[atributo]:
            return None  # Retorna None se o valor não estiver na árvore

        # Chamada recursiva para o próximo nível da árvore
        return self.prever_amostra(arvore[atributo][valor], amostra)

    def prever(self, X_teste):

        previsoes = []  # Lista onde as previsões serão armazenadas

        # Itera sobre cada linha do conjunto de teste
        for _, linha in X_teste.iterrows():
            previsoes.append(self.prever_amostra(self.arvore, linha))  # Faz a predição para a linha e adiciona na lista de previsões
        return previsoes

    def calcular_acuracia(self, previsoes, y_real):
        acertos = sum(previsoes == y_real)
        return acertos / len(y_real)