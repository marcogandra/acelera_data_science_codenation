#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


fifa.shape


# In[6]:


fifa.info()


# Existem dados ausentes na maior parte das variáveis para 48 das 18.207 linhas do *dataset*. Como todos os atributos, com exceção de três, estão com esses dados ausentes, vou desconsiderar essas linhas.

# In[7]:


# remove as linhas com dados ausentes
fifa.dropna(inplace=True)


# In[8]:


fifa.shape


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[9]:


def q1():
    pca = PCA().fit(fifa)
    perc_pc1 = pca.explained_variance_ratio_
    
    return round(perc_pc1[0],3)


# In[10]:


# chama a função
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[11]:


def q2():
    pca = PCA(0.95).fit(fifa)
    
    return len(pca.explained_variance_ratio_)


# In[12]:


# chama a função
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[13]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[14]:


def q3():
    # pca
    pca = PCA().fit(fifa)
    components = pca.components_
    
    # coordenadas
    coordenadas = components.dot(x)
    
    # cria a tupla
    tupla = tuple(np.round(coordenadas[0:2],3))
    
    return tupla


# In[15]:


# chama a função
q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# (Faltou mencionar que a variável resposta é a variável `Overall`.)

# In[16]:


def q4():
    # separa y e x
    y = fifa['Overall']
    x = fifa.drop('Overall', axis=1)
    
    # RFE
    reg = LinearRegression()
    rfe = RFE(reg, n_features_to_select=5)
    rfe.fit(x, y)
    
    # separa as variáveis que foram selecionadas
    aux = pd.DataFrame({'feature': x.columns,
                        'bool': rfe.get_support()})

    selecionadas = list(aux[aux['bool']]['feature'])
    return selecionadas


# In[17]:


# chama a função
q4()

