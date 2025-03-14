# =============================================================================
                                ## BIBLIOTECAS ##
# =============================================================================
import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import time


# =============================================================================
                                ## FUNÇÕES ##
# =============================================================================
def prever_cluster(df, kmeans_model):
    """
    Recebe as informações do cliente e retorna o cluster ao qual ele pertence.
    """
    pop_porte_dict = {
        '0. Ate 20 Mil Habitantes': 1, 
        '1. De 20 a 50 Mil Habitantes': 2,
        '2. De 50 a 100 Mil Habitantes': 3,
        '3. De 100 a 300 Mil Habitantes': 4,
        '4. De 300 a 1 Milhao de Habitantes': 5,
        '5. Acima de 1 Milhao de Habitantes': 6
    }
    pop_forte_dict = {
        '1. De 0.0 a 360 Mil': 1, 
        '2. De 360 Mil a 4.8 Milhoes': 2,
        '3. Acima de 4.8 Milhoes': 3
    }
    pop_CHECK_dict = {
        '0. De 1 a 4 Ckts': 1, 
        '1. De 5 a 9 Ckts': 2,
        '2. Acima de 10 Ckts': 3
    }
    pop_tempo_dict = {
        '1. Ate 5 anos': 1, 
        '2. De 5 a 15 anos': 2,
        '3. De 16 a 25 anos': 3,
        '4. Acima de 26 anos': 4
    }
    df['POP_PORTE'] = df['POP_PORTE'].map(pop_porte_dict)
    df['PORTE_VALOR'] = df['PORTE_VALOR'].map(pop_forte_dict)
    df['PORT_CHECK'] = df['PORT_CHECK'].map(pop_CHECK_dict)
    df['TEMPO_DE_ABERTURA'] = df['TEMPO_DE_ABERTURA'].map(pop_tempo_dict)

    df = pd.get_dummies(df, columns=['UF'])

    ordem = pd.DataFrame(columns = ['POP_PORTE', 'PORTE_VALOR', 'PORT_CHECK', 'TEMPO_DE_ABERTURA',
                                    'UF_AL', 'UF_AM', 'UF_AP', 'UF_BA', 'UF_CE', 'UF_DF',
                                    'UF_ES', 'UF_GO', 'UF_MA', 'UF_MG', 'UF_MS', 'UF_MT', 'UF_PA', 'UF_PB',
                                    'UF_PE', 'UF_PI', 'UF_PR', 'UF_RJ', 'UF_RN', 'UF_RO', 'UF_RR', 'UF_RS',
                                    'UF_SC', 'UF_SE', 'UF_SP'
                                    ]) 
    for col in ordem.columns:
            if col not in df.columns:
                df[col] = False
    df = df[ordem.columns]

    # Predição do cluster
    df['cluster'] = kmeans.predict(df)

    return df


def prever_cpr(df, model, scaler_robust_saida):
    """
    Recebe as informações do cliente e retorna o o valor previsto de saida do cpr.
    """
    
    ordem = pd.DataFrame(columns = ['POP_PORTE', 'PORTE_VALOR', 'PORT_CHECK', 'TEMPO_DE_ABERTURA', 'UF_AL',
                                    'UF_AM', 'UF_AP', 'UF_BA', 'UF_CE', 'UF_DF', 'UF_ES', 'UF_GO', 'UF_MA',
                                    'UF_MG', 'UF_MS', 'UF_MT', 'UF_PA', 'UF_PB', 'UF_PE', 'UF_PI', 'UF_PR',
                                    'UF_RJ', 'UF_RN', 'UF_RO', 'UF_RR', 'UF_RS', 'UF_SC', 'UF_SE', 'UF_SP']) 

    for col in ordem.columns:
            if col not in df.columns:
                df[col] = False
    df = df[ordem.columns]

    # Predição do cluster
    df['VALOR_CPR_SAIDA'] = model.predict(df)

    # Reverter a Padronização da Previsão
    df['VALOR_CPR_SAIDA'] = scaler_robust_saida.inverse_transform(df['VALOR_CPR_SAIDA'].values.reshape(-1, 1)).flatten()

    return df 


def prever_tef(df, model, scaler_robust_saida):
    """
    Recebe as informações do cliente e retorna o o valor previsto de saida do cpr.
    """
    
    ordem = pd.DataFrame(columns = ['POP_PORTE', 'PORTE_VALOR', 'PORT_CHECK', 'TEMPO_DE_ABERTURA', 'UF_AL',
                                    'UF_AM', 'UF_AP', 'UF_BA', 'UF_CE', 'UF_DF', 'UF_ES', 'UF_GO', 'UF_MA',
                                    'UF_MG', 'UF_MS', 'UF_MT', 'UF_PA', 'UF_PB', 'UF_PE', 'UF_PI', 'UF_PR',
                                    'UF_RJ', 'UF_RN', 'UF_RO', 'UF_RR', 'UF_RS', 'UF_SC', 'UF_SE', 'UF_SP']) 

    for col in ordem.columns:
            if col not in df.columns:
                df[col] = False
    df = df[ordem.columns]

    # Predição do cluster
    df['VALOR_TRN'] = model.predict(df)

    # Reverter a Padronização da Previsão
    df['VALOR_TRN'] = (
    scaler_robust_saida.inverse_transform(df[['VALOR_TRN']].values)
    .flatten()
    .round(2)
    )
    return df 



# =============================================================================
                                ## PREVENDO  ##
# =============================================================================

# Carregar o modelo de clusterização
kmeans = joblib.load('modelo_clusterizacao.pkl')

# Carregar modelo e scaler para o cpr
data = joblib.load('modelo_cpr.pkl')
modelo = data['model']
scaler_robust_saida = data['scaler']


# Carregar modelo e scaler para o tef
data1 = joblib.load('modelo_tef.pkl')
modelo1 = data['model']
scaler_robust_saida1 = data1['scaler']


# Interface Streamlit
st.title("Simulador do Perfil dos Clientes")
st.write("Preencha os dados do cliente para prever a qual cluster ele pertence.")

# Entradas do usuário
pop_porte = st.selectbox("População do Município", options=[
    '0. Ate 20 Mil Habitantes', '1. De 20 a 50 Mil Habitantes', '2. De 50 a 100 Mil Habitantes', 
    '3. De 100 a 300 Mil Habitantes', '4. De 300 a 1 Milhao de Habitantes', '5. Acima de 1 Milhao de Habitantes'
])


# porte_valor = st.selectbox("Porte da Loja", options=[
#     '1. De 0.0 a 360 Mil', '2. De 360 Mil a 4.8 Milhoes', '3. Acima de 4.8 Milhoes'
# ])

porte = st.selectbox("Porte da Loja", options=[
    'MICRO EMPRESA', 'EMPRESA DE PEQUENO PORTE', 'EMPRESA DE MEDIO E GRANDE PORTE'
])



port_check = st.selectbox("Número de checkouts da Loja", options=[
    '0. De 1 a 4 Ckts', '1. De 5 a 9 Ckts', '2. Acima de 10 Ckts'
])

tempo_abertura = st.selectbox("Tempo de Abertura em anos da Loja", options=[
    '1. Ate 5 anos', '2. De 5 a 15 anos', '3. De 16 a 25 anos', '4. Acima de 26 anos'
])

uf = st.selectbox("UF da Loja", options=[
    'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 
    'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP'
])

# perfil_saida = st.selectbox("Perfil de Saída", options=[
#     'MUITO PEQUENA', 'PEQUENA', 'MEDIA', 'GRANDE'
# ])


# Criando um DataFrame com os dados inseridos pelo usuário
cliente = {
    'POP_PORTE': pop_porte,
    'PORTE': porte,
    'PORT_CHECK': port_check,
    'TEMPO_DE_ABERTURA': tempo_abertura,
    'UF': uf
    }

df_cliente = pd.DataFrame([cliente])

df_cliente['PORTE_VALOR'] = df_cliente['PORTE'].apply(lambda x: 
    '1. De 0.0 a 360 Mil' if x == 'MICRO EMPRESA' else 
    '2. De 360 Mil a 4.8 Milhoes' if x == 'EMPRESA DE PEQUENO PORTE' else 
    '3. Acima de 4.8 Milhoes' if x == 'EMPRESA DE MEDIO E GRANDE PORTE' else 
    None)

df_cliente = df_cliente.drop('PORTE', axis=1)



# Botão de Previsão
if st.button('Fazer previsão'):
    with st.spinner('Fazendo a previsão...'):
        time.sleep(2)  # Simula o tempo de processamento
        resultado = prever_cluster(df_cliente, kmeans)  # Faz a previsão cluster 
        resultado1 = prever_cpr(resultado, modelo, scaler_robust_saida )  # Faz a previsão cpr
        resultado2 = prever_tef(resultado, modelo1, scaler_robust_saida1)  # Faz a previsão cpr
        st.write(f"O cliente pertence ao Cluster {resultado['cluster'].iloc[0]}.")  # Exibe o resultado
        st.write(f"O Valor Mensal de Saída do CPR estimado fica em torno de R$ {round(resultado1['VALOR_CPR_SAIDA'].iloc[0])},00 com uma margem de erro de +/- 150000,00 reais.")        
        st.write(f"O Valor Mensal Transacionado (TEF) estimado é em torno de R$ {round(resultado2['VALOR_TRN'].iloc[0])},00 com uma margem de erro de +/- 200000,00 reais.")




