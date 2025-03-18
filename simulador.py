# =============================================================================
                                ## BIBLIOTECAS ##
# =============================================================================
import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import time
import numpy as np

# =============================================================================
                                ## FUNÇÕES ##
# =============================================================================
def prever_cluster(df_cli, kmeans_model):
    """
    Recebe as informações do cliente e retorna o cluster ao qual ele pertence.
    """

    df = df_cli.copy()

    df = df.drop('VALOR_MEDIO_SAIDA', axis=1)

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

    pop_PERFIL_SAIDA_dict = {
        'MUITO PEQUENA': 1, 
        'PEQUENA': 2,
        'MEDIA': 3,
        'GRANDE': 4,
        'MUITO GRANDE': 5
    } 


    df['POP_PORTE'] = df['POP_PORTE'].map(pop_porte_dict)
    df['PORTE_VALOR'] = df['PORTE_VALOR'].map(pop_forte_dict)
    df['PORT_CHECK'] = df['PORT_CHECK'].map(pop_CHECK_dict)
    df['TEMPO_DE_ABERTURA'] = df['TEMPO_DE_ABERTURA'].map(pop_tempo_dict)
    df['PERFIL_SAIDA'] = df['PERFIL_SAIDA'].map(pop_PERFIL_SAIDA_dict)

    df = pd.get_dummies(df, columns=['UF'])

    ordem = pd.DataFrame(columns = ['POP_PORTE', 'PORTE_VALOR', 'PORT_CHECK', 'TEMPO_DE_ABERTURA', 'PERFIL_SAIDA',
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

    df["cluster"] = df["cluster"] + 1

    return df


def prever_cpr(df_cli, model, scaler_robust_entrada, scaler_robust_saida):
    """
    Recebe as informações do cliente e retorna o o valor previsto de saida do cpr.
    """

    df = df_cli.copy()

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

    pop_PERFIL_SAIDA_dict = {
        'MUITO PEQUENA': 1, 
        'PEQUENA': 2,
        'MEDIA': 3,
        'GRANDE': 4,
        'MUITO GRANDE': 5
    } 

    df['POP_PORTE'] = df['POP_PORTE'].map(pop_porte_dict)
    df['PORTE_VALOR'] = df['PORTE_VALOR'].map(pop_forte_dict)
    df['PORT_CHECK'] = df['PORT_CHECK'].map(pop_CHECK_dict)
    df['TEMPO_DE_ABERTURA'] = df['TEMPO_DE_ABERTURA'].map(pop_tempo_dict)
    df['PERFIL_SAIDA'] = df['PERFIL_SAIDA'].map(pop_PERFIL_SAIDA_dict)

    df = pd.get_dummies(df, columns=['UF'])
    
    ordem = pd.DataFrame(columns = ['POP_PORTE', 'PORTE_VALOR', 'PORT_CHECK', 'TEMPO_DE_ABERTURA', 'PERFIL_SAIDA', 'VALOR_MEDIO_SAIDA', 
                                    'UF_AL', 'UF_AM', 'UF_AP', 'UF_BA', 'UF_CE', 'UF_DF', 'UF_ES', 'UF_GO', 'UF_MA',
                                    'UF_MG', 'UF_MS', 'UF_MT', 'UF_PA', 'UF_PB', 'UF_PE', 'UF_PI', 'UF_PR',
                                    'UF_RJ', 'UF_RN', 'UF_RO', 'UF_RR', 'UF_RS', 'UF_SC', 'UF_SE', 'UF_SP']) 


    for col in ordem.columns:
            if col not in df.columns:
                df[col] = False
    df = df[ordem.columns]


    # Ajustar e transformar
    df.loc[:, 'VALOR_MEDIO_SAIDA'] = scaler_robust_entrada.transform(df[['VALOR_MEDIO_SAIDA']])


    # Predição do cluster
    df['VALOR_CPR_SAIDA'] = model.predict(df)


    # Reverter a Padronização da Previsão
    df['VALOR_CPR_SAIDA'] = (
    scaler_robust_saida.inverse_transform(df[['VALOR_CPR_SAIDA']].values)
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
scaler_robust_entrada = data['scaler_robust_entrada']
scaler_robust_saida = data['scaler_robust_saida']


# Interface Streamlit
st.title("Simulador do Perfil dos Clientes")
st.write("Preencha os dados do cliente para prever a qual cluster ele pertence.")


# Entradas do usuário
pop_porte = st.selectbox("Selecione o porte populacional do município:", options=[
    '0. Até 20 mil habitantes',
    '1. De 20 mil a 50 mil habitantes',
    '2. De 50 mil a 100 mil habitantes',
    '3. De 100 mil a 300 mil habitantes',
    '4. De 300 mil a 1 milhão de habitantes',
    '5. Acima de 1 milhão de habitantes'
])

porte = st.selectbox("Selecione o porte da loja:", options=[
    'Microempresa',
    'Empresa de pequeno porte',
    'Empresa de médio e grande porte'
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

valor_saida = st.number_input("Valor mensal aproximado de Saída da Loja:", min_value=0.0, value=100000.0, step=0.1)

if valor_saida == 1000000:
    valor_saida = 1000001


# DataFrame com os dados inseridos pelo usuário
cliente = {
    'POP_PORTE': pop_porte,
    'PORTE': porte,
    'PORT_CHECK': port_check,
    'TEMPO_DE_ABERTURA': tempo_abertura,
    'UF': uf,
    'VALOR_MEDIO_SAIDA': valor_saida
    }
df_cliente = pd.DataFrame([cliente])

mapeamento_porte = {
    'Microempresa': 'MICRO EMPRESA',
    'Empresa de pequeno porte': 'EMPRESA DE PEQUENO PORTE',
    'Empresa de médio e grande porte': 'EMPRESA DE MEDIO E GRANDE PORTE'
}
df_cliente['PORTE'] = df_cliente['PORTE'].apply(lambda x: mapeamento_porte.get(x, None))

df_cliente['PORTE_VALOR'] = df_cliente['PORTE'].apply(lambda x: 
    '1. De 0.0 a 360 Mil' if x == 'MICRO EMPRESA' else 
    '2. De 360 Mil a 4.8 Milhoes' if x == 'EMPRESA DE PEQUENO PORTE' else 
    '3. Acima de 4.8 Milhoes' if x == 'EMPRESA DE MEDIO E GRANDE PORTE' else 
    None)
df_cliente = df_cliente.drop('PORTE', axis=1)


mapeamento_pop_porte = {
    '0. Até 20 mil habitantes': '0. Ate 20 Mil Habitantes',
    '1. De 20 mil a 50 mil habitantes': '1. De 20 a 50 Mil Habitantes',
    '2. De 50 mil a 100 mil habitantes': '2. De 50 a 100 Mil Habitantes',
    '3. De 100 mil a 300 mil habitantes': '3. De 100 a 300 Mil Habitantes',
    '4. De 300 mil a 1 milhão de habitantes': '4. De 300 a 1 Milhao de Habitantes',
    '5. Acima de 1 milhão de habitantes': '5. Acima de 1 Milhao de Habitantes'
}
df_cliente['POP_PORTE'] = df_cliente['POP_PORTE'].apply(lambda x: mapeamento_pop_porte.get(x, None))

condicoes = [
    df_cliente["VALOR_MEDIO_SAIDA"] < 50000,
    (df_cliente["VALOR_MEDIO_SAIDA"] >= 50000) & (df_cliente["VALOR_MEDIO_SAIDA"] < 100000),
    (df_cliente["VALOR_MEDIO_SAIDA"] >= 100000) & (df_cliente["VALOR_MEDIO_SAIDA"] < 500000),
    (df_cliente["VALOR_MEDIO_SAIDA"] >= 500000) & (df_cliente["VALOR_MEDIO_SAIDA"] < 1000000),
    df_cliente["VALOR_MEDIO_SAIDA"] > 1000000
]
valores = ["MUITO PEQUENA", "PEQUENA", "MEDIA", "GRANDE", "MUITO GRANDE"]
df_cliente["PERFIL_SAIDA"] = np.select(condicoes, valores, default="DESCONHECIDO")



# Botão de Previsão
if st.button('Fazer previsão'):
    with st.spinner('Fazendo a previsão...'):
        time.sleep(2) 
        resultado = prever_cluster(df_cliente, kmeans) 
        resultado1 = prever_cpr(df_cliente, modelo, scaler_robust_entrada, scaler_robust_saida) 

        st.write(f"Cliente de Perfil {resultado['cluster'].iloc[0]}.")

        valor_previsto = resultado1['VALOR_CPR_SAIDA'].iloc[0]
        valor_medio_saida = df_cliente['VALOR_MEDIO_SAIDA'].iloc[0]

        if df_cliente.isna().sum().sum() > 0:  # Verifica se há NaN no DataFrame
            st.write(f"Não é possível estimar o valor do CPR.")
        elif (valor_medio_saida<1000):
            st.write(f"Não é possivel estimado o valor do CPR.")            
        elif (valor_previsto/valor_medio_saida < 0.1):
            valor_previsto = 0.15 * valor_medio_saida 
            st.write(f"O Valor Mensal de Saída do CPR estimado fica em torno de R$ {round(valor_previsto)}.")                     
        elif (0.8<= valor_previsto/valor_medio_saida < 1) :
            valor_previsto = valor_previsto * valor_previsto/valor_medio_saida
            st.write(f"O Valor Mensal de Saída do CPR estimado fica em torno de R$ {round(valor_previsto)}.")        
        elif (1.0 <= valor_previsto/valor_medio_saida < 1.5):
            valor_previsto = valor_previsto * (valor_previsto/valor_medio_saida-1)
            st.write(f"O Valor Mensal de Saída do CPR estimado fica em torno de R$ {round(valor_previsto)}.")        
        elif (valor_previsto/valor_medio_saida >= 1.5):
            valor_previsto = valor_previsto * (valor_medio_saida/(valor_previsto+1000))
            st.write(f"O Valor Mensal de Saída do CPR estimado fica em torno de R$ {round(valor_previsto)}.")        
        else:
            st.write(f"O Valor Mensal de Saída do CPR estimado fica em torno de R$ {round(valor_previsto)}.")
        
        


        



