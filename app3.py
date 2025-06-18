import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from imblearn.over_sampling import SMOTE

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Prevenção de Cancelamentos",
    page_icon="🏨",
    layout="wide"
)
sns.set_style("whitegrid")

# ==============================================================================
# DICIONÁRIOS DE TRADUÇÃO E MAPEAMENTO (DEFINIDOS NO INÍCIO)
# ==============================================================================
VAR_TRANSLATIONS = {
    'lead_time': 'Antecedência da Reserva (dias)',
    'total_of_special_requests': 'Nº de Pedidos Especiais',
    'required_car_parking_spaces': 'Vaga de Garagem Solicitada',
    'booking_changes': 'Nº de Alterações na Reserva',
    'previous_cancellations': 'Nº de Cancelamentos Anteriores',
    'is_repeated_guest': 'Cliente é Recorrente',
    'adr': 'Preço Médio por Noite (€)',
    'total_nights': 'Total de Noites da Estadia',
    'previous_bookings_not_canceled': 'Reservas Anteriores Válidas',
    'adults': 'Nº de Adultos',
    'children': 'Nº de Crianças',
    'babies': 'Nº de Bebês',
    'is_canceled': 'É Cancelado',
    'total_guests': 'Total de Hóspedes (Adultos + Crianças + Bebês)',
    'is_agent_booking': 'Reserva Feita por Agente',
    'is_company_booking': 'Reserva de Empresa',
    'is_weekend_stay': 'Estadia Inclui Fim de Semana',
    'children_present': 'Presença de Crianças/Bebês',
    'assigned_room_type_Changed': 'Tipo de Quarto Atribuído Diferente do Reservado',

    # Tradução para categorias ORIGINAIS (para os selectbox do simulador e para display da análise)
    'hotel': 'Tipo de Hotel',
    'meal': 'Regime de Refeição',
    'market_segment': 'Segmento de Mercado',
    'distribution_channel': 'Canal de Distribuição',
    'customer_type': 'Tipo de Cliente',
    'deposit_type': 'Tipo de Depósito',
    'assigned_room_type': 'Tipo de Quarto Atribuído',
    'reserved_room_type': 'Tipo de Quarto Reservado',
    'country_grouped': 'País de Origem',

    # Tradução para as Dummies (nomes exatos gerados por pd.get_dummies, que são usados no modelo e VAR_TRANSLATIONS)
    'hotel_City Hotel': 'Hotel: Cidade',
    'hotel_Resort Hotel': 'Hotel: Resort',

    'meal_BB': 'Regime de Refeição: Café da Manhã',
    'meal_FB': 'Regime de Refeição: Pensão Completa',
    'meal_HB': 'Regime de Refeição: Meia Pensão',
    'meal_SC': 'Regime de Refeição: Sem Refeição',
    'meal_Undefined': 'Regime de Refeição: Indefinido',

    'market_segment_Aviation': 'Segmento: Aviação',
    'market_segment_Complementary': 'Segmento: Cortesia',
    'market_segment_Corporate': 'Segmento: Corporativo',
    'market_segment_Direct': 'Segmento: Direto',
    'market_segment_Groups': 'Segmento: Grupos',
    'market_segment_Online TA': 'Segmento: Agência Online (OTA)',
    'market_segment_Offline TA/TO': 'Segmento: Agência Offline/Operadora',
    'market_segment_Undefined': 'Segmento: Indefinido',

    'distribution_channel_Corporate': 'Distribuição: Corporativa',
    'distribution_channel_Direct': 'Distribuição: Direto',
    'distribution_channel_GDS': 'Distribuição: GDS',
    'distribution_channel_TA/TO': 'Distribuição: Agência/Operadora',
    'distribution_channel_Undefined': 'Distribuição: Indefinida',

    'customer_type_Contract': 'Tipo de Cliente: Contrato',
    'customer_type_Group': 'Tipo de Cliente: Grupo Fechado',
    'customer_type_Transient': 'Tipo de Cliente: Avulso',
    'customer_type_Transient-Party': 'Tipo de Cliente: Grupo Avulso',

    'deposit_type_Non Refund': 'Depósito: Não Reembolsável',
    'deposit_type_No Deposit': 'Depósito: Sem Depósito',
    'deposit_type_Refundable': 'Depósito: Reembolsável',

    # Tipos de quarto designados
    'assigned_room_type_A': 'Quarto Designado: A',
    'assigned_room_type_B': 'Quarto Designado: B',
    'assigned_room_type_C': 'Quarto Designado: C',
    'assigned_room_type_D': 'Quarto Designado: D',
    'assigned_room_type_E': 'Quarto Designado: E',
    'assigned_room_type_F': 'Quarto Designado: F',
    'assigned_room_type_G': 'Quarto Designado: G',
    'assigned_room_type_H': 'Quarto Designado: H',
    'assigned_room_type_I': 'Quarto Designado: I',
    'assigned_room_type_K': 'Quarto Designado: K',
    'assigned_room_type_L': 'Quarto Designado: L',
    'assigned_room_type_P': 'Quarto Designado: P',

    # Tipos de quarto reservado
    'reserved_room_type_A': 'Quarto Reservado: A',
    'reserved_room_type_B': 'Quarto Reservado: B',
    'reserved_room_type_C': 'Quarto Reservado: C',
    'reserved_room_type_D': 'Quarto Reservado: D',
    'reserved_room_type_E': 'Quarto Reservado: E',
    'reserved_room_type_F': 'Quarto Reservado: F',
    'reserved_room_type_G': 'Quarto Reservado: G',
    'reserved_room_type_H': 'Quarto Reservado: H',
    'reserved_room_type_L': 'Quarto Reservado: L',
    'reserved_room_type_P': 'Quarto Reservado: P',

    # Países agrupados - Chaves aqui DEVEM ser country_grouped_CODIGO_PAIS
    'country_grouped_PRT': 'País: Portugal',
    'country_grouped_GBR': 'País: Reino Unido',
    'country_grouped_FRA': 'País: França',
    'country_grouped_ESP': 'País: Espanha',
    'country_grouped_DEU': 'País: Alemanha',
    'country_grouped_IRL': 'País: Irlanda',
    'country_grouped_USA': 'País: EUA',
    'country_grouped_BRA': 'País: Brasil',
    'country_grouped_CAN': 'País: Canadá',
    'country_grouped_NLD': 'País: Holanda',
    'country_grouped_ITA': 'País: Itália',
    'country_grouped_BEL': 'País: Bélgica',
    'country_grouped_CHE': 'País: Suíça',
    'country_grouped_AUT': 'País: Áustria',
    'country_grouped_SWE': 'País: Suécia',
    'country_grouped_CHN': 'País: China',
    'country_grouped_JPN': 'País: Japão',
    'country_grouped_AUS': 'País: Austrália',
    'country_grouped_MEX': 'País: México',
    'country_grouped_RUS': 'País: Rússia',
    'country_grouped_OTHER_COUNTRY': 'País: Outros Países',
}

# Mapeamento para facilitar a seleção de variáveis categóricas no simulador
# Associa o nome da coluna original a um DICIONÁRIO de suas categorias (string como no CSV original)
# e as traduções para as opções do selectbox.
CATEGORICAL_COLS_MAP = {
    'hotel': {'City Hotel': 'Hotel na Cidade', 'Resort Hotel': 'Hotel Resort'},
    'meal': {
        'BB': 'Café da Manhã', 'FB': 'Pensão Completa', 'HB': 'Meia Pensão',
        'SC': 'Sem Refeição', 'Undefined': 'Indefinido'
    },
    'market_segment': {
        'Aviation': 'Aviação', 'Complementary': 'Cortesia', 'Corporate': 'Corporativo',
        'Direct': 'Direto', 'Groups': 'Grupos', 'Offline TA/TO': 'Agência Offline / Operadora',
        'Online TA': 'Agência Online (OTA)', 'Undefined': 'Indefinido'
    },
    'distribution_channel': {
        'Corporate': 'Corporativa', 'Direct': 'Direto', 'GDS': 'GDS',
        'TA/TO': 'Agência de Viagem / Operadora', 'Undefined': 'Indefinida'
    },
    'customer_type': {
        'Contract': 'Contrato', 'Group': 'Grupo Fechado', 'Transient': 'Avulso',
        'Transient-Party': 'Grupo Avulso'
    },
    'deposit_type': {
        'No Deposit': 'Sem Depósito', 'Non Refund': 'Não Reembolsável',
        'Refundable': 'Reembolsável'
    },
    'assigned_room_type': { # Descrições Sugeridas - Ajuste conforme sua interpretação
        'A': 'Quarto Tipo A (Padrão/Básico)', 'B': 'Quarto Tipo B (Econômico/Simples)', 'C': 'Quarto Tipo C (Conforto/Médio)',
        'D': 'Quarto Tipo D (Superior/Amplo)', 'E': 'Quarto Tipo E (Luxo/Premium)', 'F': 'Quarto Tipo F (Familiar/Adaptado)',
        'G': 'Quarto Tipo G (Grande/Família Maior)', 'H': 'Quarto Tipo H (Suíte/Executiva)', 'I': 'Quarto Tipo I (Acessível/Especial)',
        'K': 'Quarto Tipo K (Com Cozinha/Kitnet)', 'L': 'Quarto Tipo L (Suíte Presidencial/Cobertura)', 'P': 'Quarto Tipo P (Promocional/Temporário)'
    },
    'reserved_room_type': { # Descrições Sugeridas - Ajuste conforme sua interpretação
        'A': 'Quarto Tipo A (Padrão/Básico)', 'B': 'Quarto Tipo B (Econômico/Simples)', 'C': 'Quarto Tipo C (Conforto/Médio)',
        'D': 'Quarto Tipo D (Superior/Amplo)', 'E': 'Quarto Tipo E (Luxo/Premium)', 'F': 'Quarto Tipo F (Familiar/Adaptado)',
        'G': 'Quarto Tipo G (Grande/Família Maior)', 'H': 'Quarto Tipo H (Suíte/Executiva)', 'L': 'Quarto Tipo L (Suíte Presidencial/Cobertura)',
        'P': 'Quarto Tipo P (Promocional/Temporário)'
    },
    'country_grouped': { # Mapeamento para as categorias agrupadas
        'PRT': 'Portugal', 'GBR': 'Reino Unido', 'FRA': 'França', 'ESP': 'Espanha',
        'DEU': 'Alemanha', 'IRL': 'Irlanda', 'USA': 'EUA', 'BRA': 'Brasil', 'CAN': 'Canadá',
        'NLD': 'Holanda', 'ITA': 'Itália', 'BEL': 'Bélgica', 'CHE': 'Suíça', 'AUT': 'Áustria',
        'SWE': 'Suécia', 'CHN': 'China', 'JPN': 'Japão', 'AUS': 'Austrália', 'MEX': 'México',
        'RUS': 'Rússia', 'OTHER_COUNTRY': 'Outros Países'
    },
}
# ==============================================================================
# FUNÇÕES EM CACHE
# ==============================================================================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('hotel_bookings.csv')

    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['agent'] = df['agent'].fillna(0)

    if 'company' in df.columns:
        df['is_company_booking'] = (df['company'].notna()).astype(int)
    else:
        df['is_company_booking'] = 0

    df = df.drop(columns=['company'], errors='ignore')

    df['is_agent_booking'] = (df['agent'] != 0).astype(int)

    df = df.dropna(subset=['adr'])
    df = df[df['adr'] >= 0]
    df = df[~((df['adults']==0) & (df['children']==0) & (df['babies']==0))]
    df = df[~((df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0))]
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    df['is_weekend_stay'] = (df['stays_in_weekend_nights'] > 0).astype(int)
    df['children_present'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['assigned_room_type_Changed'] = (df['assigned_room_type'] != df['reserved_room_type']).astype(int)

    df_model = df[['is_canceled']].copy()

    numerical_cols = ['lead_time', 'total_of_special_requests', 'required_car_parking_spaces',
                      'booking_changes', 'previous_cancellations', 'is_repeated_guest',
                      'adr', 'total_nights', 'previous_bookings_not_canceled',
                      'adults', 'children', 'babies', 'is_agent_booking',
                      'is_company_booking', 'is_weekend_stay', 'children_present', 'total_guests']

    for col in numerical_cols:
        if col in df.columns:
            df_model[col] = df[col]

    top_countries = df['country'].value_counts().nlargest(15).index.tolist()
    df['country_grouped'] = df['country'].apply(lambda x: x if x in top_countries else 'OTHER_COUNTRY')

    categorical_cols_for_dummies = [
        'market_segment', 'deposit_type', 'customer_type', 'distribution_channel',
        'hotel', 'meal', 'assigned_room_type', 'reserved_room_type', 'country_grouped'
    ]
    df_dummies = pd.get_dummies(df[categorical_cols_for_dummies], drop_first=False, dtype=int)

    df_dummies.columns = [col.replace(' ', '_').replace('/', '_').replace('-', '_') for col in df_dummies.columns]

    df_model = pd.concat([df_model, df_dummies], axis=1)
    df_model.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df_model.columns]

    for col in df_model.columns:
        if df_model[col].nunique() == 1 and col != 'is_canceled':
            df_model = df_model.drop(columns=[col])

    return df_model

@st.cache_resource
def train_model(data, selected_features_list):
    y = data['is_canceled']

    # Lógica de remoção de features para evitar multicolinearidade perfeita com total_guests
    final_features_for_model = []

    if 'total_guests' in selected_features_list:
        # Se 'total_guests' está selecionado, garantir que ele seja o único representante do número de pessoas.
        if 'total_guests' in data.columns and data['total_guests'].nunique() > 1:
            final_features_for_model.append('total_guests')

        # Adicionar outras features selecionadas, exceto os componentes de total_guests
        for feature in selected_features_list:
            if feature not in ['adults', 'children', 'babies', 'total_guests']:
                final_features_for_model.append(feature)

    else: # Se 'total_guests' NÃO está selecionado, então incluir 'adults', 'children', 'babies' se foram selecionados.
        final_features_for_model = list(selected_features_list) # Começa com tudo que o usuário selecionou

    # Garantir que todas as features no final_features_for_model realmente existam no 'data' e não sejam constantes.
    final_selected_features = [f for f in final_features_for_model if f in data.columns and data[f].nunique() > 1]

    if not final_selected_features:
        st.error("Nenhuma variável válida selecionada para o treinamento do modelo após a filtragem de colunas constantes. Isso pode acontecer se você selecionou apenas 'Nº de Adultos', 'Nº de Crianças' ou 'Nº de Bebês' sem 'Total de Hóspedes' e estas colunas individuais não possuem variância nos dados, ou se a combinação delas causa problemas. Por favor, ajuste sua seleção.")
        return None

    X = data[final_selected_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if X_train.empty:
        st.error("X_train está vazio. Isso pode ocorrer se as variáveis selecionadas não possuem variância nos dados de treino. Por favor, selecione outras variáveis.")
        return None

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    constant_cols_smote = [col for col in X_train_smote.columns if X_train_smote[col].nunique() == 1]
    if constant_cols_smote:
        st.warning(f"As seguintes colunas foram removidas do treino do modelo porque se tornaram constantes após SMOTE/rebalanceamento: {', '.join(constant_cols_smote)}. Isso pode afetar os resultados do VIF e do modelo.")
        X_train_smote = X_train_smote.drop(columns=constant_cols_smote)
        X_test = X_test.drop(columns=constant_cols_smote, errors='ignore')

    if X_train_smote.empty:
        st.error("X_train_smote está vazio após a remoção de colunas constantes. Não é possível treinar o modelo com as variáveis selecionadas.")
        return None

    X_train_smote_const = sm.add_constant(X_train_smote, has_constant='add')

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sm.tools.sm_exceptions.ConvergenceWarning)
            try:
                logit_model = sm.Logit(y_train_smote.astype(float), X_train_smote_const.astype(float)).fit(method='bfgs', maxiter=5000, disp=0)
            except Exception as e:
                st.error(f"Erro ao treinar o modelo: {e}. Isso pode ser causado por multicolinearidade perfeita, separação completa dos dados, ou um problema de otimização. Tente selecionar um conjunto diferente de variáveis.")
                st.warning(f"Detalhes técnicos do erro: {e}")
                return None

    return {
        "model": logit_model,
        "selected_features": final_selected_features,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_smote_mean": X_train_smote.mean(),
        "X_train_smote_max": X_train_smote.max(),
        "X_train_smote_min": X_train_smote.min()
    }
# ==============================================================================
# CABEÇALHO DA APLICAÇÃO
# ==============================================================================
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://em-content.zobj.net/source/microsoft/379/hotel_1f3e8.png", width=120)
with col2:
    st.title("Painel Estratégico de Prevenção de Cancelamentos")
    st.markdown("Uma ferramenta de Business Intelligence para entender *por que* reservas são canceladas e *como* agir para reduzir perdas.")
    st.markdown("Desenvolvido por: **Pedro Russo e Daniel Vianna**")

# --- CARREGAMENTO E TREINAMENTO ---
data = load_and_preprocess_data()

# Preparar a lista de todas as features disponíveis para seleção na sidebar
all_available_features_in_data = [col for col in data.columns if col != 'is_canceled']

all_features_translated_dict = {}
for col in all_available_features_in_data:
    if col in VAR_TRANSLATIONS: # Se a coluna tem uma tradução direta no VAR_TRANSLATIONS
        all_features_translated_dict[VAR_TRANSLATIONS[col]] = col
    else:
        found_translation = False
        for original_col, categories_map in CATEGORICAL_COLS_MAP.items(): # Iterar sobre CATEGORICAL_COLS_MAP para dummies
            for cat_original, cat_translated_display in categories_map.items():
                # Constrói o nome da dummy como aparece no DataFrame `data`
                dummy_name_in_data = f"{original_col}_{cat_original.replace(' ', '_').replace('/', '_').replace('-', '_')}"

                if col == dummy_name_in_data: # Se o nome da coluna corresponde a uma dummy gerada
                    # A chave para o all_features_translated_dict será o nome da FEATURE ORIGINAL + a tradução da CATEGORIA
                    # Ex: 'Hotel: Hotel na Cidade'
                    all_features_translated_dict[f"{VAR_TRANSLATIONS.get(original_col, original_col.replace('_', ' ').title())}: {cat_translated_display}"] = col
                    found_translation = True
                    break
            if found_translation:
                break
        if not found_translation: # Último fallback para qualquer coluna não traduzida
             all_features_translated_dict[col.replace('_', ' ').title()] = col

st.sidebar.header("🔧 Construção do Modelo Preditivo")

# Para que o app não quebre na primeira execução (antes de clicar no botão)
model_artifacts = None 

# --- Início do Formulário ---
with st.sidebar.form(key='form_parametros'):
    st.markdown("**Configure os parâmetros e clique em 'Analisar' para rodar o modelo.**")
    
    # Lista de variáveis padrão (mesma lógica de antes, mas dentro do contexto)
    default_selected_features_translated_keys = [
        'Antecedência da Reserva (dias)', 'Nº de Pedidos Especiais', 'Vaga de Garagem Solicitada',
        'Nº de Alterações na Reserva', 'Nº de Cancelamentos Anteriores', 'Cliente é Recorrente',
        'Preço Médio por Noite (€)', 'Total de Noites da Estadia', 'Reservas Anteriores Válidas',
        'Nº de Adultos', 'Nº de Crianças', 'Nº de Bebês', 'Reserva Feita por Agente',
        'Reserva de Empresa', 'Estadia Inclui Fim de Semana', 'Presença de Crianças/Bebês', 'Total de Hóspedes (Adultos + Crianças + Bebês)',
        'Depósito: Não Reembolsável', 'Depósito: Sem Depósito', 'Depósito: Reembolsável',
        'Segmento: Agência Online (OTA)', 'Segmento: Grupos', 'Segmento: Direto',
        'Tipo de Cliente: Avulso', 'Tipo de Cliente: Grupo Fechado', 'Tipo de Cliente: Contrato',
        'Distribuição: Agência/Operadora', 'Distribuição: Direto', 'Distribuição: Corporativa',
        'Hotel: Cidade', 'Hotel: Resort',
        'Regime de Refeição: Café da Manhã', 'Regime de Refeição: Sem Refeição', 'Regime de Refeição: Pensão Completa', 'Regime de Refeição: Meia Pensão',
        'Quarto Designado: A', 'Tipo de Quarto Atribuído Diferente do Reservado',
        'Quarto Reservado: A', 'Quarto Reservado: B',
        'País: Portugal', 'País: Reino Unido', 'País: EUA', 'País: Brasil', 'País: Outros Países'
    ]
    default_selected_translated = [
        t for t in default_selected_features_translated_keys if t in all_features_translated_dict
    ]

    # Widget de seleção de variáveis
    selected_features_translated = st.multiselect(
        "1. Fatores para Análise:",
        options=sorted(all_features_translated_dict.keys()),
        default=default_selected_translated
    )

    st.markdown("---")

    # Widgets do RFE
    st.markdown("**2. Refinamento com RFE (Opcional)**")
    use_rfe = st.checkbox("Usar RFE para refinar a seleção de variáveis?", value=False)
    num_features_rfe = 1
    if use_rfe:
        num_features_rfe = st.slider(
            "Quantas variáveis o RFE deve selecionar?",
            min_value=1,
            max_value=len(selected_features_translated) if selected_features_translated else 1,
            value=min(8, len(selected_features_translated)) if selected_features_translated else 1,
            step=1,
            help="O RFE avaliará todas as variáveis que você selecionou e manterá apenas o número de fatores mais impactantes que você definir aqui."
        )

    # Botão de submissão do formulário
    st.markdown("---")
    submitted = st.form_submit_button("✅ Analisar com Fatores Selecionados")

# --- Fim do Formulário ---

# A lógica principal do app SÓ RODA DEPOIS que o botão do formulário é clicado
if submitted:
    if not selected_features_translated:
        st.error("Por favor, selecione ao menos uma variável para a análise na barra lateral.")
        st.stop()

    selected_features = [all_features_translated_dict[t] for t in selected_features_translated]
    final_features_for_model_training = selected_features

    if use_rfe:
        if len(selected_features) >= 2:
            with st.spinner("Executando RFE para encontrar os melhores fatores..."):
                X_rfe = data[selected_features]
                y_rfe = data['is_canceled']
                rfe_model = LogisticRegression(max_iter=1000, solver='liblinear')
                rfe_selector = RFE(estimator=rfe_model, n_features_to_select=num_features_rfe)
                rfe_selector.fit(X_rfe, y_rfe)
                rfe_selected_mask = rfe_selector.get_support()
                final_features_for_model_training = X_rfe.columns[rfe_selected_mask].tolist()

                original_to_translated_map = {v: k for k, v in all_features_translated_dict.items()}
                rfe_features_translated = [original_to_translated_map[f] for f in final_features_for_model_training]
                st.sidebar.success(f"RFE selecionou as seguintes {len(rfe_features_translated)} variáveis para o modelo:")
                st.sidebar.dataframe(pd.DataFrame({'Fatores Selecionados pelo RFE': sorted(rfe_features_translated)}), use_container_width=True)

    with st.spinner("Treinando modelo e gerando análises... Por favor, aguarde."):
        model_artifacts = train_model(data, final_features_for_model_training)

# Se o botão ainda não foi apertado, o restante do código não deve rodar
if model_artifacts is None:
    st.info("⬅️ Configure os parâmetros na barra lateral e clique em 'Analisar' para gerar os resultados.")
    st.stop()

if model_artifacts is None:
    st.stop()

model = model_artifacts["model"]
X_train = model_artifacts["X_train"]
X_test = model_artifacts["X_test"]
y_test = model_artifacts["y_test"]
X_train_smote_mean = model_artifacts["X_train_smote_mean"]
X_train_smote_max = model_artifacts["X_train_smote_max"]
X_train_smote_min = model_artifacts["X_train_smote_min"]


# ==============================================================================
# ABAS DA APLICAÇÃO
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "✅ Confiança no Modelo",
    "🎯 Análise de Impacto",
    "⚙️ Simulador de Cenários",
    "💡 Plano de Ação Personalizado"
])

# ==============================================================================
# ABA 1: CONFIANÇA NO MODELO
# ==============================================================================
with tab1:
    st.header("🔍 O Modelo é Confiável?")
    st.markdown("Antes de confiar nas previsões, validamos a saúde do nosso modelo com os testes exigidos pela tarefa.")
    with st.expander("Clique para ver os detalhes dos diagnósticos técnicos"):
        st.markdown(f"- *Seleção de Variáveis:* O modelo foi construído com os **{len(model.params) - 1 if 'const' in model.params else len(model.params)} fatores** que você selecionou (e opcionalmente refinou com RFE).")
        st.markdown("- *Balanceamento:* O desequilíbrio entre cancelamentos e não-cancelamentos foi corrigido com SMOTE (Synthetic Minority Over-sampling Technique).")

        st.subheader("Teste de Multicolinearidade (VIF)")
        st.markdown("""
        O **Fator de Inflação da Variância (VIF)** mede o quão correlacionada uma variável explicativa é com as outras variáveis explicativas no modelo.
        * **VIFs baixos (próximos de 1):** Indicam pouca ou nenhuma multicolinearidade. A variável não é bem explicada pelas outras.
        * **VIFs entre 5 e 10:** Sugerem multicolinearidade moderada, que pode ser tolerável, mas deve ser observada.
        * **VIFs acima de 10:** Indicam alta multicolinearidade, o que significa que a variável pode ser quase perfeitamente explicada por uma combinação linear de outras variáveis. Isso pode tornar os coeficientes do modelo instáveis e difíceis de interpretar. 'Indefinido' significa multicolinearidade perfeita.
        """)

        X_const_for_vif = sm.add_constant(X_train, has_constant='add')

        numeric_cols_for_vif = [col for col in X_const_for_vif.columns if pd.api.types.is_numeric_dtype(X_const_for_vif[col]) and X_const_for_vif[col].nunique() > 1]

        vif_data = pd.DataFrame()
        if 'const' in numeric_cols_for_vif:
            numeric_cols_for_vif.remove('const')

        if len(numeric_cols_for_vif) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                vif_data["feature"] = numeric_cols_for_vif
                vif_data["VIF"] = [variance_inflation_factor(X_const_for_vif[numeric_cols_for_vif].values.astype(float), i)
                                   for i in range(len(numeric_cols_for_vif))]

            vif_data['feature'] = vif_data['feature'].map(VAR_TRANSLATIONS)

            vif_data['Sortable_VIF'] = pd.to_numeric(vif_data['VIF'], errors='coerce')

            max_finite_vif = vif_data['Sortable_VIF'].replace([np.inf, -np.inf], np.nan).dropna().max()

            if pd.isna(max_finite_vif):
                vif_data['Sortable_VIF'] = vif_data['Sortable_VIF'].fillna(np.finfo(np.float64).max)
            else:
                vif_data['Sortable_VIF'] = vif_data['Sortable_VIF'].fillna(max_finite_vif + 1e9)

            vif_data['VIF'] = vif_data['VIF'].replace([np.inf, -np.inf], 'Indefinido (Multicolinearidade Perfeita)')

            st.dataframe(vif_data.sort_values(by="Sortable_VIF", ascending=False).drop(columns="Sortable_VIF"))
            st.caption("Valores de VIF acima de 5-10 podem indicar multicolinearidade preocupante, o que pode afetar a estabilidade e a interpretação dos coeficientes do modelo. 'Indefinido' significa multicolinearidade perfeita. Valores mais baixos são preferíveis. O modelo Logístico do Statsmodels, contudo, é robusto a certo grau de multicolinearidade.")
        else:
            st.info("O cálculo do VIF não foi realizado porque ele requer ao menos duas variáveis numéricas no modelo para comparar a colinearidade entre elas.")


        st.subheader("Gráficos de Curva Logística para Variáveis Chave")

        numeric_selected_features = [f for f in final_features_for_model_training if data[f].nunique() > 2 and f in X_train.columns]

        if len(numeric_selected_features) >= 3:
            plot_features = numeric_selected_features[:3]
        elif len(numeric_selected_features) > 0:
            plot_features = numeric_selected_features
        else:
            plot_features = []
            st.info("Não há variáveis contínuas suficientes selecionadas para plotar a curva logística.")

        for feature in plot_features:
            fig, ax = plt.subplots(figsize=(8, 5))

            x_min = X_train[feature].min()
            x_max = X_train[feature].max()
            x_values = np.linspace(x_min, x_max, 100)

            mean_values = X_train_smote_mean.copy()
            mean_values['const'] = 1.0

            temp_df = pd.DataFrame(np.tile(mean_values.values, (len(x_values), 1)), columns=mean_values.index)
            temp_df[feature] = x_values

            cols_to_use = [col for col in model.params.index if col in temp_df.columns]
            temp_df = temp_df[cols_to_use]

            for col in model.params.index:
                if col not in temp_df.columns:
                    temp_df[col] = 0.0

            temp_df = temp_df[model.params.index]

            with np.errstate(over='ignore'):
                y_pred_proba_plot = model.predict(temp_df.astype(float))

            ax.plot(x_values, y_pred_proba_plot, color='blue')
            ax.set_xlabel(VAR_TRANSLATIONS.get(feature, feature))
            ax.set_ylabel("Probabilidade de Cancelamento")
            ax.set_title(f"Curva Logística para {VAR_TRANSLATIONS.get(feature, feature)}")
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.header("🎯 Performance do Modelo em Ação")
    st.markdown("""
    A performance do modelo é avaliada por métricas que indicam quão bem ele consegue prever os cancelamentos.
    """)

    X_test_const = sm.add_constant(X_test, has_constant='add')

    missing_cols = set(model.params.index) - set(X_test_const.columns)
    for c in missing_cols:
        X_test_const[c] = 0

    X_test_for_predict = X_test_const[model.params.index]

    with np.errstate(over='ignore'):
        y_pred_proba = model.predict(X_test_for_predict.astype(float))

    threshold = st.slider("Selecione o Limiar de Classificação (Threshold)", 0.0, 1.0, 0.5, 0.05)
    y_pred_class = (y_pred_proba >= threshold).astype(int)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Capacidade de Discernimento (AUC)")
        auc_score = roc_auc_score(y_test, y_pred_proba)
        st.metric(label="Nota de Discernimento (de 0 a 1)", value=f"{auc_score:.3f}")
        st.progress(auc_score)
        st.caption("""
        Mede a habilidade do modelo em separar corretamente as classes (reservas que cancelam das que não cancelam).
        * Um AUC de 0.5 significa que o modelo não é melhor que um chute aleatório.
        * Um AUC de 1.0 significa que o modelo é perfeito.
        * **Como é calculado:** É a área sob a Curva ROC (Receiver Operating Characteristic), que plota a Taxa de Verdadeiros Positivos (sensibilidade) versus a Taxa de Falsos Positivos (1 - especificidade) em vários limiares.
        """)
    with c2:
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curva ROC')
        ax.legend(loc="lower right")
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    st.subheader("Métricas de Classificação (com Threshold selecionado)")
    st.markdown("""
    As métricas abaixo dependem do "limiar de classificação" (Threshold) que você escolheu.
    Um Threshold de 0.5 (padrão) significa que se a probabilidade de cancelamento prevista for >= 0.5, o modelo prevê "cancelamento".
    """)
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.metric("Acurácia", f"{accuracy_score(y_test, y_pred_class):.3f}")
        st.caption("""
        Proporção de previsões corretas (acertos totais) sobre o total de observações.

        **Cálculo:** (Verdadeiros Positivos + Verdadeiros Negativos) / Total de Observações.

        *Pontos fortes:* Intuitiva, fácil de entender.

        *Limitações:* Pode ser enganosa em dados desbalanceados (se 95% das reservas não cancelam, um modelo que sempre prevê 'não cancela' terá 95% de acurácia, mas é inútil).
        """)
    with col_metrics2:
        st.metric("Precisão", f"{precision_score(y_test, y_pred_class):.3f}")
        st.caption("""
        Das reservas que o modelo previu como **canceladas**, quantas realmente cancelaram.

        **Foco:** Minimizar falsos positivos (alarmes falsos). Alta precisão significa que, quando o modelo diz 'vai cancelar', ele geralmente está certo. Isso é importante para evitar gastos desnecessários com ações preventivas em reservas que não cancelariam.

        **Cálculo:** Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos).
        """)
    with col_metrics3:
        st.metric("Recall", f"{recall_score(y_test, y_pred_class):.3f}")
        st.caption("""
        Das reservas que **realmente cancelaram**, quantas foram corretamente previstas pelo modelo.

        **Foco:** Minimizar falsos negativos (cancelamentos perdidos). Alto recall significa que o modelo consegue identificar a maioria dos cancelamentos reais. Isso é importante para não perder a oportunidade de aplicar uma ação preventiva e reter o cliente.

        **Cálculo:** Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos).
        """)
    with col_metrics4:
        st.metric("F1-Score", f"{f1_score(y_test, y_pred_class):.3f}")
        st.caption("""
        Média harmônica entre Precisão e Recall. É útil quando há um desequilíbrio entre as classes e você busca um equilíbrio entre minimizar falsos positivos e falsos negativos.

        **Cálculo:** 2 * (Precisão * Recall) / (Precisão + Recall).

        *Interpretação:* Um F1-Score alto indica que o modelo tem boa precisão e recall, retornando poucas previsões incorretas (falsos positivos) e não perdendo muitas das positivas reais (falsos negativos).
        """)

    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred_class)
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Não Cancelou (Previsto)', 'Cancelou (Previsto)'], yticklabels=['Não Cancelou (Real)', 'Cancelou (Real)'])
    ax.set_xlabel('Previsão do Modelo')
    ax.set_ylabel('Valor Real')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)
    st.caption("""
    - **Verdadeiro Negativo (Superior Esquerdo):** Reservas não canceladas, corretamente previstas.
    - **Falso Positivo (Superior Direito):** Reservas não canceladas, *incorretamente* previstas como canceladas (custo de oportunidade, super-alerta).
    - **Falso Negativo (Inferior Esquerdo):** Reservas canceladas, *incorretamente* previstas como não canceladas (custo real de cancelamento, perda de receita).
    - **Verdadeiro Positivo (Inferior Direito):** Reservas canceladas, corretamente previstas (base para ações preventivas eficazes).
    """)
# ==============================================================================
# ABA 2: ANÁLISE DE IMPACTO
# ==============================================================================
with tab2:
    st.header("🎯 Análise de Impacto dos Fatores de Risco")
    st.markdown("Esta seção detalha como cada variável selecionada influencia a probabilidade de uma reserva ser cancelada. Entender esses impactos é crucial para desenvolver estratégias eficazes de prevenção.")

    st.subheader("Análise dos Coeficientes do Modelo (Log-Odds)")
    with st.expander("Clique para entender e ver os coeficientes em Log-Odds"):
        st.markdown("""
        #### O que são Log-Odds e por que são úteis?

        Antes de se tornarem o "Odds Ratio" (que é mais fácil de interpretar), os fatores de impacto são calculados como **Coeficientes de Log-Odds**. Pense neles como o **"motor" do modelo**, mostrando a força e a direção brutas da influência de cada fator.

        * **Como é calculado?** A regressão logística não prevê a probabilidade diretamente. Ela prevê a "Log-Odds" (logaritmo da chance) de um evento acontecer.
            * **Chance (Odds):** É a probabilidade de um evento acontecer dividida pela probabilidade de não acontecer. Por exemplo, se a chance de cancelamento é de 2 para 1, o Odds é 2.
            * **Log-Odds:** É simplesmente o logaritmo dessa chance. O modelo usa essa escala porque ela é matematicamente conveniente para os cálculos.

        * **Como isso me ajuda?** Analisar os Log-Odds permite entender o impacto fundamental de cada variável no modelo:
            * **Coeficiente Positivo (> 0):** É um **fator de risco**. Quanto maior o número, mais forte é seu poder de aumentar a chance de um cancelamento.
            * **Coeficiente Negativo (< 0):** É um **fator de proteção**. Quanto mais negativo o número, mais forte é seu poder de *diminuir* a chance de um cancelamento.
            * **Coeficiente Próximo de Zero (~ 0):** O fator tem pouco ou nenhum efeito.

        A tabela abaixo mostra esses coeficientes. Embora menos intuitivos que o Odds Ratio, eles são a base matemática de toda a análise de impacto que vem a seguir.
        """)

        log_odds_results = pd.DataFrame({
            "Coeficiente (Log-Odds)": model.params,
            "P-valor": model.pvalues,
        }).drop('const', errors='ignore')

        log_odds_results.index = log_odds_results.index.map(lambda x: VAR_TRANSLATIONS.get(x, x.replace('_', ' ').title()))
        log_odds_results = log_odds_results.sort_values(by="Coeficiente (Log-Odds)", ascending=False)

        st.dataframe(log_odds_results.style.format({'Coeficiente (Log-Odds)': '{:.4f}', 'P-valor': '{:.4f}'}), use_container_width=True)
    st.markdown("---")

    # Explicação da tabela e dos valores (Aprimorada com detalhes sobre o cálculo do Odds Ratio)
    st.subheader("Como interpretar a tabela de impacto (Odds Ratio)?")
    st.info("""
    Esta tabela exibe a **Força do Impacto** (Odds Ratio) de cada fator (variável) que você selecionou sobre a **chance de cancelamento** de uma reserva, mantendo os outros fatores constantes. Além disso, mostra a **Significância Estatística** do impacto.

    * **O que é o Odds Ratio e como é calculado?**
        O Odds Ratio é derivado dos coeficientes do modelo de Regressão Logística. Em um modelo logístico, os coeficientes $(\\beta)$ representam o impacto de cada variável na **log-odds** do evento (cancelamento). Para converter a log-odds de volta para uma medida de chance mais intuitiva (as odds), usamos a função exponencial:
        $$ \\text{Odds Ratio} = e^{\\beta} $$
        Como a função exponencial ($e^x$) é sempre positiva para qualquer número real $x$, o Odds Ratio será **sempre um número positivo ($> 0$)**.

    * **Interpretando o Odds Ratio:**
        * **Odds Ratio > 1 (Vermelho):** O fator **aumenta a chance de cancelamento**. Isso ocorre quando o coeficiente $(\\beta)$ da variável é positivo. Quanto maior o Odds Ratio, maior é o aumento. Por exemplo, um valor de **1.5** significa que a chance de cancelar é **50% maior** (1.5 - 1 = 0.5, ou 50%) para cada unidade de aumento no fator (para variáveis contínuas) ou quando a categoria do fator está presente (para variáveis binárias).
        * **Odds Ratio < 1 (Verde):** O fator **diminui a chance de cancelamento** (é um fator de proteção). Isso ocorre quando o coeficiente $(\\beta)$ da variável é negativo. Quanto mais próximo de 0 o Odds Ratio (ou seja, quanto mais negativo o coeficiente), maior a diminuição da chance. Por exemplo, um valor de **0.7** significa que a chance de cancelar é **30% menor** (1 - 0.7 = 0.3, ou 30%) para cada unidade de aumento no fator (ou quando a categoria está ativa).
        * **Odds Ratio = 1 (ou muito próximo):** O fator **não tem impacto estatisticamente relevante** na chance de cancelamento. Isso ocorre quando o coeficiente $(\\beta)$ da variável é próximo de zero. Sua presença ou variação não altera significativamente a probabilidade de cancelamento.

    * **P-valor:** Indica a probabilidade de que o impacto observado do fator seja apenas por acaso. Um P-valor baixo (tipicamente **< 0.05**) sugere que o impacto é estatisticamente **real** e não fruto da sorte, sendo confiável para tomada de decisão.
    * **Significância:** Baseado no P-valor. 'Significativo' indica que podemos confiar no impacto do fator para a previsão de cancelamento. 'Não Significativo' significa que o efeito observado pode ser apenas aleatório e não deve ser usado para basear ações.
    """)

    results = pd.DataFrame({
        "Força do Impacto (Odds Ratio)": np.exp(model.params),
        "P-valor": model.pvalues,
    }).drop('const', errors='ignore')

    results['Significância'] = np.where(results['P-valor'] < 0.05, 'Significativo', 'Não Significativo')
    results.index = results.index.map(lambda x: VAR_TRANSLATIONS.get(x, x))

    def style_impact_table(df):
        def color_cells(val):
            if isinstance(val, (int, float)):
                if val > 1:
                    intensity = min(1, (val - 1) / 4)
                    g_b_value = int(255 * (1 - intensity))
                    return f'background-color: rgb(255, {g_b_value}, {g_b_value}); color: black'
                elif val < 1:
                    intensity = min(1, (1 - val) / 1)
                    r_b_value = int(255 * (1 - intensity))
                    return f'background-color: rgb({r_b_value}, 255, {r_b_value}); color: black'
            return 'color: black'

        styled_df = df.style.map(color_cells, subset=["Força do Impacto (Odds Ratio)"])
        styled_df = styled_df.format({'Força do Impacto (Odds Ratio)': '{:.4f}', 'P-valor': '{:.4f}'})
        return styled_df

    st.subheader("Tabela de Impacto dos Fatores (Odds Ratio)")
    st.dataframe(style_impact_table(results.sort_values(by="Força do Impacto (Odds Ratio)", ascending=False)), use_container_width=True)

    st.markdown("---")
    st.subheader("Análise Detalhada por Fator")
    st.markdown("Clique em cada fator para entender seu impacto específico na chance de cancelamento:")

    sorted_results_for_analysis = results.sort_values(by="Força do Impacto (Odds Ratio)", ascending=False)

    with np.errstate(over='ignore'): # Suprimir warnings de overflow ao calcular np.exp em Odds Ratios extremos
        for index, row in sorted_results_for_analysis.iterrows():
            feature_name_translated = index # Este é o nome já traduzido (ex: 'País: Portugal')
            odds_ratio_raw = row["Força do Impacto (Odds Ratio)"]
            p_value = row["P-valor"]
            significance = row["Significância"]

            # Tratar Odds Ratio infinito para exibição e categorização
            if np.isinf(odds_ratio_raw):
                odds_ratio_display = "Indefinido (muito alto)"
                odds_ratio = 999999999 # Um número muito grande para cair na maior categoria de risco
            else:
                odds_ratio_display = f"{odds_ratio_raw:.3f}"
                odds_ratio = odds_ratio_raw

            with st.expander(f"**{feature_name_translated}** (Odds Ratio: {odds_ratio_display}, P-valor: {p_value:.3f})"):
                st.markdown(f"**Análise para '{feature_name_translated}':**")

                if significance == 'Não Significativo':
                    st.info(f"O impacto de '{feature_name_translated}' (Odds Ratio: {odds_ratio_display}) **não é estatisticamente significativo** (P-valor: {p_value:.3f} > 0.05). Isso significa que, com base nos dados, não podemos afirmar com confiança que este fator realmente influencia a chance de cancelamento; a variação observada pode ser apenas por acaso. Portanto, **não é recomendado basear decisões estratégicas importantes apenas neste fator.**")
                elif odds_ratio > 1:
                    percentage_increase = (odds_ratio - 1) * 100

                    if odds_ratio >= 100:
                        st.error(f"**RISCO EXTREMAMENTE CRÍTICO!** O Odds Ratio de **{odds_ratio_display}** é imensamente maior que 1. Isso indica que a presença ou aumento de '{feature_name_translated}' **aumenta a chance de cancelamento em mais de {percentage_increase:.0f}%**. Este é um fator de risco **massivo** e sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma que seu impacto é real. **Requer atenção imediata.**")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem, calculamos $({odds_ratio_display} - 1) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['deposit_type_Non Refund']:
                            st.markdown("Este é o **maior indicador de risco** na maioria dos modelos. Reservas com este tipo de depósito têm uma chance drasticamente maior de serem canceladas, muitas vezes por falta de pagamento ou comprometimento inicial. É crucial monitorar pagamentos associados a esta condição e considerar políticas de pagamento mais rigorosas ou alternativas para este segmento.")
                    elif odds_ratio >= 10:
                        st.warning(f"**RISCO MUITO ALTO!** O Odds Ratio de **{odds_ratio_display}** é substancialmente maior que 1. Isso indica que a presença ou aumento de '{feature_name_translated}' **aumenta a chance de cancelamento em aproximadamente {percentage_increase:.1f}%**. É um fator de risco **muito forte**, e sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma que seu impacto é real. **Recomenda-se vigilância.**")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem, calculamos $({odds_ratio_display} - 1) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['previous_cancellations']:
                            st.markdown("Um histórico de cancelamentos anteriores é um **fortíssimo preditor** de cancelamento futuro. Indica um padrão de comportamento do cliente que exige atenção imediata e, possivelmente, uma abordagem personalizada ou políticas de pré-pagamento mais rigorosas.")
                    elif odds_ratio >= 2:
                        st.warning(f"**RISCO ELEVADO!** O Odds Ratio de **{odds_ratio_display}** é significativamente maior que 1. Isso indica que a presença ou aumento de '{feature_name_translated}' **aumenta a chance de cancelamento em aproximadamente {percentage_increase:.1f}%**. É um fator de risco importante e sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma que seu impacto é real. **Monitore de perto.**")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem, calculamos $({odds_ratio_display} - 1) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['market_segment_Online TA']:
                            st.markdown("Canais como OTAs (Online Travel Agencies) frequentemente oferecem maior flexibilidade de cancelamento, o que contribui para o risco. É importante entender as políticas específicas de cada OTA, que podem ter prazos de cancelamento mais longos ou menos restritivos, e tentar converter a reserva para um canal direto oferecendo benefícios exclusivos.")
                        elif feature_name_translated == VAR_TRANSLATIONS['customer_type_Transient']:
                            st.markdown("Clientes avulsos (Transient), que não fazem parte de grupos ou contratos, podem ter menos lealdade ou compromisso estabelecido, aumentando ligeiramente a chance de cancelamento. Eles podem estar mais propensos a comparar ofertas e mudar de ideia até a data da estadia.")
                        elif feature_name_translated == VAR_TRANSLATIONS['hotel_City Hotel']:
                            st.markdown("Se este fator for significativo, indica que reservas em **Hotéis de Cidade** podem ter um risco de cancelamento maior em comparação com outros tipos de hotel (ex: Resort), devido a características de viagem ou público-alvo distintos, como viagens a negócios mais suscetíveis a alterações de agenda.")
                        elif feature_name_translated == VAR_TRANSLATIONS['distribution_channel_TA/TO']:
                             st.markdown("O canal de distribuição 'Agência de Viagem/Operadora' pode estar associado a um risco maior de cancelamento, possivelmente devido a políticas de cancelamento mais flexíveis ou a tipos de reservas específicos (ex: pacotes turísticos) intermediados por esses canais que têm maior probabilidade de serem alterados.")
                        elif feature_name_translated == VAR_TRANSLATIONS['meal_SC']:
                             st.markdown("O regime de refeição 'Sem Refeição' (SC) pode estar associado a um maior risco de cancelamento, talvez indicando um cliente que busca apenas hospedagem básica e tem menos 'laços' com a experiência completa do hotel ou menos comprometimento com a estadia planejada.")
                        elif feature_name_translated == VAR_TRANSLATIONS['total_guests']:
                             st.markdown(f"Um maior número de hóspedes ({feature_name_translated}) pode aumentar a complexidade da reserva e a probabilidade de cancelamento em aproximadamente {percentage_increase:.1f}%, talvez devido a alterações de planos de um dos membros do grupo.")
                        elif feature_name_translated == VAR_TRANSLATIONS['assigned_room_type_Changed']:
                             st.markdown(f"Se o tipo de quarto foi alterado após a reserva original, isso pode indicar um risco de cancelamento de {percentage_increase:.1f}%, talvez por insatisfação com a mudança ou incerteza no planejamento.")
                        elif feature_name_translated == VAR_TRANSLATIONS['is_agent_booking']:
                             st.markdown(f"Se for uma reserva feita por um agente e este fator for de risco, pode indicar que certos agentes têm maior taxa de cancelamento, talvez devido a volume alto ou características de suas reservas. Monitore a performance de agentes específicos.")
                    else: # Odds Ratio entre 1 e 2 (impacto menor, 0.1% a 99%)
                        st.info(f"**RISCO MODERADO/BAIXO!** O Odds Ratio de **{odds_ratio_display}** é ligeiramente maior que 1. Isso indica que a presença ou aumento de '{feature_name_translated}' **aumenta a chance de cancelamento em aproximadamente {percentage_increase:.1f}%**. É um fator de risco presente, mas com impacto mais discreto. Sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma que seu impacto é real.")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem, calculamos $({odds_ratio_display} - 1) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['lead_time']:
                            st.markdown(f"Para cada dia a mais de antecedência na reserva, a chance de cancelamento aumenta em aproximadamente {percentage_increase:.1f}%. Embora o impacto por dia seja pequeno, para reservas com **muita antecedência (centenas de dias)**, o efeito cumulativo pode ser substancial, tornando a reserva mais vulnerável a mudanças de plano ou a encontrar melhores ofertas. Monitore proativamente reservas com lead time elevado.")
                        elif feature_name_translated == VAR_TRANSLATIONS['adr']:
                            st.markdown(f"Para cada euro a mais no preço médio por noite, a chance de cancelamento aumenta em aproximadamente {percentage_increase:.1f}%*. Um ADR (Average Daily Rate) mais alto pode levar a uma maior reavaliação por parte do cliente, especialmente se o valor percebido não justificar o preço. O impacto é marginal por euro, mas pode somar em reservas caras, onde o cliente pode buscar alternativas mais em conta. (*Obs: Este é um impacto percentual por unidade. Um pequeno percentual pode significar um grande impacto em valores altos de ADR.)")
                        elif feature_name_translated == VAR_TRANSLATIONS['total_nights']:
                             st.markdown(f"Para cada noite adicional de estadia, a chance de cancelamento aumenta em aproximadamente {percentage_increase:.1f}%*. Em estadias muito longas, esse efeito pode ser mais perceptível e indica uma maior flexibilidade nos planos do cliente, que pode estar mais propenso a ajustar ou cancelar partes da estadia. (*Obs: Este é um impacto percentual por unidade. Um pequeno percentual pode significar um grande impacto em valores altos de Noites.)")
                        elif feature_name_translated == VAR_TRANSLATIONS['adults']:
                             st.markdown(f"Para cada adulto adicional, a chance de cancelamento aumenta em aproximadamente {percentage_increase:.1f}%*. Este fator, se significativo, pode indicar maior complexidade na reserva ou maior chance de alterações de planos para grupos maiores de adultos. (*Obs: Este é um impacto percentual por unidade de adulto, o impacto real aumenta com mais adultos.)")
                        elif feature_name_translated == VAR_TRANSLATIONS['is_weekend_stay']:
                             st.markdown(f"Estadias que incluem noites de fim de semana podem ter uma chance de cancelamento {percentage_increase:.1f}% maior, talvez indicando uma flexibilidade maior em planos de lazer.")
                        elif feature_name_translated == VAR_TRANSLATIONS['customer_type_Group']:
                             st.markdown(f"Reservas do tipo 'Grupo' podem apresentar uma chance de cancelamento {percentage_increase:.1f}% maior, possivelmente devido à complexidade da coordenação de múltiplos indivíduos.")


                else: # odds_ratio < 1
                    percentage_decrease = (1 - odds_ratio) * 100

                    if odds_ratio <= 0.1:
                        st.success(f"**FORTE FATOR DE PROTEÇÃO!** O Odds Ratio de **{odds_ratio_display}** é extremamente baixo. Isso indica que a presença ou aumento de '{feature_name_translated}' **diminui a chance de cancelamento em mais de {percentage_decrease:.0f}%**. Este é um fator protetor **excepcional**, e sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma seu impacto real. **Invista neste aspecto!**")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem de diminuição, calculamos $(1 - {odds_ratio_display}) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['is_repeated_guest']:
                            st.markdown("Clientes recorrentes demonstram **lealdade e confiança** no hotel, resultando em uma chance de cancelamento significativamente menor. Eles já conhecem e valorizam a experiência oferecida, tornando-os um segmento de baixo risco e alto valor. Invista pesado na fidelização desses clientes.")
                        elif feature_name_translated == VAR_TRANSLATIONS['total_of_special_requests']:
                            st.markdown("Clientes que fazem pedidos especiais demonstram um **maior engajamento e compromisso** com a estadia e a experiência no hotel, tornando-os menos propensos a cancelar. Isso indica um investimento emocional na reserva, pois o cliente já está personalizando sua experiência, o que reduz a probabilidade de desistência.")
                    elif odds_ratio <= 0.5:
                        st.success(f"**FATOR DE PROTEÇÃO SÓLIDO!** O Odds Ratio de **{odds_ratio_display}** é significativamente menor que 1. Isso indica que a presença ou aumento de '{feature_name_translated}' **diminui a chance de cancelamento em aproximadamente {percentage_decrease:.1f}%**. Este é um fator protetor **muito valioso**, e sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma que seu impacto é real.")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem de diminuição, calculamos $(1 - {odds_ratio_display}) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['booking_changes']:
                            st.markdown("A realização de alterações na reserva sugere que o cliente está **ajustando seus planos em vez de cancelar completamente**, indicando maior comprometimento e flexibilidade. Clientes que interagem para modificar a reserva são mais propensos a mantê-la e menos propensos a desistir totalmente, o que é um sinal positivo.")
                        elif feature_name_translated == VAR_TRANSLATIONS['previous_bookings_not_canceled']:
                             st.markdown("Um maior número de reservas anteriores que *não foram canceladas* demonstra um histórico de confiabilidade do cliente. Isso indica que ele tende a seguir com suas reservas e é um bom indicador de menor risco futuro. Clientes com esse histórico são mais previsíveis e menos propensos a cancelar.")
                        elif feature_name_translated == VAR_TRANSLATIONS['children_present']:
                             st.markdown("A presença de crianças ou bebês na reserva, se significativa, pode indicar um planejamento familiar mais robusto e menos propenso a cancelamentos de última hora, pois viagens em família geralmente envolvem mais coordenação e comprometimento prévio.")
                    else: # Odds Ratio entre 0.5 e 1 (proteção menor, 0.1% a 49%)
                        st.info(f"**Fator de Proteção MODERADO!** O Odds Ratio de **{odds_ratio_display}** é ligeiramente menor que 1. Isso indica que a presença ou aumento de '{feature_name_translated}' **diminui a chance de cancelamento em aproximadamente {percentage_decrease:.1f}%**. É um fator protetor, mas com impacto mais discreto. Sua significância estatística (P-valor: {p_value:.3f} < 0.05) confirma que seu impacto é real.")
                        st.markdown(f"**Como interpretar a variação percentual:** Um Odds Ratio de {odds_ratio_display} significa que a cada unidade de aumento neste fator (ou quando a categoria está presente), a chance de cancelamento é multiplicada por {odds_ratio_display}. Para expressar isso em porcentagem de diminuição, calculamos $(1 - {odds_ratio_display}) \\times 100\\%$.")
                        if feature_name_translated == VAR_TRANSLATIONS['required_car_parking_spaces']:
                            st.markdown("Solicitar uma vaga de garagem pode indicar que o cliente tem planos de viagem mais concretos (ex: viagem de carro), ou que ele valoriza comodidades específicas, tornando a reserva mais firme e menos sujeita a cancelamentos por indecisão. É um sinal de comprometimento com a viagem.")
                        elif feature_name_translated == VAR_TRANSLATIONS['children']:
                             st.markdown("A presença de crianças na reserva, se significativa, pode indicar um planejamento familiar mais robusto e menos propenso a cancelamentos de última hora, pois viagens em família geralmente envolvem mais coordenação e comprometimento prévio.")
                        elif feature_name_translated == VAR_TRANSLATIONS['babies']:
                             st.markdown("De forma similar às crianças, a presença de bebês pode estar associada a um planejamento mais cuidadoso e, portanto, a uma menor chance de cancelamento, devido à complexidade adicional de viajar com bebês que exige maior certeza nos planos e menor flexibilidade.")
                        elif feature_name_translated == VAR_TRANSLATIONS['country_grouped_PRT']:
                             st.markdown("A origem do cliente de Portugal (PRT), se significativa, pode ser um fator de proteção ou risco, dependendo da base de dados e do contexto do hotel. Geralmente, clientes locais ou de países próximos podem ter padrões de cancelamento diferentes, talvez com maior familiaridade com o destino ou menor burocracia para viagens.")
                        elif feature_name_translated == VAR_TRANSLATIONS['meal_BB']:
                             st.markdown("O regime de refeição 'Café da Manhã' (BB) pode ser um fator de proteção, indicando um cliente que busca uma experiência mais completa no hotel e está mais propenso a seguir com a reserva. Este cliente pode valorizar as comodidades do hotel além da simples hospedagem.")
                        elif feature_name_translated == VAR_TRANSLATIONS['reserved_room_type_A']:
                             st.markdown("Ter o tipo de quarto reservado como 'A' (ou outro tipo específico) pode ser um fator de proteção ou risco, dependendo da popularidade e características desse quarto. Isso pode indicar que o cliente encontrou exatamente o que procurava, solidificando a reserva.")
                        elif feature_name_translated == VAR_TRANSLATIONS['distribution_channel_Direct']:
                             st.markdown("Reservas feitas diretamente com o hotel (canal Direto) frequentemente apresentam menor risco de cancelamento, pois o cliente tem um contato mais direto e, muitas vezes, políticas de cancelamento mais claras ou benefícios diretos que incentivam a manutenção da reserva.")
                        elif feature_name_translated == VAR_TRANSLATIONS['is_company_booking']:
                             st.markdown("Se for uma reserva de empresa, e este fator for protetor, pode indicar maior estabilidade, pois reservas corporativas tendem a ser mais firmes devido a compromissos de negócios. É um sinal de responsabilidade.")
                        elif feature_name_translated == VAR_TRANSLATIONS['customer_type_Contract']:
                             st.markdown(f"Reservas do tipo 'Contrato' podem apresentar uma chance de cancelamento {percentage_decrease:.1f}% menor, indicando uma maior estabilidade e comprometimento, possivelmente devido a acordos de longo prazo.")
                        elif feature_name_translated == VAR_TRANSLATIONS['market_segment_Direct']:
                             st.markdown(f"O segmento de mercado 'Direto' pode ter uma chance de cancelamento {percentage_decrease:.1f}% menor, pois esses clientes muitas vezes têm um relacionamento mais direto com o hotel ou maior convicção em sua escolha.")


    st.markdown("---")
    st.subheader("Visualização dos Principais Fatores de Impacto")
    st.markdown("O gráfico abaixo ilustra a magnitude e direção do impacto (se aumenta ou diminui o risco) dos fatores mais significativos, ranqueados por sua força.")

    significant_results = results[results['Significância'] == 'Significativo'].copy()
    if not significant_results.empty:
        significant_results['Impacto Percentual'] = (significant_results['Força do Impacto (Odds Ratio)'] - 1) * 100
        significant_results.loc[significant_results['Impacto Percentual'] < 0, 'Impacto Percentual'] = \
            (1 - significant_results.loc[significant_results['Impacto Percentual'] < 0, 'Força do Impacto (Odds Ratio)']) * 100 * (-1)

        significant_results = significant_results.sort_values(by="Impacto Percentual", ascending=True)

        # Gráfico 1: Todos os fatores significativos
        fig_impact_all, ax_impact_all = plt.subplots(figsize=(10, max(6, len(significant_results) * 0.5)))
        colors_all = ['red' if x > 0 else 'green' for x in significant_results['Impacto Percentual']]
        ax_impact_all.barh(significant_results.index, significant_results['Impacto Percentual'], color=colors_all)
        ax_impact_all.set_xlabel("Impacto na Chance de Cancelamento (%)")
        ax_impact_all.set_ylabel("Fator de Risco/Proteção")
        ax_impact_all.set_title("Impacto Percentual de TODOS os Fatores Significativos")
        ax_impact_all.axvline(0, color='gray', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig_impact_all, use_container_width=True)
        plt.close(fig_impact_all)
        st.caption("Barras vermelhas indicam fatores que aumentam o risco de cancelamento. Barras verdes indicam fatores que diminuem o risco. Este gráfico apresenta a visão geral de todos os fatores com impacto estatisticamente significativo.")

    else:
        st.info("Não há fatores estatisticamente significativos para exibir nos gráficos com as variáveis selecionadas. Por favor, selecione variáveis que tenham um impacto demonstrável no cancelamento.")
# ==============================================================================
# ABA 3: SIMULADOR DE CENÁRIOS
# ==============================================================================
with tab3:
    st.header("⚙️ Simulador de Cenários de Reserva")
    st.markdown("Experimente diferentes combinações de características de reserva para entender o risco de cancelamento em tempo real. Isso permite visualizar o impacto das suas decisões e planejar ações proativas.")

    st.subheader("Configure o Cenário Hipotético")
    st.info("""
    Ajuste os controles abaixo para simular as condições de uma nova reserva e veja o risco de cancelamento estimado pelo modelo.
    * **Variáveis numéricas:** Digite o valor exato no campo numérico.
    * **Variáveis categóricas:** Selecione a opção desejada em um menu dropdown.
    """)

    sim_data = {}

    numerical_features_all = [f for f in data.columns if data[f].nunique() > 2 and f != 'is_canceled']
    binary_s_features_base = ['is_agent_booking', 'is_company_booking', 'is_weekend_stay', 'children_present', 'assigned_room_type_Changed']

    numerical_features_selected = [f for f in final_features_for_model_training if f in numerical_features_all]
    binary_s_features_selected = [f for f in final_features_for_model_training if f in binary_s_features_base]

    original_categorical_features_selected = []
    for original_cat_feature in CATEGORICAL_COLS_MAP.keys():
        if (original_cat_feature in VAR_TRANSLATIONS and
            any(f"{original_cat_feature}_{cat.replace(' ', '_').replace('/', '_').replace('-', '_')}" in final_features_for_model_training
                for cat in CATEGORICAL_COLS_MAP[original_cat_feature].keys())):
            original_categorical_features_selected.append(original_cat_feature)

    numerical_features_selected.sort()
    original_categorical_features_selected.sort()
    binary_s_features_selected.sort()


    st.markdown("#### Características Numéricas da Reserva")
    num_cols_num = 3
    cols_num = st.columns(num_cols_num)

    for i, feature in enumerate(numerical_features_selected):
        label = VAR_TRANSLATIONS.get(feature, feature.replace('_', ' ').title())
        col_widget = cols_num[i % num_cols_num]

        default_value = float(X_train_smote_mean.get(feature, 0))

        is_integer_type = (data[feature].dtype in ['int64', 'int32']) or (data[feature].dropna().apply(lambda x: x == int(x)).all())

        if is_integer_type:
            value_input = col_widget.number_input(
                f"{label} (Valor)",
                min_value=0,
                value=int(default_value),
                step=1,
                key=f"num_input_{feature}"
            )
            sim_data[feature] = value_input
        else:
            value_input = col_widget.number_input(
                f"{label} (Valor)",
                min_value=0.0,
                value=float(default_value),
                step=0.01,
                format="%.2f",
                key=f"num_input_{feature}"
            )
            sim_data[feature] = value_input


    st.markdown("#### Características Categóricas da Reserva")
    num_cols_cat = 3
    cols_cat = st.columns(num_cols_cat)

    for i, original_cat_feature in enumerate(original_categorical_features_selected):
        label = VAR_TRANSLATIONS.get(original_cat_feature, original_cat_feature.replace('_', ' ').title())
        col_widget = cols_cat[i % num_cols_cat]

        options_original_keys = list(CATEGORICAL_COLS_MAP[original_cat_feature].keys())

        default_cat_value_original_key = None
        dummy_counts = {}
        for option_orig_key in options_original_keys:
            dummy_col_name = f"{original_cat_feature}_{option_orig_key.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            if dummy_col_name in X_train_smote_mean:
                dummy_counts[option_orig_key] = X_train_smote_mean[dummy_col_name]

        if dummy_counts:
            default_cat_value_original_key = max(dummy_counts, key=dummy_counts.get)
        else:
            default_cat_value_original_key = options_original_keys[0]

        selected_option_original_key = col_widget.selectbox(
            label,
            options_original_keys, # Passa as opções originais como valor interno
            index=options_original_keys.index(default_cat_value_original_key),
            format_func=lambda x: CATEGORICAL_COLS_MAP[original_cat_feature].get(x, x), # Exibe a tradução da opção
            key=f"sb_{original_cat_feature}"
        )
        sim_data[original_cat_feature] = selected_option_original_key

        # O mapeamento para dummies é feito na seção de Previsão, para garantir consistência.

    if binary_s_features_selected:
        st.markdown("#### Características Binárias Adicionais")
        for i, feature in enumerate(binary_s_features_selected):
            label = VAR_TRANSLATIONS.get(feature, feature.replace('_', ' ').title())
            col_widget = cols_cat[i % num_cols_cat]

            default_value = int(X_train_smote_mean.get(feature, 0))
            default_index = 0 if default_value == 0 else 1

            sim_data[feature] = col_widget.selectbox(
                label,
                [0, 1],
                index=default_index,
                format_func=lambda x: "Sim" if x == 1 else "Não",
                key=f"sb_binary_{feature}"
            )


    # --- Previsão (Bloco Robusto) ---
    # Cria um DataFrame de uma linha com todos os dados da simulação
    final_sim_data_for_prediction = {}
    for feature in final_features_for_model_training: # Itera sobre as features que o MODELO realmente usa
        if feature in numerical_features_all or feature in binary_s_features_base:
            final_sim_data_for_prediction[feature] = sim_data.get(feature, 0)
        else: # Se a feature é uma dummy gerada de uma categoria original
            original_cat_name = None
            for cat_key in CATEGORICAL_COLS_MAP.keys():
                if feature.startswith(f"{cat_key}_"):
                    original_cat_name = cat_key
                    break

            if original_cat_name and original_cat_name in sim_data:
                selected_cat_value_original = sim_data[original_cat_name]
                dummy_name_from_selected = f"{original_cat_name}_{selected_cat_value_original.replace(' ', '_').replace('/', '_').replace('-', '_')}"

                final_sim_data_for_prediction[feature] = 1 if feature == dummy_name_from_selected else 0
            else:
                final_sim_data_for_prediction[feature] = 0

    # Passo 2: Preparar o DataFrame para o modelo
    # Cria um DataFrame de uma linha com todos os dados da simulação
    sim_df = pd.DataFrame([final_sim_data_for_prediction])

    # Garante que todas as colunas que o modelo espera existam, preenchendo com 0 se faltarem
    for col in model.params.index:st.sidebar.header("🔧 1. Construção do Modelo Preditivo")
        if col != 'const' and col not in sim_df.columns:
            sim_df[col] = 0

    # Adiciona a constante para o cálculo do modelo
    sim_df_const = sm.add_constant(sim_df, has_constant='add')

    # Reordena as colunas para bater exatamente com a ordem que o modelo foi treinado
    sim_df_final = sim_df_const[model.params.index]

    # Converte tudo para número, forçando erros a se tornarem 'NaN' (Not a Number)
    sim_df_final = sim_df_final.apply(pd.to_numeric, errors='coerce')

    # Substitui qualquer 'NaN' que possa ter surgido por 0, para segurança máxima
    sim_df_final.fillna(0, inplace=True)

    # Passo 3: Realizar a predição
    with np.errstate(over='ignore'):
        try:
            # Envia os dados limpos e garantidos para a predição
            sim_proba = model.predict(sim_df_final)[0]
        except Exception as e:
            st.error(f"Erro ao realizar a predição: {e}. Verifique as variáveis de entrada no simulador.")
            sim_proba = 0.5

    st.session_state.sim_data = sim_data
    st.session_state.sim_proba = sim_proba

    st.markdown("---")
    st.subheader("Diagnóstico do Risco para o Cenário Simulado")
    gauge_col, _ = st.columns(2)
    with gauge_col:
        st.metric("Risco de Cancelamento:", f"{sim_proba:.1%}")
        st.progress(sim_proba)

        if sim_proba > 0.7:
            st.error("RISCO CRÍTICO! 🚨 Ação preventiva é altamente recomendada para evitar a perda desta reserva.")
            st.markdown("Este cenário indica uma alta probabilidade de cancelamento. É fundamental agir rapidamente e de forma direcionada.")
        elif sim_proba > 0.4:
            st.warning("RISCO MODERADO. ⚠️ Monitore esta reserva de perto e considere ações proativas.")
            st.markdown("A chance de cancelamento é significativa. Medidas preventivas podem reduzir o risco de forma eficaz.")
        else:
            st.success("RISCO BAIXO. ✅ Ótima reserva! Foque em oferecer uma excelente experiência para fidelizar o cliente.")
            st.markdown("Este cenário apresenta baixa probabilidade de cancelamento. Concentre-se em garantir a satisfação e fidelização do cliente.")
    st.info("Acesse a aba *'Plano de Ação Personalizado'* para ver recomendações específicas e acionáveis para este cenário.")


# ==============================================================================
# ABA 4: PLANO DE AÇÃO PERSONALIZADO
# ==============================================================================
with tab4:
    st.header("💡 Plano de Ação Personalizado")
    st.markdown("Com base no cenário simulado e no risco de cancelamento calculado, esta seção oferece um plano de ação estratégico para mitigar o risco e otimizar a gestão de reservas.")

    if 'sim_proba' not in st.session_state:
        st.warning("Por favor, configure um cenário na aba 'Simulador de Cenários' primeiro para gerar um plano de ação.")
    else:
        sim_data = st.session_state.sim_data
        sim_proba = st.session_state.sim_proba

        st.subheader(f"Análise do Cenário: Risco de Cancelamento de {sim_proba:.1%}")

        if sim_proba < 0.2:
             st.success("✅ **Cenário de Baixo Risco! Parabéns!**")
             st.markdown("""
             As características desta reserva indicam uma alta probabilidade de não cancelamento. O foco aqui deve ser na **fidelização e otimização da receita**.

             **Estratégias de Ação:**
             * **Fidelização Proativa:** Considere enviar um e-mail de boas-vindas personalizado, oferecendo um pequeno mimo no check-in (como um upgrade de quarto se disponível, voucher para um drink no bar, ou um late check-out cortesia). Isso cria uma experiência memorável e incentiva futuras reservas diretas.
             * **Upselling/Cross-selling:** Com baixo risco de cancelamento, você tem a oportunidade de oferecer serviços adicionais antes ou durante a estadia (ex: pacotes de spa, jantar especial, passeios turísticos) para aumentar a receita por hóspede.
             * **Pesquisa de Satisfação:** Após a estadia, envie uma pesquisa de satisfação para coletar feedback valioso e identificar promotores da sua marca, incentivando reviews positivos online.
             """)
        else:
            st.markdown("Com base no cenário simulado e na chance de cancelamento, as seguintes ações preventivas são **altamente recomendadas**:")
            recomendacoes_mostradas = 0

            # Identificar fatores de risco ativos no cenário simulado
            active_risks_in_scenario_map = {} # {feature_model_name: odds_ratio}
            for feature_model_name in model.params.index: # Itera sobre as features que o MODELO realmente usou
                if feature_model_name == 'const': continue

                coef_log_odds = model.params[feature_model_name]
                with np.errstate(over='ignore'):
                    odds_ratio = np.exp(coef_log_odds)

                if odds_ratio > 1.05: # Se é um fator de risco (aumenta a chance em pelo menos 5%)
                    is_active_in_sim_data = False

                    # 1. Variáveis numéricas (diretamente do sim_data)
                    if feature_model_name in sim_data and sim_data[feature_model_name] is not None and feature_model_name in data.columns and data[feature_model_name].nunique() > 2:
                        if sim_data[feature_model_name] > data[feature_model_name].quantile(0.75): # Valor alto para numéricas
                            is_active_in_sim_data = True
                    # 2. Variáveis binárias sintéticas (diretamente do sim_data)
                    elif feature_model_name in ['is_agent_booking', 'is_company_booking', 'is_weekend_stay', 'children_present', 'assigned_room_type_Changed']:
                        if sim_data.get(feature_model_name, 0) == 1:
                            is_active_in_sim_data = True
                    # 3. Dummies de variáveis categóricas originais
                    else:
                        original_cat_name = None
                        for cat_key in CATEGORICAL_COLS_MAP.keys():
                            if feature_model_name.startswith(f"{cat_key}_"):
                                original_cat_name = cat_key
                                break

                        if original_cat_name and original_cat_name in sim_data: # Se a categoria original foi selecionada no simulador
                            selected_category_original_value = sim_data[original_cat_name] # Valor original (ex: 'City Hotel')
                            # Construir o nome da dummy exatamente como o modelo espera
                            dummy_name_from_selected = f"{original_cat_name}_{selected_category_original_value.replace(' ', '_').replace('/', '_').replace('-', '_')}"
                            if feature_model_name == dummy_name_from_selected:
                                is_active_in_sim_data = True

                    if is_active_in_sim_data:
                        active_risks_in_scenario_map[feature_model_name] = odds_ratio

            sorted_active_risks_in_scenario = sorted(active_risks_in_scenario_map.items(), key=lambda item: item[1], reverse=True)

            # 2. Gerar recomendações com base nos fatores de risco ativos (priorizados)
            for feature_model_name, odds_ratio in sorted_active_risks_in_scenario:
                # O nome traduzido para display no plano de ação
                feature_display_name = VAR_TRANSLATIONS.get(feature_model_name, feature_model_name.replace('_', ' ').title())

                if feature_model_name == 'deposit_type_Non Refund':
                    st.error(f"🚨 **Ponto Crítico: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("""
                    Esta política de depósito está **fortemente associada a cancelamentos**. Se o cliente escolheu esta opção e não efetuou o pagamento, a chance de cancelamento é extremamente alta.
                    * **Ação Imediata:** Confirme o pagamento ativamente com o cliente. Se o pagamento não for feito dentro do prazo estipulado, proceda com o cancelamento o mais rápido possível para liberar o quarto e minimizar perdas.
                    * **Estratégia de Longo Prazo:** Avalie a viabilidade de flexibilizar as políticas de depósito não reembolsável ou oferecer opções alternativas (ex: pequeno sinal, pré-autorização de cartão) para reservas de alto risco, equilibrando segurança da receita com a redução do atrito para o cliente.
                    """)
                    recomendacoes_mostradas += 1

                elif feature_model_name == 'lead_time':
                    st.warning(f"⏳ **Atenção: {feature_display_name} ({sim_data.get('lead_time', 0)} dias) (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("""
                    Reservas feitas com grande antecedência apresentam maior incerteza nos planos do cliente, aumentando o risco de cancelamento.
                    * **Ação Proativa:** Implemente uma "jornada de engajamento" pré-estadia. Envie e-mails periódicos com conteúdo de valor (dicas sobre a cidade, atrações próximas, serviços do hotel, previsão do tempo para a data da estadia) para manter o cliente conectado e animado com a reserva.
                    * **Incentivo à Confirmação:** Considere oferecer um pequeno benefício (ex: 10% de desconto em um serviço do hotel, voucher para o restaurante, upgrade de amenidades) para clientes que confirmarem sua intenção de manter a reserva 30 ou 60 dias antes do check-in.
                    """)
                    recomendacoes_mostradas += 1

                elif feature_model_name == 'market_segment_Online TA':
                    st.warning(f"🌐 **Risco do Canal: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("""
                    Reservas originadas via OTAs (Booking, Expedia, etc.) frequentemente possuem políticas de cancelamento mais flexíveis, contribuindo para o risco.
                    * **Ação Estratégica:** Tente "converter" o cliente para uma reserva direta. Ofereça um pequeno incentivo exclusivo (ex: late check-out gratuito, um drink de boas-vindas na chegada, upgrade de quarto sujeito a disponibilidade) se ele confirmar a reserva ou considerar futuras reservas diretas.
                    * **Comunicação Direta:** Obtenha o contato do cliente via OTA e inicie uma comunicação direta para criar um relacionamento e mostrar os benefícios de reservar diretamente no futuro.
                    """)
                    recomendacoes_mostradas += 1

                elif feature_model_name == 'previous_cancellations':
                    st.error(f"🛑 **Histórico: {feature_display_name} ({sim_data.get('previous_cancellations', 0)}) (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("""
                    Um histórico de cancelamentos anteriores é um forte indicador de risco. Este cliente já demonstrou um padrão de comportamento de cancelamento.
                    * **Ação Personalizada:** Entre em contato proativamente e personalize a comunicação. Tente entender os motivos dos cancelamentos anteriores e ofereça soluções ou garantias que abordem essas preocupações. Para reservas de alto valor, um contato direto e atencioso pode ser decisivo na retenção.
                    * **Política de Cancelamento:** Para clientes com múltiplos cancelamentos, considere aplicar políticas de pagamento mais rigorosas ou exigir um pré-pagamento para futuras reservas.
                    """)
                    recomendacoes_mostradas += 1

                elif feature_model_name == 'adr':
                    st.warning(f"💸 **Atenção: {feature_display_name} ({sim_data.get('adr', 0):.2f}€) (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("""
                    Um ADR (Average Daily Rate) elevado pode levar a mais comparações e reconsiderações por parte do cliente, aumentando o risco de cancelamento se o valor percebido não for alto.
                    * **Ação de Valor:** Garanta que o cliente compreenda o valor e os benefícios *exclusivos* incluídos na tarifa. Destaque serviços adicionais, comodidades premium, experiências personalizadas ou a qualidade superior que justificam o preço.
                    * **Experiência de Luxo:** Desde o primeiro contato, ofereça um atendimento de excelência para justificar o investimento do cliente e superar suas expectativas, solidificando a decisão da reserva.
                    """)
                    recomendacoes_mostradas += 1

                elif feature_model_name == 'adults' and sim_data.get('adults', 0) == 0 and (sim_data.get('children', 0) > 0 or sim_data.get('babies', 0) > 0):
                    st.warning("👤 **Alerta: Número de Adultos Zero (mas com Crianças/Bebês)**")
                    st.markdown("""
                    Uma reserva com zero adultos, mas indicando a presença de crianças ou bebês, pode ser um erro de preenchimento ou uma inconsistência séria nos dados da reserva.
                    * **Ação Corretiva:** Entre em contato imediato com o cliente para validar e corrigir as informações da reserva. Isso evita problemas no check-in, frustração do cliente e possíveis cancelamentos de última hora devido a dados incorretos ou falta de clareza.
                    """)
                    recomendacoes_mostradas += 1

                elif feature_model_name == 'total_nights' and sim_data.get('total_nights', 0) <= 1:
                    if model.params.get('total_nights', 0) < 0:
                        st.warning("🌙 **Atenção: Estadia Muito Curta (1 Noite)**")
                        st.markdown("""
                        Estadias de apenas uma noite tendem a ter maior probabilidade de cancelamento, talvez por serem mais flexíveis ou sujeitas a mudanças de planos de última hora.
                        * **Ação de Eficiência:** Para estas reservas, foque em agilizar o processo de check-in e check-out. Forneça informações essenciais de forma clara e rápida (ex: wi-fi, horários de refeição, contatos de emergência) para garantir uma experiência suave e sem atritos.
                        * **Pequenos Encantamentos:** Considere oferecer um pequeno benefício ou amenidade (ex: garrafa de água no quarto, desconto no minibar) para tornar a curta estadia mais memorável e desincentivar o cancelamento de última hora.
                        """)
                        recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['hotel_City Hotel']:
                    st.warning(f"🏨 **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("Se este fator for significativo, indica que reservas em **Hotéis de Cidade** podem ter um risco de cancelamento maior em comparação com outros tipos de hotel (ex: Resort), devido a características de viagem ou público-alvo distintos, como viagens a negócios mais suscetíveis a alterações de agenda.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['distribution_channel_TA/TO']:
                   st.warning(f"🤝 **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                   st.markdown("O canal de distribuição 'Agência de Viagem/Operadora' pode estar associado a um risco maior de cancelamento, possivelmente devido a políticas de cancelamento mais flexíveis ou a tipos de reservas específicos (ex: pacotes turísticos) intermediados por esses canais que têm maior probabilidade de serem alterados.")
                   recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['meal_SC']:
                   st.warning(f"🍽️ **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                   st.markdown("O regime de refeição 'Sem Refeição' (SC) pode estar associado a um maior risco de cancelamento, talvez indicando um cliente que busca apenas hospedagem básica e tem menos 'laços' com a experiência completa do hotel ou menos comprometimento com a estadia planejada.")
                   recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['total_guests']:
                    st.warning(f"👪 **Risco Potencial: {feature_display_name} ({sim_data.get('total_guests', 0)}) (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("Um maior número de hóspedes (adultos + crianças + bebês) pode aumentar a complexidade da reserva e a probabilidade de cancelamento. Coordenar planos para mais pessoas é mais difícil, tornando a reserva mais suscetível a mudanças ou desistências de última hora.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['assigned_room_type_Changed']:
                    st.warning(f"🔄 **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("Se o tipo de quarto atribuído ao cliente foi alterado em relação ao que foi reservado, isso pode indicar um risco de cancelamento. A alteração pode gerar insatisfação, confusão ou incerteza no cliente, levando-o a reconsiderar a reserva.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['is_agent_booking']:
                    st.warning(f"🤝 **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("Se as reservas feitas por um agente são um fator de risco, isso pode indicar que certos agentes têm maior taxa de cancelamento, talvez devido a volume alto ou características de suas reservas. Monitore a performance de agentes específicos.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['is_weekend_stay']:
                    st.warning(f"🗓️ **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("Estadias que incluem noites de fim de semana podem ter um risco de cancelamento maior. Viagens de lazer podem ser mais flexíveis e suscetíveis a mudanças de planos de última hora em comparação com viagens de negócios, por exemplo.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated == VAR_TRANSLATIONS['customer_type_Group']:
                    st.warning(f"👥 **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown(f"Reservas do tipo 'Grupo' podem apresentar uma chance de cancelamento {percentage_increase:.1f}% maior, possivelmente devido à complexidade da coordenação de múltiplos indivíduos.")

                elif feature_name_translated == VAR_TRANSLATIONS['market_segment_Undefined']:
                    st.warning(f"❓ **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown("Um segmento de mercado indefinido pode indicar problemas na origem da reserva ou baixa rastreabilidade, o que pode estar associado a um risco maior de cancelamento por falta de informações claras.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated.startswith('País:'):
                    st.warning(f"🌍 **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                    st.markdown(f"Clientes de {feature_name_translated.replace('País: ', '')}, se significativa, podem ter um risco maior de cancelamento. Isso pode ser devido a padrões de viagem locais ou políticas de reserva comuns na região.")
                    recomendacoes_mostradas += 1

                elif feature_name_translated in VAR_TRANSLATIONS: # Catch-all para outras dummies de risco
                   st.warning(f"⚠️ **Risco Potencial: {feature_display_name} (Odds Ratio: {odds_ratio:.2f})**")
                   st.markdown(f"Este fator está associado a um risco aumentado de cancelamento. Ações preventivas específicas para {feature_display_name.lower()} devem ser consideradas.")
                   recomendacoes_mostradas += 1

            # 3. Gerar recomendações para fatores de PROTEÇÃO que *não estão presentes* ou estão em um nível desfavorável
            # Nota: para variáveis categóricas originais, estamos verificando a AUSÊNCIA da categoria protetora
            # Ou a presença de uma categoria que não é a mais protetora.

            # Checar a ausência de Pedidos Especiais (total_of_special_requests)
            if 'total_of_special_requests' in model.params and model.params['total_of_special_requests'] < 0: # Se é um fator de proteção
                if sim_data.get('total_of_special_requests', 0) == 0:
                    st.info("ℹ️ **Oportunidade: Cliente Pouco Engajado (Sem Pedidos Especiais)**")
                    st.markdown("""
                    A ausência de pedidos especiais pode indicar um baixo nível de engajamento do cliente com a reserva. Aumentar o engajamento pode reduzir a incerteza e o risco de cancelamento.
                    * **Ação de Engajamento:** Envie um e-mail de "Pré-Check-in" proativo, perguntando se o hotel pode ajudar com algo para a estadia (ex: transfer, reservas em restaurantes, amenidades especiais no quarto, dicas de passeios ou eventos locais). Isso cria um canal de comunicação, aumenta o engajamento e pode transformar a reserva em um compromisso mais firme.
                    """)
                    recomendacoes_mostradas += 1

            # Checar a ausência de Alterações na Reserva (booking_changes)
            if 'booking_changes' in model.params and model.params['booking_changes'] < 0:
                if sim_data.get('booking_changes', 0) == 0:
                    st.info("ℹ️ **Oportunidade: Reserva Sem Alterações (Indica Potencial Inatividade)**")
                    st.markdown("""
                    Reservas sem alterações podem, paradoxalmente, ter um risco ligeiramente maior se o cliente não está ativamente engajado. A capacidade de alteração demonstra compromisso com a reserva, ajustando-a aos planos.
                    * **Ação de Flexibilidade:** Comunique proactivamente a flexibilidade das políticas de alteração do hotel. Isso pode incentivar o cliente a fazer pequenos ajustes em vez de cancelar completamente, mantendo a reserva ativa.
                    """)
                    recomendacoes_mostradas += 1

            if 'is_repeated_guest' in sim_data and sim_data['is_repeated_guest'] == 0:
               if 'is_repeated_guest' in model.params and model.params['is_repeated_guest'] < 0:
                    st.info("🆕 **Oportunidade: Cliente Novo (Potencial para Fidelização)**")
                    st.markdown("""
                    Este é um cliente novo. Embora não seja um fator de risco *direto* em si, a ausência do fator 'cliente recorrente' (que é protetor) significa que há uma oportunidade de construir lealdade.
                    * **Ação de Fidelização:** Ofereça um programa de fidelidade no check-in, um pequeno presente de boas-vindas, ou um cupom de desconto para a próxima estadia. Concentre-se em garantir uma primeira experiência excepcional para incentivar a recorrência.
                    """)
                    recomendacoes_mostradas += 1

            if 'required_car_parking_spaces' in sim_data and sim_data['required_car_parking_spaces'] == 0:
                if 'required_car_parking_spaces' in model.params and model.params['required_car_parking_spaces'] < 0:
                    st.info("🅿️ **Oportunidade: Não Há Solicitação de Vaga de Garagem**")
                    st.markdown("""
                    A ausência de solicitação de vaga de garagem pode significar que o cliente não viajará de carro ou tem planos menos firmes para a viagem. Se este for um fator protetor, a sua ausência representa uma oportunidade.
                    * **Ação de Engajamento:** Verifique se o cliente precisará de alguma forma de transporte ou orientação sobre como chegar ao hotel. Ofereça informações sobre estacionamentos próximos ou transporte público para aumentar a conveniência e o comprometimento.
                    """)
                    recomendacoes_mostradas += 1

            # Checar a ausência de Café da Manhã (meal_BB) se for fator protetor
            if 'meal_BB' in model.params and model.params['meal_BB'] < 0:
                if sim_data.get('meal') != 'BB':
                    st.info("☕ **Oportunidade: Regime de Refeição Sem Café da Manhã (BB)**")
                    st.markdown("""
                    Se o regime de Café da Manhã (BB) é um fator protetor e não foi selecionado, pode indicar uma oportunidade de aumentar o engajamento.
                    * **Ação de Upselling/Engajamento:** Ofereça o café da manhã como um adicional com um pequeno desconto, ou destaque a qualidade do café da manhã do hotel em um e-mail de pré-estadia para incentivar a adição e o compromisso.
                    """)
                    recomendacoes_mostradas += 1

            # Checar se Quarto A não foi reservado (reserved_room_type_A) se for fator protetor
            if 'reserved_room_type_A' in model.params and model.params['reserved_room_type_A'] < 0:
                if sim_data.get('reserved_room_type') != 'A':
                    st.info("🛏️ **Oportunidade: Outro Tipo de Quarto Reservado (Não Tipo A)**")
                    st.markdown("""
                    Se o Tipo de Quarto 'A' for um fator de proteção (se o cliente tiver reservado este tipo, a chance de cancelar é menor) e este tipo não foi reservado no cenário, pode haver uma oportunidade.
                    * **Ação de Upgrade/Engajamento:** Considere oferecer um upgrade para este tipo de quarto (se disponível e apropriado) ou destaque as vantagens do quarto reservado para aumentar a satisfação e reduzir a chance de cancelamento.
                    """)
                    recomendacoes_mostradas += 1

            if 'is_company_booking' in sim_data and sim_data['is_company_booking'] == 0:
                if 'is_company_booking' in model.params and model.params['is_company_booking'] < 0:
                    st.info("🏢 **Oportunidade: Reserva Não Corporativa**")
                    st.markdown("""
                    Se reservas corporativas são um fator de proteção (menor risco de cancelamento), a ausência desse fator significa uma oportunidade para o hotel.
                    * **Ação de Segmentação:** Identifique potenciais clientes corporativos e ofereça pacotes ou benefícios específicos para empresas para incentivar esse tipo de reserva mais estável.
                    """)
                    recomendacoes_mostradas += 1

            if 'children_present' in sim_data and sim_data['children_present'] == 0:
                if 'children_present' in model.params and model.params['children_present'] < 0:
                    st.info("👨‍🦰 **Oportunidade: Ausência de Crianças/Bebês**")
                    st.markdown("""
                    Se a presença de crianças ou bebês é um fator protetor (indica maior planejamento), a ausência deles pode ser uma oportunidade para reforçar o comprometimento.
                    * **Ação de Engajamento:** Para reservas sem crianças, foque em aspectos como flexibilidade, opções de lazer para adultos, ou conveniência, para solidificar a reserva e reduzir incertezas.
                    """)
                    recomendacoes_mostradas += 1

            if 'total_guests' in sim_data and sim_data['total_guests'] < 2:
                if 'total_guests' in model.params and model.params['total_guests'] < 0:
                    st.info("🧍 **Oportunidade: Reserva para Hóspede Único**")
                    st.markdown("""
                    Reservas para um único hóspede, se este for um fator de menor proteção (ou maior risco) em comparação com múltiplos hóspedes, podem ser uma oportunidade.
                    * **Ação de Experiência Individual:** Personalize a comunicação para hóspedes únicos, destacando conveniências e serviços que tornam a estadia confortável para uma pessoa.
                    """)
                    recomendacoes_mostradas += 1

            if 'assigned_room_type_Changed' in sim_data and sim_data['assigned_room_type_Changed'] == 0:
                if 'assigned_room_type_Changed' in model.params and model.params['assigned_room_type_Changed'] > 0:
                    st.info("✅ **Ponto Positivo: Tipo de Quarto Atribuído NÃO Alterado**")
                    st.markdown("""
                    O fato de o tipo de quarto atribuído não ter sido alterado é um bom sinal, pois mudanças podem gerar insatisfação. Este é um ponto positivo que contribui para a estabilidade da reserva.
                    """)
                    recomendacoes_mostradas += 1

            if 'is_weekend_stay' in sim_data and sim_data['is_weekend_stay'] == 0:
                if 'is_weekend_stay' in model.params and model.params['is_weekend_stay'] > 0:
                    st.info("🗓️ **Ponto Positivo: Estadia NÃO Inclui Fim de Semana**")
                    st.markdown("""
                    Se estadias de fim de semana são mais arriscadas, o fato de esta reserva não incluir o fim de semana é um ponto positivo que contribui para a sua estabilidade.
                    """)
                    recomendacoes_mostradas += 1

            # Checar se o País 'BRA' não foi selecionado E se country_grouped_BRA é um fator de risco
            if 'country_grouped_BRA' in model.params and model.params['country_grouped_BRA'] > 0:
                if sim_data.get('country_grouped') != 'BRA':
                    st.info("🌍 **Ponto Positivo: Cliente NÃO É do Brasil (se Brasil for fator de risco)**")
                    st.markdown("""
                    Se clientes do Brasil (BRA) têm um risco maior de cancelamento (dada a base de dados do hotel), o fato de o cliente não ser do Brasil é um ponto positivo que contribui para a estabilidade da reserva.
                    """)
                    recomendacoes_mostradas += 1

            if 'customer_type_Contract' in sim_data and sim_data['customer_type_Contract'] == 0:
                if 'customer_type_Contract' in model.params and model.params['customer_type_Contract'] < 0:
                    st.info("🤝 **Oportunidade: Cliente NÃO É de Contrato**")
                    st.markdown("""
                    Se clientes de 'Contrato' são um fator protetor, a ausência de um contrato pode ser uma oportunidade para o hotel.
                    * **Ação de Parceria:** Explore parcerias ou acordos corporativos que possam trazer reservas mais estáveis e de longo prazo.
                    """)
                    recomendacoes_mostradas += 1


            if recomendacoes_mostradas == 0:
               st.info("ℹ️ **Risco Moderado Geral: Foco em Monitoramento e Comunicação Padrão**")
               st.markdown("""
               Nenhum dos fatores de risco mais críticos está ativo de forma proeminente neste cenário, mas a combinação dos fatores selecionados leva a um risco moderado.
               * **Monitoramento:** Mantenha um monitoramento regular sobre esta reserva, especialmente conforme a data do check-in se aproxima.
               * **Comunicação Padrão Otimizada:** Certifique-se de que todas as comunicações automáticas (e-mails de confirmação, lembretes de pagamento, informações de check-in) sejam claras, amigáveis e incluam informações essenciais para evitar dúvidas que possam levar a cancelamentos.
               * **Canais Abertos:** Tenha uma equipe pronta para responder rapidamente a quaisquer dúvidas ou solicitações do cliente, oferecendo suporte contínuo.
               """)