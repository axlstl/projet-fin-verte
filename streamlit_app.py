import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data import DataImport
from dico import tickers_brut
from portfolio import PortfolioBuilder
from performancemetrics import PortfolioPerformanceAnalyzer

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Portefeuilles ESG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les données
@st.cache_data
def load_data():
    # Chargement des tickers
    tickers = list(tickers_brut.keys())
    importer = DataImport(tickers)
    
    # Chargement des données ESG
    df_esg = importer.fetch_esg_data()
    
    # Chargement des rendements
    start_date = "2015-01-01"
    end_date = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df_returns = importer.fetch_returns_data(start_date, end_date)
    
    return tickers, df_esg, df_returns

# Fonction pour créer les portefeuilles
@st.cache_data
def create_portfolios(df_esg):
    pb = PortfolioBuilder(df_esg)
    portefeuille_environnement = pb.environnement()
    portefeuille_social = pb.social()
    portefeuille_esg = pb.ESG()
    
    # Préparation pour analyse
    df_esg["ratingDate"] = pd.to_datetime(df_esg["ratingYear"].astype(str) + "-" + df_esg["ratingMonth"].astype(str) + "-01")
    most_recent_date = df_esg["ratingDate"].max()
    
    # Création des dictionnaires de portefeuille au format attendu par l'analyseur
    portfolios_to_analyze = []
    
    portfolios_data = {
        "Environnement": portefeuille_environnement,
        "Social": portefeuille_social, 
        "ESG": portefeuille_esg
    }
    
    for name, weights in portfolios_data.items():
        portfolio_dict = {
            "name": name,
            "tickers": list(weights.keys()),
            "weights": weights,
            "investment_start_date": most_recent_date.strftime('%Y-%m') + "-01"
        }
        portfolios_to_analyze.append(portfolio_dict)
    
    return portfolios_data, portfolios_to_analyze

# Fonction pour analyser les portefeuilles
@st.cache_data
def analyze_portfolios(portfolios_to_analyze):
    analyzer = PortfolioPerformanceAnalyzer()
    resultats = analyzer.analyze_portfolios(portfolios_to_analyze)
    return resultats

# Fonction pour créer un graphique des rendements cumulés
def plot_cumulative_returns(df_returns, tickers, weights, title):
    # Sélectionner les tickers du portefeuille
    portfolio_returns = df_returns[tickers].copy()
    
    # Calculer les rendements pondérés
    weighted_returns = pd.Series(0.0, index=portfolio_returns.index)
    for ticker, weight in weights.items():
        if ticker in portfolio_returns.columns:
            weighted_returns += portfolio_returns[ticker] * weight
    
    # Calculer les rendements cumulés
    cumulative_returns = (1 + weighted_returns).cumprod() - 1
    
    # Créer le graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values * 100,  # Convertir en pourcentage
        mode='lines',
        name=title,
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'Rendements cumulés - {title}',
        xaxis_title='Date',
        yaxis_title='Rendement cumulé (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

# Fonction pour afficher les détails du portefeuille
def show_portfolio_details(portfolio_name, weights, df_esg, df_returns, metrics):
    st.header(f"Portefeuille {portfolio_name}")
    
    # Section 1: Répartition des poids
    st.subheader("Composition du portefeuille")
    fig_weights = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        title=f"Répartition des actifs - {portfolio_name}"
    )
    st.plotly_chart(fig_weights, use_container_width=True)
    
    # Section 2: Tableau des actifs
    st.subheader("Détails des actifs")
    portfolio_tickers = list(weights.keys())
    if portfolio_tickers:
        portfolio_esg = df_esg[df_esg['Ticker'].isin(portfolio_tickers)][['Ticker', 'totalEsg', 'environmentScore', 'socialScore', 'governanceScore', 'peerGroup']]
        st.dataframe(portfolio_esg.set_index('Ticker'), use_container_width=True)
    
    # Section 3: Graphique des rendements
    st.subheader("Rendements")
    fig_returns = plot_cumulative_returns(df_returns, portfolio_tickers, weights, portfolio_name)
    st.plotly_chart(fig_returns, use_container_width=True)
    
    # Section 4: Métriques de performance
    st.subheader("Métriques de performance")
    
    if metrics is not None and portfolio_name in metrics.index:
        # Créer deux colonnes
        col1, col2 = st.columns(2)
        
        # Préparer les données pour les tableaux
        backtest_metrics = metrics.loc[portfolio_name, [col for col in metrics.columns if '_Backtest' in col]].reset_index()
        backtest_metrics.columns = ['Métrique', 'Valeur']
        backtest_metrics['Métrique'] = backtest_metrics['Métrique'].apply(lambda x: x.replace('_Backtest', ''))
        
        live_metrics = metrics.loc[portfolio_name, [col for col in metrics.columns if '_Live' in col]].reset_index()
        live_metrics.columns = ['Métrique', 'Valeur']
        live_metrics['Métrique'] = live_metrics['Métrique'].apply(lambda x: x.replace('_Live', ''))
        
        # Afficher les tableaux
        with col1:
            st.subheader("Métriques Backtest")
            st.table(backtest_metrics.set_index('Métrique'))
        
        with col2:
            st.subheader("Métriques Live")
            st.table(live_metrics.set_index('Métrique'))
    else:
        st.warning("Aucune métrique disponible pour ce portefeuille.")


# INTERFACE PRINCIPALE
st.title("Dashboard de Portefeuilles ESG")

# Chargement des données avec indication de progression
with st.spinner('Chargement des données...'):
    tickers, df_esg, df_returns = load_data()
    portfolios_data, portfolios_to_analyze = create_portfolios(df_esg)
    metrics = analyze_portfolios(portfolios_to_analyze)

# Onglets principaux
tab_overview, tab_env, tab_social, tab_esg = st.tabs([
    "📋 Présentation", 
    "🌿 Portefeuille Environnement", 
    "👪 Portefeuille Social", 
    "🌐 Portefeuille ESG"
])

# Onglet Présentation
with tab_overview:
    st.header("Présentation des portefeuilles ESG")
    
    st.write("""
    Cette application présente trois portefeuilles construits selon différentes stratégies ESG :
    
    - **Portefeuille Environnement** : Focalisé sur les entreprises ayant de bonnes performances environnementales
    - **Portefeuille Social** : Focalisé sur les entreprises ayant de bonnes performances sociales
    - **Portefeuille ESG** : Équilibré entre les critères environnementaux, sociaux et de gouvernance
    """)
    
    # Comparaison des rendements cumulés
    st.subheader("Comparaison des performances")
    
    fig_comparison = go.Figure()
    
    for portfolio_name, weights in portfolios_data.items():
        portfolio_tickers = list(weights.keys())
        
        # Calculer les rendements pondérés
        portfolio_returns = df_returns[portfolio_tickers].copy()
        weighted_returns = pd.Series(0.0, index=portfolio_returns.index)
        for ticker, weight in weights.items():
            if ticker in portfolio_returns.columns:
                weighted_returns += portfolio_returns[ticker] * weight
                
        # Calculer les rendements cumulés
        cumulative_returns = (1 + weighted_returns).cumprod() - 1
        
        # Ajouter au graphique
        fig_comparison.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name=portfolio_name
        ))
    
    fig_comparison.update_layout(
        title='Comparaison des rendements cumulés',
        xaxis_title='Date',
        yaxis_title='Rendement cumulé (%)',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Tableau comparatif des métriques clés
    st.subheader("Comparaison des métriques clés")
    
    if not metrics.empty:
        key_metrics = [
            'Total Return (%)_Backtest', 
            'Annualized Return (%)_Backtest',
            'Sharpe Ratio_Backtest', 
            'Max Drawdown (%)_Backtest',
            'Total Return (%)_Live',
            'Sharpe Ratio_Live'
        ]
        
        selected_metrics = metrics[key_metrics].copy()
        # Renommer les colonnes pour plus de clarté
        selected_metrics.columns = [
            'Rendement Total (%) - Backtest', 
            'Rendement Annualisé (%) - Backtest',
            'Ratio de Sharpe - Backtest', 
            'Drawdown Max (%) - Backtest',
            'Rendement Total (%) - Live',
            'Ratio de Sharpe - Live'
        ]
        
        st.table(selected_metrics.round(2))
    
# Onglet Environnement
with tab_env:
    show_portfolio_details("Environnement", portfolios_data["Environnement"], df_esg, df_returns, metrics)

# Onglet Social
with tab_social:
    show_portfolio_details("Social", portfolios_data["Social"], df_esg, df_returns, metrics)

# Onglet ESG
with tab_esg:
    show_portfolio_details("ESG", portfolios_data["ESG"], df_esg, df_returns, metrics)

# Footer
st.markdown("---")
st.caption("Dashboard développé pour visualiser les portefeuilles ESG et leurs performances")