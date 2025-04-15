import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data import DataImport
from dico import tickers_brut
from portfolio import PortfolioBuilder
from performancemetrics import PortfolioPerformanceAnalyzer

#cette fonction permet de charger les données des portefeuilles au lancement de l'application
@st.cache_data
def load_data():
    tickers = list(tickers_brut.keys())
    importer = DataImport(tickers)

    df_esg = importer.fetch_esg_data()
    start_date = "2015-01-01"
    end_date = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df_returns = importer.fetch_returns_data(start_date, end_date)
    
    return tickers, df_esg, df_returns

# Ici, on crée les portefeuilles. cela permet d'avoir un portefeuille adapté à la date de lancement de l'application.
@st.cache_data
def create_portfolios(df_esg):
    pb = PortfolioBuilder(df_esg)
    portefeuille_environnement = pb.environnement()
    portefeuille_social = pb.social()
    portefeuille_esg = pb.ESG()
    

    df_esg["ratingDate"] = pd.to_datetime(df_esg["ratingYear"].astype(str) + "-" + df_esg["ratingMonth"].astype(str) + "-01")
    most_recent_date = df_esg["ratingDate"].max()
    
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

#Ici, on utilise la classe qui calcule les métriques des portefeuilles
@st.cache_data
def analyze_portfolios(portfolios_to_analyze):
    analyzer = PortfolioPerformanceAnalyzer()
    resultats = analyzer.analyze_portfolios(portfolios_to_analyze)
    return resultats

# Nouvelle fonction pour calculer les rendements et rendements cumulés
@st.cache_data
def calculate_portfolio_performance(df_returns, tickers, weights):
   
    #On initie la classe PPA dans analyzer et on calcule les rendements cumulés
    analyzer = PortfolioPerformanceAnalyzer()

    portfolio_returns = df_returns[tickers].copy()
    weighted_returns = analyzer._calculate_portfolio_returns(portfolio_returns, weights)

    cumulative_returns = (1 + weighted_returns).cumprod() - 1
    return weighted_returns, cumulative_returns

# On se sert de la fonction juste au dessus et onen fait un graphque
def plot_cumulative_returns(df_returns, tickers, weights, title):

    _, cumulative_returns = calculate_portfolio_performance(df_returns, tickers, weights)
    

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values * 100, 
        mode='lines',
        name=title,
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'Rendements cumulés - Portefeuille {title}',
        xaxis_title='Année',
        yaxis_title='Rendement cumulé (en %)',
        template='plotly_white',
        height=500
    )
    
    return fig

#La fonction permet d'avoir un environnement uniformisé pour chaque onglet de portefeuille
def show_portfolio_details(portfolio_name, weights, df_esg, df_returns, metrics):
    st.header(f"Portefeuille {portfolio_name}")
    
    #On a d'abord la pondération de chaque actif dans le portefeuille
    st.subheader("Composition du portefeuille")
    fig_weights = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        title=f"Répartition des actifs - Portefeuille {portfolio_name}"
    )
    st.plotly_chart(fig_weights, use_container_width=True)
    
    #On a ici les scores ESG des actifs du portefeuille
    st.subheader("Détails des scores ESG des actifs du portefeuille")
    portfolio_tickers = list(weights.keys())
    if portfolio_tickers:
        portfolio_esg = df_esg[df_esg['Ticker'].isin(portfolio_tickers)][['Ticker', 'totalEsg', 'environmentScore', 'socialScore', 'governanceScore', 'peerGroup']]
        st.dataframe(portfolio_esg.set_index('Ticker'), use_container_width=True)
    
    #On insère maintenant le graphique des rendements cumulés
    st.subheader("Rendements")
    fig_returns = plot_cumulative_returns(df_returns, portfolio_tickers, weights, portfolio_name)
    st.plotly_chart(fig_returns, use_container_width=True)
    
    #Enfin, on affiche tous les ratios de performance
    st.subheader("Métriques de performance")
    
    #Construction des deux colonnes en bas de page relatives aux metrics
    #Ici on distingue les métriques de backtest, et celles en direct (à partir du 1er avril2025)
    col_backtest, col_live = st.columns(2)
    
    backtest_metrics = metrics.loc[portfolio_name, [col for col in metrics.columns if '_Backtest' in col]].reset_index()
    backtest_metrics.columns = ['Métrique', 'Valeur']
    backtest_metrics['Métrique'] = backtest_metrics['Métrique'].str.replace('_Backtest', '')
    
    live_metrics = metrics.loc[portfolio_name, [col for col in metrics.columns if '_Live' in col]].reset_index()
    live_metrics.columns = ['Métrique', 'Valeur']
    live_metrics['Métrique'] = live_metrics['Métrique'].str.replace('_Live', '')
    with col_backtest:
        st.subheader("Métriques Backtest")
        st.table(backtest_metrics.set_index('Métrique'))
    
    with col_live:
        st.subheader("Métriques de l'investissement depuis le 1er avril 2025")
        st.table(live_metrics.set_index('Métrique'))


#Cela permet de créer le graphique comparatif des rendements cumulés de chaque portefeuille
# On se sert de la fonction juste au dessus et on en fait un graphque
def create_comparison_chart(portfolios_data, df_returns):
    fig_comparison = go.Figure() #on initialise le graphique
    
    for portfolio_name, weights in portfolios_data.items():
        portfolio_tickers = list(weights.keys())
        
        #On calcule la performance des rendements cumulés
        _, cumulative_returns = calculate_portfolio_performance(df_returns, portfolio_tickers, weights)
        

        fig_comparison.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name=portfolio_name
        ))
    
    fig_comparison.update_layout(
        title='Comparaison des rendements cumulés sur la période de backtest',
        xaxis_title='Année',
        yaxis_title='Rendement cumulé ( en %)',
        template='plotly_white',
        height=500
    )
    
    return fig_comparison