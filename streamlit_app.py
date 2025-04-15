import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Importer les fonctions de configuration
from streamlit_config import (
    load_data, create_portfolios, analyze_portfolios, 
    show_portfolio_details, create_comparison_chart
)

#paramètres de l'application (nom de l'onglet et titre du projet)
st.set_page_config(
    page_title="Projet Finance Verte",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Projet Finance Verte - Présentation de 3 portefeuilles ESG")


with st.spinner('Chargement des données - Projet réalisé par Tristan Voirin, Axel Sottile et céline Nevo.'):
    tickers, df_esg, df_returns = load_data()
    portfolios_data, portfolios_to_analyze = create_portfolios(df_esg)
    metrics = analyze_portfolios(portfolios_to_analyze).rename_axis("Portefeuille")

# on initie l'ensemble des onglets de l'interface
tab_overview, tab_env, tab_social, tab_esg = st.tabs([
    "📋 Présentation du projet", 
    "🌿 Portefeuille Environnement", 
    "👪 Portefeuille Social", 
    "🌐 Portefeuille ESG"
])

#1er onglet : 
with tab_overview:
    st.header("Présentation des portefeuilles ")
    st.write("""
    Cette application présente le résultat du projet de la matière Économie de l'énergie et de l'environnement dirigé par Jérémy Dudek. Ce projet a été réalisé par Céline Nevo, Axel Sottile et Tristan Voirin. Nous avons eu comme idée de créer trois portefeuilles se basant sur des critères extra-financiers. L'ensemble des données relatives aux performances ESG ont été récupérées grâce à la librairie Yahoo Finance.
    \n
    Les performances des portefeuilles ont été calculées sur deux périodes : 
    - **Une période de backtest** : de 2015 au 1er avril 2025
    - **Une période de test en direct** : du 1er avril 2025 à aujourd'hui. Cela permet de mesurer les performances du portefeuille depuis sa date d'investissement.
    \n
    Afin de construire ces portefeuilles, nous avons suivi une approche Best-in-Class à partir de 80 entreprises du S\&P 500, c'est-à-dire que nous avons sélectionné les entreprises qui performent le mieux dans la thématique correspondante par rapport à la moyenne des entreprises du même secteur. Par ailleurs, toutes les entreprises qui effectuent des tests sur les animaux ont été exclues de la sélection. Enfin, le portefeuille ESG combine les trois scores (Environnement, Social et Gouvernance) et inclut également un indicateur reflétant le positionnement ESG des entreprises par rapports à ses concurrents, en combinant les performances ESG actuelles avec les signaux d'engagement des entreprises. 
    \n
    Les performances du portefeuille Social sont accentuées car elles contiennent des parts de l'entreprise NVIDIA qui a connu une énorme croissances ces dernières années. Par ailleurs, les performances de l'investissement en direct sont relativement faibles à court terme car elles subissent le choc lié à l'incertitude autour de l'inflation et de la mise en place des droits de douanes par les États-Unis.
    \n
    Le programme calcule et stocke les rebalancements du portefeuille en fonction des évolutions des scores extra-financiers. Nous n'affichons sur streamlit que la dernière version des allocations du portefeuille. Toutefois, le calcul des performances intègre ces changements.
    """)
    
    #On affiche les performances globales (évolution des rendements et des métriques clés)
    st.subheader("Comparaison des performances des portefeuilles")
    fig_comparison = create_comparison_chart(portfolios_data, df_returns)
    st.plotly_chart(fig_comparison, use_container_width=True)
    


    #Ici, on s'occupe de sélectionner les metrics importantes par rapport à la classe d'analyse de portefeuille
    st.subheader("Comparaison des métriques clés")    
    key_metrics = [
        'Rendement Total (%)_Backtest', 
        'Rendement Annualisé (%)_Backtest',
        'Ratio de Sharpe_Backtest', 
        'Drawdown Max (%)_Backtest',
        'Rendement Total (%)_Live',
        'Ratio de Sharpe_Live']
    
    #j'ai refait un tableau parce que c'étiat plus simple que de remodifier la classe
    selected_metrics = metrics[key_metrics].copy()
    selected_metrics.columns = [
        'Rendement Total (%) (Backtest)', 
        'Rendement Annualisé (%) (Backtest)',
        'Ratio de Sharpe (Backtest)', 
        'Drawdown Max (%) (Backtest)',
        'Rendement Total (%) (Direct)',
        'Ratio de Sharpe (Direct)']
    st.table(selected_metrics.round(2))
        
    
#mise en place des trois onglets
with tab_env:
    show_portfolio_details("Environnement", portfolios_data["Environnement"], df_esg, df_returns, metrics)
with tab_social:
    show_portfolio_details("Social", portfolios_data["Social"], df_esg, df_returns, metrics)
with tab_esg:
    show_portfolio_details("ESG", portfolios_data["ESG"], df_esg, df_returns, metrics)

#pied de page
st.markdown("---")
st.caption("Projet de comparaison de portefeuilles pour la matière Économie de l'énergie et de l'environnement - Réalisé par Axel Sottile, Céline Nevo et Tristan Voirin")