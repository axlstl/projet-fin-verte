import pandas as pd
import ast
from datetime import datetime


class PortfolioBuilder:
    def __init__(self, dataframe):
        """
        Initialise la classe avec un dataframe contenant des scores ESG et des tickers afin de créer des portefeuille.
        """
        self.dataframe = dataframe

    def calculate_weights(self, tickers, score_column = "totalEsg"):
        """
        Calcule les poids de chaque action dans le portefeuille en fonction de son score.

        """

        df = self.dataframe.copy()

        # Filtration du dataframe pour ne garder que les tickers sélectionnés
        df_filtered = df[df["Ticker"].isin(tickers)].copy()
        
        df_filtered["Inverse_Score"] = 1 / df_filtered[score_column]
        total_inverse_score = df_filtered["Inverse_Score"].sum()
        # Calcul des poids pour chaque ticker
        df_filtered["Weight"] = df_filtered["Inverse_Score"] / total_inverse_score

        return df_filtered.set_index("Ticker")["Weight"].to_dict()

    def create_portfolio_dict(self, portfolio_name, tickers, weights, investment_start_date=None):
        """
        Crée un dictionnaire de portefeuille au format attendu par PortfolioPerformanceAnalyzer
        
        Args:
            portfolio_name (str): Nom du portefeuille
            tickers (list): Liste des tickers
            weights (dict): Dictionnaire des poids pour chaque ticker
            investment_start_date (str): Date de début d'investissement au format YYYY-MM-DD
            
        Returns:
            dict: Dictionnaire de portefeuille formaté
        """
        if investment_start_date is None:
            # Utiliser la date actuelle si aucune date n'est fournie
            investment_start_date = datetime.now().strftime('%Y-%m-%d')
            
        return {
            "name": portfolio_name,
            "tickers": tickers,
            "weights": weights,
            "investment_start_date": investment_start_date
        }

    def environnement(self, investment_start_date=None, return_dict=False):
        """
        Sélectionne les entreprises pour le portefeuille axé sur l'environnement.
        
        Args:
            investment_start_date (str): Date de début d'investissement (format YYYY-MM-DD)
            return_dict (bool): Si True, retourne un dictionnaire complet au format de PortfolioPerformanceAnalyzer
                              Si False, retourne uniquement le dictionnaire des poids
        
        Returns:
            dict: Poids du portefeuille ou dictionnaire complet
        """

        df = self.dataframe.copy()

        # On filtre les entreprises ayant un score environnemental inférieur à la moyenne de son secteur
        # + dont le lien avec les tests sur animaux est faux
        # + dont le score ESG est inférieur à 30
        df["peerEnv_avg"] = df["peerEnvironmentPerformance"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["peerEnv_avg"] = df["peerEnv_avg"].apply(lambda d: d.get("avg") if isinstance(d, dict) else None)
        filtered_df = df[(df["animalTesting"] != True) & (df["environmentScore"] < df["peerEnv_avg"]) & (df["totalEsg"] < 30)]

        # Retourne la liste des tickers sélectionnés (top 30% des entreprises avec le meilleur score environnemental)
        tickers = filtered_df[filtered_df["environmentScore"] <= filtered_df["environmentScore"].quantile(0.30)]["Ticker"].tolist()
        weights = self.calculate_weights(tickers)
        
        if return_dict:
            return self.create_portfolio_dict("Environnement", tickers, weights, investment_start_date)
        else:
            return weights
    
    def social(self, investment_start_date=None, return_dict=False):
        """
        Sélectionne les entreprises pour le portefeuille axé sur le social.
        
        Args:
            investment_start_date (str): Date de début d'investissement (format YYYY-MM-DD)
            return_dict (bool): Si True, retourne un dictionnaire complet au format de PortfolioPerformanceAnalyzer
                              Si False, retourne uniquement le dictionnaire des poids
        
        Returns:
            dict: Poids du portefeuille ou dictionnaire complet
        """

        df = self.dataframe.copy()

        # On filtre les entreprises ayant un score social inférieur à la moyenne de son secteur 
        # + dont le lien avec les tests sur animaux est faux
        # + dont le score ESG est inférieur à 30
        df["peerSoc_avg"] = df["peerSocialPerformance"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["peerSoc_avg"] = df["peerSoc_avg"].apply(lambda d: d.get("avg") if isinstance(d, dict) else None)
        filtered_df = df[(df["animalTesting"] != True) & (df["socialScore"] < df["peerSoc_avg"]) & (df["totalEsg"] < 30)]

        # Retourne la liste des tickers sélectionnés (top 30% des entreprises avec le meilleur score social)
        tickers = filtered_df[filtered_df["socialScore"] <= filtered_df["socialScore"].quantile(0.30)]["Ticker"].tolist()
        weights = self.calculate_weights(tickers)
        
        if return_dict:
            return self.create_portfolio_dict("Social", tickers, weights, investment_start_date)
        else:
            return weights


    def ESG(self, investment_start_date=None, return_dict=False):
        """
        Sélectionne les entreprises pour le portefeuille axé sur l'ESG.
        
        Args:
            investment_start_date (str): Date de début d'investissement (format YYYY-MM-DD)
            return_dict (bool): Si True, retourne un dictionnaire complet au format de PortfolioPerformanceAnalyzer
                              Si False, retourne uniquement le dictionnaire des poids
        
        Returns:
            dict: Poids du portefeuille ou dictionnaire complet
        """

        df = self.dataframe.copy()

        # On extrait le score ESG moyen du secteur 
        df["peerESG_dict"] = df["peerEsgScorePerformance"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["peerESG_avg"] = df["peerESG_dict"].apply(lambda d: d.get("avg") if isinstance(d, dict) else None)


        # Filtre sur les critères choisis :
        # On filtre les entreprises ayant un score ESG inférieur à la moyenne de son secteur
        # + dont le lien avec les tests sur animaux est faux
        # + dont le score ESG est inférieur à 30
        filtered_df = df[(df["animalTesting"] != True) & (df["totalEsg"] < df["peerESG_avg"]) & (df["totalEsg"] < 30)]

        # On tri selon si l'entreprise est en avance ou non sur sont secteur
        esg_order = ["LAG_PERF", "UNDER_PERF", "AVG_PERF", "LEAD_PERF"]
        filtered_df["esgPerformance"] = pd.Categorical(filtered_df["esgPerformance"], categories=esg_order, ordered=True)
        df_sorted = filtered_df.sort_values(by=["esgPerformance","totalEsg"], ascending=[False,True])

        # Retourne la liste des tickers sélectionnés (top 25% des entreprises avec le meilleur score ESG)
        tickers = df_sorted[df_sorted["esgPerformance"].isin(["LEAD_PERF", "AVG_PERF"])]["Ticker"].tolist()
        weights = self.calculate_weights(tickers)
        
        if return_dict:
            return self.create_portfolio_dict("ESG", tickers, weights, investment_start_date)
        else:
            return weights
            
    def generate_all_portfolios(self, investment_start_date=None):
        """
        Génère tous les portefeuilles disponibles au format attendu par PortfolioPerformanceAnalyzer
        
        Args:
            investment_start_date (str): Date de début d'investissement (format YYYY-MM-DD)
            
        Returns:
            list: Liste de dictionnaires représentant les portefeuilles
        """
        return [
            self.environnement(investment_start_date, return_dict=True),
            self.social(investment_start_date, return_dict=True),
            self.ESG(investment_start_date, return_dict=True)
        ]

