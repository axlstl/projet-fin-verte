import yfinance as yf
import pandas as pd

class DataImport:
    """
    Classe pour importer les données ESG de plusieurs entreprises via yfinance.
    """

    def __init__(self, tickers: list):
        self.tickers = tickers
        self.data = pd.DataFrame()

    def fetch_esg_data(self):
        """
        Récupère les données ESG pour chaque ticker dans la liste.
        Retourne un DataFrame consolidé.
        """
        records = []

        for ticker in self.tickers:
            try:
                t = yf.Ticker(ticker)
                esg = t.sustainability

                if esg is not None and not esg.empty:
                    esg = esg.transpose()
                    esg.reset_index(drop=True, inplace=True)
                    esg["Ticker"] = ticker
                    records.append(esg)
                else:
                    print(f"Aucune donnée ESG pour {ticker}")

            except Exception as e:
                print(f"Erreur lors de la récupération de {ticker} : {e}")

        if records:
            self.data = pd.concat(records, ignore_index=True)
        else:
            self.data = pd.DataFrame()

        columns_to_drop = [
            "alcoholic", "catholic", "controversialWeapons", "smallArms",
            "furLeather", "gambling", "gmo", "militaryContract", "nuclear",
            "pesticides", "palmOil", "coal", "tobacco", "adult",
            "peerHighestControversyPerformance",
            "percentile", "environmentPercentile", "socialPercentile", "governancePercentile",
            "highestControversy", "maxAge"
        ]

        self.data.drop(columns=columns_to_drop, inplace=True)

        return self.data
    
    def fetch_returns_data(self, start_date: str, end_date: str):
        """
        Récupère les rendements quotidiens pour chaque ticker dans la liste.
        Retourne un DataFrame avec les dates en index et une colonne pour chaque ticker.
        """
        # DataFrame pour stocker les rendements
        all_returns = pd.DataFrame()
        
        for ticker in self.tickers:
            try:
                # Téléchargement des données historiques
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if data is not None and not data.empty:
                    # Vérifiez si 'Adj Close' existe, sinon utilisez 'Close'
                    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                    
                    # Calcul des rendements quotidiens
                    returns = data[price_col].pct_change()
                    
                    # Renommer la série avec le symbole du ticker
                    returns.name = ticker
                    
                    # Ajouter au DataFrame principal
                    if all_returns.empty:
                        all_returns = pd.DataFrame(returns)
                    else:
                        all_returns[ticker] = returns
                else:
                    print(f"Aucune donnée historique pour {ticker}")

            except Exception as e:
                print(f"Erreur lors de la récupération des rendements pour {ticker} : {e}")
        
        # Supprimer la première ligne qui contient des NaN en raison du pct_change()
        if not all_returns.empty:
            all_returns = all_returns.iloc[1:].copy()
        
        return all_returns

from dico import tickers_brut
if __name__ == "__main__":
    # Correction ici: convertir dict_keys en liste de strings
    tickers = list(tickers_brut.keys())  # Portefeuille
    importer = DataImport(tickers)
    df_esg = importer.fetch_esg_data()

    print(df_esg)
