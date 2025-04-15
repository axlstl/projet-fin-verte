import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
from datetime import datetime, timedelta
import warnings

# Ignorer les avertissements futurs de pandas/numpy si nécessaire
warnings.simplefilter(action='ignore', category=FutureWarning)

class PortfolioPerformanceAnalyzer:
    """
    Classe pour calculer les métriques de performance de portefeuilles,
    en distinguant les périodes de backtesting et de performance réelle.
    """
    def __init__(self, risk_free_rate=0.03, var_confidence_level=0.95, target_return_rate=0.00, trading_days_per_year=252):
        """
        Initialise l'analyseur avec les paramètres de configuration.
        """
        self.risk_free_rate = risk_free_rate
        self.var_confidence_level = var_confidence_level
        self.target_return_rate = target_return_rate
        self.trading_days_per_year = trading_days_per_year
        self.price_data = pd.DataFrame()

    # --- Fonctions Privées de Calcul des Métriques ---
    def _calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calcule les rendements journaliers."""
        return prices.pct_change().dropna()

    def _calculate_portfolio_returns(self, returns: pd.DataFrame, weights: dict) -> pd.Series:
        """Calcule les rendements pondérés d'un portefeuille."""
        # Assurer que les poids correspondent aux colonnes de returns
        aligned_weights = pd.Series(weights).reindex(returns.columns).fillna(0)
        portfolio_returns = (returns * aligned_weights).sum(axis=1)
        return portfolio_returns

    def _calculate_total_return(self, cumulative_returns: pd.Series) -> float:
        """Calcule le rendement total sur la période."""
        if cumulative_returns.empty:
            return 0.0
        return cumulative_returns.iloc[-1] - 1

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calcule le rendement annualisé."""
        if returns.empty:
            return 0.0
        num_years = len(returns) / self.trading_days_per_year
        if num_years == 0:
            return 0.0
        total_return = self._calculate_total_return((1 + returns).cumprod())
        # Gérer le cas où total_return est -1 (perte de 100%)
        if total_return <= -1.0:
             # Éviter log d'un nombre négatif ou zéro, retourne -100%
             return -1.0
        return (1 + total_return) ** (1 / num_years) - 1

    def _calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """Calcule la volatilité annualisée."""
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(self.trading_days_per_year)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sharpe."""
        if returns.empty:
            return 0.0
        annualized_return = self._calculate_annualized_return(returns)
        annualized_volatility = self._calculate_annualized_volatility(returns)
        if annualized_volatility == 0:
            return 0.0
        return (annualized_return - self.risk_free_rate) / annualized_volatility

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calcule l'écart-type des rendements négatifs (pour Sortino)."""
        downside_returns = returns[returns < self.target_return_rate / self.trading_days_per_year]
        if downside_returns.empty:
            return 0.0
        return downside_returns.std() * np.sqrt(self.trading_days_per_year)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sortino."""
        if returns.empty:
            return 0.0
        annualized_return = self._calculate_annualized_return(returns)
        downside_dev = self._calculate_downside_deviation(returns)
        if downside_dev == 0:
            return 0.0
        return (annualized_return - self.target_return_rate) / downside_dev

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calcule la perte maximale (Maximum Drawdown)."""
        if cumulative_returns.empty:
            return 0.0
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min() if not drawdown.empty else 0.0 # Retourne la perte maximale (valeur négative)

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio de Calmar."""
        if returns.empty:
            return 0.0
        annualized_return = self._calculate_annualized_return(returns)
        cumulative_returns = (1 + returns).cumprod()
        max_dd = self._calculate_max_drawdown(cumulative_returns)
        if max_dd == 0:
             # Si pas de drawdown, le ratio est techniquement infini, retourner une grande valeur ou NaN
             return np.inf if annualized_return > 0 else 0.0
        return annualized_return / abs(max_dd)

    def _calculate_var(self, returns: pd.Series) -> float:
        """Calcule la Value at Risk (VaR) historique."""
        if returns.empty:
            return 0.0
        return returns.quantile(1 - self.var_confidence_level) # Retourne une perte (valeur négative)

    def _calculate_cvar(self, returns: pd.Series) -> float:
        """Calcule la Conditional Value at Risk (CVaR) historique."""
        if returns.empty:
            return 0.0
        var = self._calculate_var(returns)
        cvar_returns = returns[returns <= var]
        return cvar_returns.mean() if not cvar_returns.empty else 0.0 # Retourne une perte moyenne (valeur négative)

    def _calculate_hit_ratio(self, returns: pd.Series) -> float:
        """Calcule le pourcentage de périodes avec rendement positif."""
        if returns.empty:
            return 0.0
        return (returns > 0).mean()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calcule le Profit Factor (Gains bruts / Pertes brutes)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0 # Éviter la division par zéro
        return gross_profit / gross_loss

    def _calculate_omega_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio Omega."""
        if returns.empty:
            return 0.0
        threshold_return_daily = self.target_return_rate / self.trading_days_per_year
        returns_above_threshold = returns[returns > threshold_return_daily]
        returns_below_threshold = returns[returns < threshold_return_daily]

        sum_gains = (returns_above_threshold - threshold_return_daily).sum()
        sum_losses = abs((returns_below_threshold - threshold_return_daily).sum())

        if sum_losses == 0:
            return np.inf if sum_gains > 0 else 1.0 # Éviter la division par zéro
        return sum_gains / sum_losses

    def _calculate_all_metrics(self, returns: pd.Series, period_name: str) -> dict:
        """Calcule toutes les métriques pour une série de rendements donnée."""
        if returns.empty or returns.isnull().all():
            print(f"Avertissement : Données de rendement vides ou NaN pour la période {period_name}. Métriques non calculées.")
            return {f"{metric}_{period_name}": np.nan for metric in [
                "Total Return (%)", "Annualized Return (%)", "Annualized Volatility (%)",
                "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio",
                f"VaR {self.var_confidence_level*100:.0f}% (%)", f"CVaR {self.var_confidence_level*100:.0f}% (%)",
                "Skewness", "Kurtosis", "Hit Ratio (%)", "Profit Factor", "Omega Ratio"
            ]}

        cumulative_returns = (1 + returns).cumprod()

        metrics = {
            f"Total Return (%)_{period_name}": self._calculate_total_return(cumulative_returns) * 100,
            f"Annualized Return (%)_{period_name}": self._calculate_annualized_return(returns) * 100,
            f"Annualized Volatility (%)_{period_name}": self._calculate_annualized_volatility(returns) * 100,
            f"Sharpe Ratio_{period_name}": self._calculate_sharpe_ratio(returns),
            f"Sortino Ratio_{period_name}": self._calculate_sortino_ratio(returns),
            f"Max Drawdown (%)_{period_name}": self._calculate_max_drawdown(cumulative_returns) * 100,
            f"Calmar Ratio_{period_name}": self._calculate_calmar_ratio(returns),
            f"VaR {self.var_confidence_level*100:.0f}% (%)_{period_name}": self._calculate_var(returns) * 100,
            f"CVaR {self.var_confidence_level*100:.0f}% (%)_{period_name}": self._calculate_cvar(returns) * 100,
            f"Skewness_{period_name}": skew(returns.dropna()),
            f"Kurtosis_{period_name}": kurtosis(returns.dropna()), # Excès de Kurtosis (Kurtosis - 3)
            f"Hit Ratio (%)_{period_name}": self._calculate_hit_ratio(returns) * 100,
            f"Profit Factor_{period_name}": self._calculate_profit_factor(returns),
            f"Omega Ratio_{period_name}": self._calculate_omega_ratio(returns)
        }
        return metrics

    # --- Méthodes Publiques ---
    def load_price_data(self, tickers: list, start_date: str, end_date: str) -> bool:
        """Charge les prix de clôture ajustés depuis Yahoo Finance."""
        try:
            print(f"Tentative de téléchargement pour les tickers: {tickers}")
            # Supprimer les tickers vides ou None
            valid_tickers = [t for t in tickers if t]
            if not valid_tickers:
                print("Erreur: Aucun ticker valide fourni.")
                self.price_data = pd.DataFrame()
                return False
                
            # Utiliser 'Adj Close' pour tenir compte des dividendes et splits
            data = yf.download(valid_tickers, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print("Erreur: yf.download a retourné un DataFrame vide.")
                self.price_data = pd.DataFrame()
                return False
                
            print(f"Structure des données téléchargées: {data.shape}, MultiIndex: {isinstance(data.columns, pd.MultiIndex)}")
            
            # Gérer différemment selon que les données ont un MultiIndex ou non
            if isinstance(data.columns, pd.MultiIndex):
                # Cas où il y a plusieurs tickers: data a un MultiIndex comme ('Adj Close', 'AAPL')
                if 'Adj Close' in data.columns.levels[0]:
                    price_data = data['Adj Close']
                    
                elif 'Close' in data.columns.levels[0]:
                    price_data = data['Close']
                  
                else:
                    
                    self.price_data = pd.DataFrame()
                    return False
            else:
                # Cas où il n'y a qu'un seul ticker: data n'a pas de MultiIndex
                if 'Adj Close' in data.columns:
                    price_data = data['Adj Close']
                    print("Extraction du prix 'Adj Close' pour un seul ticker.")
                    # Convertir en DataFrame si c'est une Series
                    if isinstance(price_data, pd.Series):
                        price_data = price_data.to_frame(name=valid_tickers[0])
                elif 'Close' in data.columns:
                    price_data = data['Close']
                    print("Avertissement: Utilisation du prix 'Close' pour un seul ticker car 'Adj Close' non disponible.")
                    # Convertir en DataFrame si c'est une Series
                    if isinstance(price_data, pd.Series):
                        price_data = price_data.to_frame(name=valid_tickers[0])
                else:
                    print("Erreur: Ni 'Adj Close' ni 'Close' trouvés dans les colonnes.")
                    self.price_data = pd.DataFrame()
                    return False
            
            # S'assurer que l'index est un DatetimeIndex
            price_data.index = pd.to_datetime(price_data.index)
            print(f"Shape après extraction des prix: {price_data.shape}")
            
            # Supprimer les colonnes qui sont entièrement NaN
            initial_cols = list(price_data.columns)
            price_data = price_data.dropna(axis=1, how='all')
            dropped_cols = [col for col in initial_cols if col not in price_data.columns]
            if dropped_cols:
                print(f"Avertissement: Colonnes entièrement NaN supprimées: {dropped_cols}")
            
            if price_data.empty:
                print("Avertissement: Aucune colonne avec des données valides après suppression des colonnes NaN.")
                self.price_data = pd.DataFrame()
                return False
                
            print(f"Shape après suppression des colonnes NaN: {price_data.shape}")
            
            # Remplir les NaN (d'abord forward, puis backward pour les NaNs initiaux)
            price_data_filled = price_data.ffill().bfill()
            
            # Supprimer les lignes où toutes les colonnes sont NaN
            price_data_cleaned = price_data_filled.dropna(how='all')
            
            self.price_data = price_data_cleaned
            
            if self.price_data.empty:
                print("Avertissement: Aucune donnée de prix après nettoyage complet.")
                return False
                
            print(f"Données de prix chargées et nettoyées pour {len(self.price_data.columns)} tickers de {self.price_data.index.min().date()} à {self.price_data.index.max().date()}.")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement ou du traitement des données de prix : {e}")
            import traceback
            traceback.print_exc()
            self.price_data = pd.DataFrame()
            return False

    def analyze_portfolios(self, portfolios_data: list, start_date_prices: str = "2018-01-01") -> pd.DataFrame:
        """
        Analyse une liste de portefeuilles et retourne un DataFrame avec les métriques.

        Args:
            portfolios_data (list): Liste de dictionnaires, chaque dict décrivant un portefeuille
                                    avec les clés 'name', 'tickers', 'weights', 'investment_start_date'.
            start_date_prices (str): Date de début pour charger l'historique des prix.

        Returns:
            pd.DataFrame: DataFrame contenant les métriques de performance pour chaque portefeuille.
        """
        if not portfolios_data:
            print("Aucune donnée de portefeuille fournie pour l'analyse.")
            return pd.DataFrame()

        all_tickers = list(set(ticker for p in portfolios_data for ticker in p.get('tickers', [])))
        if not all_tickers:
             print("Aucun ticker trouvé dans les données de portefeuille.")
             return pd.DataFrame()

        end_date_prices = datetime.now().strftime('%Y-%m-%d')

        print(f"\nChargement des données de prix pour {len(all_tickers)} actifs...")
        if not self.load_price_data(all_tickers, start_date_prices, end_date_prices):
            print("Impossible de calculer les métriques car les données de prix n'ont pas pu être chargées.")
            return pd.DataFrame()

        all_metrics_results = []
        print("\nCalcul des métriques de performance pour chaque portefeuille...")

        for portfolio in portfolios_data:
            portfolio_name = portfolio.get('name', 'Portefeuille Inconnu')
            tickers = portfolio.get('tickers', [])
            weights = portfolio.get('weights', {})
            try:
                investment_start_date = pd.to_datetime(portfolio.get('investment_start_date'))
                if pd.isna(investment_start_date):
                    raise ValueError("Date de début d'investissement invalide ou manquante.")
            except Exception as e:
                 print(f"Erreur de date pour {portfolio_name}: {e}. Portefeuille ignoré.")
                 continue

            print(f"--- Traitement de {portfolio_name} ---")

            if not tickers or not weights:
                print(f"Avertissement: Tickers ou poids manquants pour {portfolio_name}. Portefeuille ignoré.")
                continue

            # Vérifier si tous les tickers sont présents dans les données chargées
            available_tickers = [t for t in tickers if t in self.price_data.columns]
            missing_tickers = [t for t in tickers if t not in self.price_data.columns]
            if missing_tickers:
                print(f"Avertissement: Tickers manquants pour {portfolio_name}: {missing_tickers}. Ils seront ignorés.")
            if not available_tickers:
                 print(f"Avertissement: Aucun ticker disponible pour {portfolio_name} après filtrage. Portefeuille ignoré.")
                 continue

            # Ajuster les poids pour ne considérer que les tickers disponibles
            available_weights = {t: w for t, w in weights.items() if t in available_tickers}
            # Normaliser les poids si certains tickers ont été retirés
            total_weight = sum(available_weights.values())
            if total_weight > 0 and not np.isclose(total_weight, 1.0):
                 print(f"Avertissement: Normalisation des poids pour {portfolio_name} car certains tickers sont manquants.")
                 available_weights = {t: w / total_weight for t, w in available_weights.items()}
            elif total_weight == 0:
                 print(f"Avertissement: Poids totaux nuls pour les tickers disponibles de {portfolio_name}. Portefeuille ignoré.")
                 continue


            # Sélectionner les prix et calculer les rendements des actifs du portefeuille
            asset_prices = self.price_data[available_tickers]
            # S'assurer qu'il n'y a pas de lignes entièrement NaN avant pct_change
            asset_prices = asset_prices.dropna(how='all')
            if asset_prices.empty:
                 print(f"Avertissement: Aucune donnée de prix pour les tickers disponibles de {portfolio_name}. Portefeuille ignoré.")
                 continue

            asset_returns = self._calculate_returns(asset_prices)

            # Calculer les rendements du portefeuille
            portfolio_returns = self._calculate_portfolio_returns(asset_returns, available_weights)

            # Séparer les périodes de backtesting et live
            backtest_returns = portfolio_returns[portfolio_returns.index < investment_start_date]
            live_returns = portfolio_returns[portfolio_returns.index >= investment_start_date]

            print(f"Période Backtesting: {backtest_returns.index.min().date() if not backtest_returns.empty else 'N/A'} -> {backtest_returns.index.max().date() if not backtest_returns.empty else 'N/A'} ({len(backtest_returns)} jours)")
            print(f"Période Live: {live_returns.index.min().date() if not live_returns.empty else 'N/A'} -> {live_returns.index.max().date() if not live_returns.empty else 'N/A'} ({len(live_returns)} jours)")

            # Calculer les métriques pour les deux périodes
            metrics_backtest = self._calculate_all_metrics(backtest_returns, "Backtest")
            metrics_live = self._calculate_all_metrics(live_returns, "Live")

            # Combiner les métriques
            combined_metrics = {"Portfolio": portfolio_name}
            combined_metrics.update(metrics_backtest)
            combined_metrics.update(metrics_live)
            all_metrics_results.append(combined_metrics)

        # Créer un DataFrame final avec tous les résultats
        if all_metrics_results:
            results_df = pd.DataFrame(all_metrics_results)
            results_df = results_df.set_index("Portfolio")
            return results_df
        else:
            print("\nAucune métrique n'a pu être calculée (vérifiez les données de portefeuille et de prix).")
            return pd.DataFrame()

# --- Exemple d'utilisation (peut être appelé depuis un notebook) ---
if __name__ == "__main__":
    # --- REMPLACER PAR LA LOGIQUE RÉELLE DE CHARGEMENT DES PORTFEUILLES ---
    # Exemple statique si l'import échoue ou n'est pas disponible
    print("Avertissement: Utilisation de données de portefeuille statiques pour l'exemple.")
    portfolios_to_analyze = [
        {
            "name": "Portefeuille Alpha",
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "weights": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
            "investment_start_date": "2023-01-01" # Date de début de l'investissement réel
        },
        {
            "name": "Portefeuille Beta",
            "tickers": ["TSLA", "NVDA", "AMD", "NONEXISTENT"], # Ajout d'un ticker inexistant pour tester
            "weights": {"TSLA": 0.5, "NVDA": 0.3, "AMD": 0.2, "NONEXISTENT": 0.1},
            "investment_start_date": "2022-06-01"
        },
        {
            "name": "Portefeuille Gamma (Crypto)",
            "tickers": ["BTC-USD", "ETH-USD"],
            "weights": {"BTC-USD": 0.6, "ETH-USD": 0.4},
            "investment_start_date": "2023-05-01"
        },
         {
            "name": "Portefeuille Vide",
            "tickers": [],
            "weights": {},
            "investment_start_date": "2023-01-01"
        }
    ]
    # --- FIN DE LA PARTIE À REMPLACER ---

    # Créer une instance de l'analyseur
    analyzer = PortfolioPerformanceAnalyzer()

    # Analyser les portefeuilles
    results = analyzer.analyze_portfolios(portfolios_to_analyze)

    # Afficher les résultats
    if not results.empty:
        print("\n--- Résultats des Métriques de Performance ---")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        # Afficher transposé pour une meilleure lisibilité si peu de portefeuilles
        if len(results.columns) > len(results.index):
             print(results.round(2))
        else:
             print(results.round(2).T)

        # Sauvegarder les résultats dans un fichier CSV
        try:
            results.to_csv("performance_metrics_results.csv")
            print("\nRésultats sauvegardés dans 'performance_metrics_results.csv'")
        except Exception as e:
            print(f"\nErreur lors de la sauvegarde des résultats : {e}")
    else:
        print("\nL'analyse n'a produit aucun résultat.")
    # Pour éviter que le script ne se termine immédiatement dans certains environnements


