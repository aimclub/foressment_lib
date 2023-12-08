import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """
    Class for feature selection using elastic net or principal component analysis (PCA)

    :param method: The method for feature selection, choose from 'elastic_net' or 'pca'
    :type method: str

    :param params: Additional parameters for the selected method
    :type params: dict, optional

    :param selected_features: List of indices of the selected features
    :type selected_features: list, optional

    :param explained_ratio: Total explained ratio for the selected features (for PCA)
    :type explained_ratio: float, optional
    """

    def __init__(self, method='elastic_net', params=None):
        """
        Initialize the FeatureSelector with the specified method and parameters

        :param method: The method for feature selection, choose from 'elastic_net' or 'pca'
        :type method: str

        :param params: Additional parameters for the selected method
        :type params: dict, optional
        """
        self.method = method
        self.params = params
        self.selected_features = None
        self.explained_ratio = None

        if params is not None:
            if 'alpha' in params:
                alpha = params['alpha']
                if isinstance(alpha, float) and 0 < alpha < 1:
                    self.alpha = alpha
                else:
                    raise ValueError("'alpha' should be float between 0 and 1")
            if 'components' in params:
                components = params['components']
                if isinstance(components, int):
                    self.components = components
                else:
                    raise ValueError("'components' should be an integer")

        if not hasattr(self, 'alpha'):  # Set default values if not provided
            self.alpha = 0.1
        if not hasattr(self, 'components'):
            self.components = 2

    def fit_transform(self, X, y=None):
        """
        Fit the feature selection method to the input data and transform it

        :param X: The input features
        :type X: array-like

        :param y: The target variable (optional, not used for 'pca')
        :type y: array-like, optional

        :return: The transformed features
        :rtype: array-like
        """
        if self.method == 'elastic_net':
            return self.feature_selection(X, y, 'elastic_net')
        elif self.method == 'pca':
            return self.feature_selection(X, y=None, method='pca')
        else:
            raise ValueError("Invalid method. Choose 'elastic_net' or 'pca'.")

    def feature_selection(self, X, y=None, method='elastic_net'):
        if method == 'elastic_net':
            enet = ElasticNet(alpha=self.alpha)
            enet.fit(X, y)
            self.selected_features = np.nonzero(enet.coef_)[0]
            important_features = np.array(X)[:, self.selected_features]
        elif method == 'pca':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=self.components)
            X_transformed = pca.fit_transform(X_scaled)
            self.explained_ratio = pca.explained_variance_ratio_.sum()
            important_features = X_transformed
        else:
            raise ValueError("Invalid method. Choose 'elastic_net' or 'pca'.")

        self.print_report(method)
        """
        Perform feature selection based on the specified method

        :param X: The input features
        :type X: array-like

        :param y: The target variable (optional, not used for 'pca')
        :type y: array-like, optional

        :param method: The feature selection method, choose from 'elastic_net' or 'pca'
        :type method: str

        :return: The important features after selection
        :rtype: array-like
        """
        return important_features

    def print_report(self, method):
        """
        Print a report based on the selected feature selection method

        :param method: The feature selection method, choose from 'elastic_net' or 'pca'
        :type method: str
        """
        if method == 'elastic_net':
            select_num = len(self.selected_features)
            print(select_num, "Features Selected (using Elastic Net)")
        elif method == 'pca':
            print("Total Explained Ratio (PCA):", self.explained_ratio)
            if self.explained_ratio < 0.95:
                print(
                    "Warning: The total explained data portion is less than 95%; try increasing the value of components.")
        else:
            raise ValueError("Invalid method. Choose 'elastic_net' or 'pca'.")