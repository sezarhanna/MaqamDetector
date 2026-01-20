import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier as SklearnMLP

class MLPClassifier:
    """
    Feed-Forward Neural Network for Maqam Classification.
    Uses 36x36 transitions or statistical features as input.
    """
    def __init__(self, model_path="mlp_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.classes_ = []
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.classes_ = saved_data['classes']
                print(f"Loaded MLP Model with classes: {self.classes_}")
        else:
            print("No MLP model found. Initializing new classifier.")
            # Architecture: Input -> 512 -> 256 -> Output
            self.model = SklearnMLP(hidden_layer_sizes=(512, 256), 
                                    activation='relu', 
                                    solver='adam', 
                                    max_iter=500,
                                    random_state=42)

    def train(self, X_train, y_train):
        """
        X_train: List of flattened transition matrices (arrays of shape 1296)
        y_train: List of string labels
        """
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Save
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'classes': self.classes_}, f)
        print(f"MLP Model saved to {self.model_path}")

    def predict_proba(self, transition_matrix):
        """
        Predicts probabilities for a single transition matrix (36x36).
        """
        if not self.is_trained():
            return {}

        # Flatten 36x36 -> 1296 features
        features = transition_matrix.flatten().reshape(1, -1)
        
        probs = self.model.predict_proba(features)[0]
        return {cls: prob for cls, prob in zip(self.classes_, probs)}

    def is_trained(self):
        # Sklearn models usually have 'coefs_' attribute after fitting
        return hasattr(self.model, 'coefs_')
