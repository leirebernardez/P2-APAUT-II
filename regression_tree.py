import numpy as np

class Nodo:
    """
    Nodo del árbol de regresión.
    Si es hoja -> value ≠ None y f_index = threshold = None
    Si no es hoja -> define un split con f_index y threshold
    """
    def __init__(self, f_index=None, threshold=None, value=None):
        self.f_index = f_index  # Índice del atributo
        self.threshold = threshold  # Umbral
        self.value = value  # Valor medio (si es hoja)
        self.left = None  # Hijo izquierdo
        self.right = None  # Hijo derecho


class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.raiz = None  

    def _region_error(self, y):
        """Devuelve la varianza de y o 0 si está vacío."""
        return np.var(y) if len(y) > 0 else 0
        
    def _best_split(self, X, y):
        """Busca el mejor atributo y umbral para dividir los datos."""
        _, n_features = X.shape
        best_feature, best_threshold = None, None
        best_error = float("inf")

        for f in range(n_features):
            values = np.sort(np.unique(X[:, f]))
            if len(values) < 2:
                continue
            # Puntos medios entre valores consecutivos
            thresholds = (values[:-1] + values[1:]) / 2 
            for t in thresholds:
                left_idx = X[:, f] <= t
                right_idx = ~left_idx
                y_left, y_right = y[left_idx], y[right_idx]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                n = len(y)
                error = (len(y_left) / n) * self._region_error(y_left) + \
                        (len(y_right) / n) * self._region_error(y_right)
                if error < best_error:
                    best_error = error
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def _construir_nivel(self, X, y, profundidad):
        """Construye recursivamente el árbol."""
        # Caso base: profundidad agotada o pocas muestras
        if profundidad == 0 or len(y) <= self.min_samples_split:
            return Nodo(value=np.mean(y))

        # Buscar mejor división
        f_opt, t_opt = self._best_split(X, y)
        if f_opt is None:
            return Nodo(value=np.mean(y))

        nodo = Nodo(f_index=f_opt, threshold=t_opt)
        left_idx = X[:, f_opt] <= t_opt
        right_idx = ~left_idx

        nodo.left = self._construir_nivel(X[left_idx], y[left_idx], profundidad - 1)
        nodo.right = self._construir_nivel(X[right_idx], y[right_idx], profundidad - 1)

        return nodo

    def fit(self, X, y):
        """Entrena el árbol y define el nodo raíz."""
        self.raiz = self._construir_nivel(np.array(X), np.array(y), self.max_depth)

    def _predict_sample(self, x, nodo):
        """Predice una sola muestra recorriendo el árbol."""
        if nodo.value is not None:
            return nodo.value
        if x[nodo.f_index] <= nodo.threshold:
            return self._predict_sample(x, nodo.left)
        else:
            return self._predict_sample(x, nodo.right)

    def predict(self, X):
        """Predice todas las muestras de X."""
        X = np.array(X)
        return np.array([self._predict_sample(x, self.raiz) for x in X])

    def decision_path(self, x):
        """Devuelve un string con las decisiones seguidas hasta la predicción."""
        pasos = []
        nodo = self.raiz
        i = 1

        while nodo.value is None:
            if x[nodo.f_index] <= nodo.threshold:
                pasos.append(f"{i}. Atributo {nodo.f_index} menor que {nodo.threshold}")
                nodo = nodo.left
            else:
                pasos.append(f"{i}. Atributo {nodo.f_index} mayor que {nodo.threshold}")
                nodo = nodo.right
            i += 1

        pasos.append(f"Predicción final = {nodo.value:.3f}")
        return "\n".join(pasos)
