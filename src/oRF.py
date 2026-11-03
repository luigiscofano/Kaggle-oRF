import numpy as np
from scipy.stats import mode
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

class Node():
    def __init__(self, 
                 feature_i_star=None, 
                 th_star=None, 
                 left=None, 
                 right=None, 
                 label=None):
        
        # for decision node
        self.feature_i_star = feature_i_star
        self.th_star = th_star
        self.left = left
        self.right = right
        
        # y_hat for leaf nodes
        self.label = label

class DecisionTreeClassifier():
    def __init__(self, max_depth=100):
        self.root = None
        self.max_depth = max_depth
        
    def build_tree(self, X, Y, depth=0):

        if len(np.unique(Y)) == 1: #StopCriteria
            return  Node(label=Y[0])
        
        if depth >= self.max_depth:
            return  Node(label=self.calculate_leaf_label(Y))
        
        if len(Y) < 10:
            return Node(label=self.calculate_leaf_label(Y))

        n, m = X.shape

        feature_i_star, th_star = self.get_best_split(X, Y) 

        if feature_i_star == -1:
             return Node(label=self.calculate_leaf_label(Y))

        #***Implemente*** o split dos dados para enviar para os filhos
        Xi_left = []
        Yi_left = []
        Xi_right = []
        Yi_right = []
        for i in range(n):
            if X[i][feature_i_star] > th_star:
                Xi_right.append(X[i, :])
                Yi_right.append(Y[i]) 
            else:
                Xi_left.append(X[i, :])
                Yi_left.append(Y[i])
        
        Xi_left = np.array(Xi_left) #transformando em arrays numpy pra aceitar o método shape durante a recursão
        Xi_right = np.array(Xi_right)
        Yi_right = np.array(Yi_right)
        Yi_left = np.array(Yi_left)
        
        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left, depth+1)
        right_subtree = self.build_tree(Xi_right, Yi_right, depth+1)
        
        return Node(feature_i_star, th_star, 
                    left_subtree, right_subtree)
    
    def get_best_split(self, X, Y):
            i_star, th_star = -1, -1
            max_info_gain = -float("inf")
            
            m = X.shape[-1]
            n = X.shape[0]
            for feature_i in range(m):
                feature_values = X[:, feature_i] #salvando todos os valores de uma feature num array
                
                #***Implemente*** os valores de limiares
                thresholds = np.unique(feature_values)
                
                for th in thresholds:
                                          
                        #***Implemente*** o split dos dados
                        Xi_right = []
                        Yi_right = []
                        Xi_left = []
                        Yi_left = []
                        for i in range(n):
                            if X[i][feature_i] > th:
                                Xi_right.append(X[i, :])
                                Yi_right.append(Y[i])
                            
                            else:
                                Xi_left.append(X[i, :])
                                Yi_left.append(Y[i])
                                
                        Yi_left = np.array(Yi_left)
                        Yi_right = np.array(Yi_right)
                        if len(Xi_left)>0 and len(Xi_right)>0:
                            curr_info_gain = self.information_gain(Y, Yi_left, Yi_right)

                            if curr_info_gain>max_info_gain:
                                i_star = feature_i
                                th_star = th
                                max_info_gain = curr_info_gain
                            
            return i_star, th_star
    
    def entropy_calc(self, y):
         values, counts = np.unique(y, return_counts=True)
         p = counts / counts.sum()
         return entropy(p, base=2)

    def information_gain(self, parent, l_child, r_child):
           #***Implemente*** a função de ganho de informação 
        eta = len(l_child)/len(parent)
        return self.entropy_calc(parent) - (eta*self.entropy_calc(l_child) + (1-eta)*self.entropy_calc(r_child))
    
        
    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat  = self.make_prediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def make_prediction(self, x, tree):
        
        if tree.label is not None: #Leaf
            return tree.label
        
        feature_val = x[tree.feature_i_star]
        if feature_val<=tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
class RF ():

    def __init__ (self, k=10, p=0.6, s=0.1):
        self.k = k   # number of trees
        self.p = p # Subsets of 60% data (bagging)
        self.s = s #Subsets of s% features (feature subsampling)
        self.trees = []
        self.features_list = []

    #Training phase
    def fit(self, X_train, Y_train):

        self.trees = []
        self.features_list = []
        _, m = X_train.shape

        for _ in range(self.k):
            # bagging
            idx = np.random.choice(len(X_train), int(self.p * len(X_train)), replace=True)
            
            # feature selection
            feat_idx = np.random.choice(m, int(m*self.s), replace=False)
            
            X_sub = X_train[idx][:, feat_idx]
            y_sub = Y_train[idx]
            
            tree = DecisionTreeClassifier()
            tree.fit(X_sub, y_sub)
            
            self.trees.append(tree)
            self.features_list.append(feat_idx)

    #Testing phase
    def predict(self, X_test):
        preds = []
        for tree, feat_idx in zip(self.trees, self.features_list):
                y_hat = tree.predict(X_test[:, feat_idx])
                preds.append(y_hat)

        preds = np.array(preds)
        final_pred = mode(preds, axis=0, keepdims=False).mode
        return final_pred

class ObliqueNode():
    def __init__(self, 
                 w=None, 
                 mean=None,
                 std=None,
                 th_star=None, 
                 left=None, 
                 right=None, 
                 label=None):
        
        # for decision node
        self.w = w
        self.mean = mean
        self.std = std
        self.th_star = th_star
        self.left = left
        self.right = right
        
        # y_hat for leaf nodes
        self.label = label

class ObliqueDecisionTreeClassifier():
    def __init__(self, max_depth=10):
        self.root = None
        self.max_depth = max_depth
        
    def build_tree(self, X, Y, depth=0):

        if len(np.unique(Y)) == 1: #StopCriteria
            return  ObliqueNode(label=Y[0])
        
        if depth >= self.max_depth:
            return  ObliqueNode(label=self.calculate_leaf_label(Y))
        
        if len(Y) < 10:
            return  ObliqueNode(label=self.calculate_leaf_label(Y))

        n, m = X.shape

        w_star, mean, std, th_star = self.get_best_split(X, Y) 

        if w_star is None:
            return ObliqueNode(label=self.calculate_leaf_label(Y))

        #***Implemente*** o split dos dados para enviar para os filhos
        Xi_left = []
        Yi_left = []
        Xi_right = []
        Yi_right = []

        X_norm = (X-mean)/std
        Z = X_norm @ w_star

        for i in range(n):
            if Z[i] > th_star:
                Xi_right.append(X[i, :])
                Yi_right.append(Y[i]) 
            else:
                Xi_left.append(X[i, :])
                Yi_left.append(Y[i])
        
        Xi_left = np.array(Xi_left) #transformando em arrays numpy pra aceitar o método shape durante a recursão
        Xi_right = np.array(Xi_right)
        Yi_right = np.array(Yi_right)
        Yi_left = np.array(Yi_left)
        
        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return ObliqueNode(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left, depth+1)
        right_subtree = self.build_tree(Xi_right, Yi_right, depth+1)
        
        return ObliqueNode(w_star, mean, std, th_star, 
                    left_subtree, right_subtree)
    
    def get_best_split(self, X, Y):
            th_star = -1
            w_star = None
            max_info_gain = -float("inf")
            #sum_var = 0
            #num_componentes = 0
            
            #PCA
            mean = X.mean(axis=0)
            std = np.std(X, axis=0)
            std[std==0]=1 #correção de divisão por 0
            X_norm = (X - mean)/std #normalização Z-score
            U, S, Vt = np.linalg.svd(X_norm, full_matrices = False)
            #w = Vt[0] #guardando vetor do hiperplano
           
            #projetando X no meu espaço latente
            for j in range(min(3,Vt.shape[0])): #pegar as 5 Componentes Principais e ver qual faz o melhor corte
                Z = X_norm @ Vt[j]
                n = Z.shape[0]

                    #***Implemente*** os valores de limiares
                thresholds = np.unique(Z)
                    
                for th in thresholds:
                                            
                            #***Implemente*** o split dos dados
                    Xi_right = []
                    Yi_right = []
                    Xi_left = []
                    Yi_left = []
                    for i in range(n):
                        if Z[i] > th:
                            Xi_right.append(X[i, :])
                            Yi_right.append(Y[i])
                                
                        else:
                            Xi_left.append(X[i, :])
                            Yi_left.append(Y[i])
                                    
                    Yi_left = np.array(Yi_left)
                    Yi_right = np.array(Yi_right)
                    if len(Xi_left)>0 and len(Xi_right)>0:
                        curr_info_gain = self.information_gain(Y, Yi_left, Yi_right)

                        if curr_info_gain>max_info_gain:
                            w_star = Vt[j]
                            th_star = th
                            max_info_gain = curr_info_gain
                            
            return w_star, mean, std, th_star
    
    def entropy_calc(self, y):
         values, counts = np.unique(y, return_counts=True)
         p = counts / counts.sum()
         return entropy(p, base=2)

    def information_gain(self, parent, l_child, r_child):
           #***Implemente*** a função de ganho de informação 
        eta = len(l_child)/len(parent)
        return self.entropy_calc(parent) - (eta*self.entropy_calc(l_child) + (1-eta)*self.entropy_calc(r_child))
    
        
    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat  = self.make_prediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def make_prediction(self, x, tree):
        
        if tree.label is not None: #Leaf
            return tree.label
        
        x_norm = (x - tree.mean)/tree.std
        z = x_norm @ tree.w
        if z <=tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

class oRF ():

    def __init__ (self, k=250, p=0.63, s=0.27):
        self.k = k   # number of trees
        self.p = p # Subsets of 60% data (bagging)
        self.s = s #Subsets of s% features (feature subsampling)
        self.trees = []
        self.features_list = []

    #Training phase
    def fit(self, X_train, Y_train):

        self.trees = []
        self.features_list = []
        _, m = X_train.shape

        for _ in range(self.k):
            # bagging
            idx = np.random.choice(len(X_train), int(self.p * len(X_train)), replace=True)
            
            # feature selection
            feat_idx = np.random.choice(m, int(m*self.s), replace=False)
            
            X_sub = X_train[idx][:, feat_idx]
            y_sub = Y_train[idx]
            
            tree = ObliqueDecisionTreeClassifier()
            tree.fit(X_sub, y_sub)
            
            self.trees.append(tree)
            self.features_list.append(feat_idx)

    #Testing phase
    def predict(self, X_test):
        preds = []
        for tree, feat_idx in zip(self.trees, self.features_list):
                y_hat = tree.predict(X_test[:, feat_idx])
                preds.append(y_hat)

        preds = np.array(preds)
        final_pred = mode(preds, axis=0, keepdims=False).mode
        return final_pred
    
def main():

    # Gera um conjunto de classificação
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # Treina a Random Forest
    model = oRF()
    model.fit(X_train, y_train)

    # Predição
    y_hat = model.predict(X_test)

    # Avaliação
    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()