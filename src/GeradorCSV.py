import numpy as np
import pandas as pd
from oRF import oRF

# Carrega a base de dados a partir de seu caminho
data = np.load('data/data.npz')
X_test = data['X_test']
X_train = data['X_train']
y_train = data['y_train']
num_samples = X_test.shape[0]

# oRF
model = oRF()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Gera um DataFrame no formato esperado da submissão
submission_df = pd.DataFrame({
    'ID': np.arange(1, num_samples + 1),
    'Prediction': y_hat
})

# Salva o arquivo CSV no diretório atual
submission_df.to_csv('submissions/submission.csv', index=False)
