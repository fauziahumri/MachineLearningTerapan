# -*- coding: utf-8 -*-
"""MLT_Submission 1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11NFFjA8CpdxtYMBL755f9LyTPpBPl3h-

# **Prediksi Analis : Wine Quality Prediction**

# **Pendahuluan**

Topik yang saya bahas di proyek ini yaitu mengenai bidang ekonomi dan bisnis yang dibuat untuk mengetahui prediksi kualitas dari Wine/Anggur merah.

**Memasukkan library yang dibutuhkan**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

"""**Menyipkan Dataset yang akan diprediksi dengan menggunakan Kredensial Kaggle**"""

# Membuat folder .kaggle di dalam folder root
!rm -rf ~/.kaggle && mkdir ~/.kaggle/

# Menyalin berkas kaggle.json pada direktori aktif saat ini ke folder .kaggle
!mv kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

# Mengunduh dataset menggunakan Kaggle CLI
!kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009

# Mengekstrak berkas zip ke direktori aktif saat ini
!unzip /content/red-wine-quality-cortez-et-al-2009.zip

"""**Data Understanding**

Membuat data pada DataFrame dengan menggunakan Pandas
"""

wine = pd.read_csv('/content/winequality-red.csv')
wine.head()

"""**Exploratory Data Analysis (EDA)**


Explanatory Data Analysis (EDA) merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

Deskripsi Variabel
"""

wine.shape

wine.info()

"""Mengubah data numerik menjadi data kategori agar mempermudah analisa selanjutnya"""

# Variabel Dependent --> Dalam hal ini variabel dependent merupakan variabel target 
wine.quality.replace(3,'easy', inplace=True)
wine.quality.replace(4,'easy', inplace=True)
wine.quality.replace(5,'medium', inplace=True)
wine.quality.replace(6,'high', inplace=True)
wine.quality.replace(7,'high', inplace=True)
wine.quality.replace(8,'very high', inplace=True)

"""Melihat informasi yang telah dimuat pada dataset"""

def report(wine):
  col = []
  d_type = []
  uniques = []
  n_uniques = []

  for i in wine.columns:
    col.append(i) 
    d_type.append(wine[i].dtypes) 
    uniques.append(wine[i].unique()[:5]) 
    n_uniques.append(wine[i].nunique()) 

  return pd.DataFrame({'Column': col, 'd_type':d_type, 'unique_sample':uniques, 'n_unique_sample':n_uniques})

report(wine)

"""Deskripsi statistik"""

wine.describe()

"""Melakukan penghapusan data dengan nilai yang sama"""

print('Jumlah data sebelum dihapus :', wine.shape[0])
wine = wine.drop_duplicates()
print('Jumlah data setelah dihapus :', wine.shape[0])

"""Memperbaiki missing value"""

wine.isnull().sum()

"""Memeriksa kembali apakah ada data yang sama atau tidak"""

wine.duplicated().sum()

"""Visualisasi data yang kosong"""

sorted_null = msno.nullity_sort(wine, sort='descending')
figures = msno.matrix(sorted_null, color=(1, 0.42, 0.42))

"""Seperti yang terlihat pada data diatas bahwa tidak ada misiing value dan data yang sama

# **Analisa Unvariat**

Membagi fitur pada dataset menjadi 2 bagian yaitu *Numerical* dan *Categorical*
"""

categorical_features = wine.loc[:, wine.dtypes == 'object'].columns.to_list()
numerical_features = wine.loc[:, wine.dtypes != 'object'].columns.to_list()

"""Melakukan analisa pada *Categorical features*"""

categorical_features = ['quality']

for column in categorical_features:
  count = wine[column].value_counts()
  percent = 100*wine[column].value_counts(normalize=True)
  new_data = pd.DataFrame({'Jumlah':count, 'Persentase':percent.round(1)})
  print(new_data, end='\n\n')
  count.plot(kind='bar', title=column)
  plt.show()

"""Melakukan analisa pada *Numerical* features dan melihat *outlier*."""

fig, axs = plt.subplots(len(numerical_features), figsize= (15,10))
i=0
for feature in numerical_features:
  sns.boxplot(wine[feature], ax=axs[i])
  i+=1
  plt.tight_layout()
plt.show()

wine.hist(bins=50, figsize=(15,10))
plt.show()

"""# **Analisa Multivariat**"""

sns.pairplot(wine, diag_kind = 'kde', hue='quality')

"""Melakukan pengecekan korelasi"""

plt.figure(figsize=(15,15))
plt.title('Collerasi Matrix untuk Fitur Numerik', fontsize=9)
sns.heatmap(wine.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

"""# **Encoding Fitur Kategori (Categorical Encoding)**

Categorical Encoding adalah proses mengubah data kategori menjadi data numerik
"""

wine = pd.get_dummies(wine, columns=wine.loc[:, (wine.dtypes == 'object') & (wine.columns != 'quality')].columns.to_list())

"""Encoding manual dengan memasukan varibel dependet sebagai variabel target"""

wine.quality.replace('easy', 3, inplace=True)
wine.quality.replace('easy', 4, inplace=True)
wine.quality.replace('medium', 5, inplace=True)
wine.quality.replace('high', 6, inplace=True)
wine.quality.replace('high', 7, inplace=True)
wine.quality.replace('very high', 8, inplace=True)

wine

"""# ***Train-Test-Split***

Pemisahan dataset menjadi 2 bagian yaitu data latih dan data tes
"""

X = wine.drop('quality', axis=1).values
y = wine['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify=y)

print(f'Jumlah seluruh sampel: {len(X)}')
print(f'Jumlah data train: {len(X_train)}')
print(f'Jumlah data test: {len(X_test)}')

"""Standarisasi nilai pada numeric features dengan StandardScaler"""

print(X_train[0:2])
print('')
print(X_test[0:2])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print(X_train[0:2])
print("")
print(X_test[0:2])

"""# **Pembuatan Model**

Pada tahap ini pembuatan model dilakukan menggunakan * Support Vector Machine* (SVM) dan *Hyper Parameter Tuning* dari SVM.

Model *baseline* dengan SVM
"""

# Pembuatan model
baseline_model = SVC()

# Melakukan training
baseline_model.fit(X_train, y_train)

# melakukan ujicoba pada data test
y_pred = baseline_model.predict(X_test)

# report klasifikasi untuk model baseline
cr_baseline = classification_report(y_test, y_pred, output_dict=True, target_names=['easy', 'medium', 'high', 'very high'])
pd.DataFrame(cr_baseline)

# confusion matrix model baseline
cf_baseline = confusion_matrix(y_test, y_pred)

"""Pengembangan Model SVM menggunakan Hyper Parameter Tuning dengan GridSearchCV"""

# hyperparameter yang akan di tuning
parameters = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'] 
}

# Menetapkan StratifiedKFold
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Pembuatan model untuk GridSearchCV
grid = GridSearchCV(baseline_model, parameters, cv=skf, verbose=1, scoring='accuracy', n_jobs=-1)

# Melakukan Training
grid.fit(X_train,y_train)

print("Best parameter: ", grid.best_estimator_)
print("Score: ", grid.best_score_)

# Pembuatan model best parameter
best_model = grid.best_estimator_

# Melakukan training
best_model.fit(X_train, y_train)

# Pengujian model terhadap data test
y_pred = best_model.predict(X_test)

# classification report model best parameter
cr_best = classification_report(y_test, y_pred, output_dict=True, target_names=['easy', 'medium', 'high', 'very high'])
pd.DataFrame(cr_best)

# confusion matrix untuk model best parameter
cf_best = confusion_matrix(y_test, y_pred)

"""# **Evaluasi Model**"""

print("Classification Report untuk Model Baseline")
pd.DataFrame(cr_baseline)

print("Classification Report Model untuk Parameter Terbaik")
pd.DataFrame(cr_best)

# Visualisasi hasil prediksi model baseline
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(cf_baseline, annot=True, fmt='g')

ax.set_yticklabels(['easy', 'medium', 'high', 'very high'])
ax.set_xticklabels(['easy', 'medium', 'high', 'very high'])

ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

ax.set_title('Confusion Matriks untuk Model Baseline', fontweight='bold')
plt.show()

# Visualisasi hasil prediksi model untuk best parameter
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(cf_best, annot=True, fmt='g')

ax.set_yticklabels(['easy', 'medium', 'high', 'very high'])
ax.set_xticklabels(['easy', 'medium', 'high', 'very high'])

ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

ax.set_title('Confusion Matrik Model Best Parameter', fontweight='bold')
plt.show()