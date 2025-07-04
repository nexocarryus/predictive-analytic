# -*- coding: utf-8 -*-
"""predictiveAnalytic_Naufal Dzakwan Zakianto.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R_C423qEMJtolqZBF3Co1zTZaJg7qgKe

# Proyek predictive analytic

- **Nama:** Naufal Dzakwan Zakianto
- **Email:** naufal.dzakwann28@gmail.com
- **ID Dicoding:** MC012D5Y2416

# Problem statement

Saat ini properti terutama rumah adalah kebutuhan yang dibutuhkan oleh semua orang, namun harga rumah yang bervariasi tentunya menjadi kesulitan bagi pembeli maupun penjual dalam menentukan harga yang ideal, memprediksi harga perumahan secara akurat juga sangat penting untuk investasi real estate, pembangunan perkotaan, dan pengambilan keputusan konsumen. Maka dari itu, proyek ini bertujuan untuk membangun model regresi yang dapat memperkirakan harga jual rumah berdasarkan berbagai fitur terkait properti dan lokasi.

# Data understanding

Dataset yang digunakan bersumber dari situs kaggle yang dapat di akses di link berikut: https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

Data tersebut berisi 500+ baris data yang terdiri dari rincian kolom sebagai berikut:

- Price: Harga rumah.
- Area: Total luas rumah dalam kaki persegi.
- Bedroom: Jumlah kamar tidur di rumah.
- Bathroom: Jumlah kamar mandi di rumah.
- Stories: Jumlah lantai di rumah.
- Main road: Apakah rumah terhubung ke jalan utama (Ya/Tidak).
- Guestroom: Apakah rumah memiliki kamar tamu (Ya/Tidak).
- Basement: Apakah rumah memiliki ruang bawah tanah (Ya/Tidak).
- Hot water heating: Apakah rumah memiliki sistem pemanas air panas (Ya/Tidak).
- Airconditioning: Apakah rumah memiliki sistem pendingin udara (Ya/Tidak).
- Parking: Jumlah tempat parkir yang tersedia di dalam rumah.
- Prefarea: Apakah rumah berada di area yang disukai (Ya/Tidak).
- Furnishing status: Status perabotan rumah (Sepenuhnya berperabotan, Semi-perabotan, Tidak berperabotan).

# Import Library
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import kagglehub
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

"""# Data loading"""

path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
print("Path to dataset files:", path)

house = pd.read_csv(path+"/Housing.csv")

house.head()

"""# Exploratory data analysis

Memeriksa deskripsi variabel dan mising value
"""

house.info()

house.describe(include = 'all')

"""**INSIGHT:**

Tidak ditemukan ke anehan dalam deskripsi variabel, semua nilai yang ada pada variabel terlihat wajar, tidak ada error value serta tidak ditemukan missing value.

Memeriksa outlier
"""

# Fungsi deteksi outlier IQR method
def detect_outliers_iqr(house):
    outliers = pd.DataFrame()
    for column in house.select_dtypes(include=['int64']).columns:
        Q1 = house[column].quantile(0.25)
        Q3 = house[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_in_column = house[(house[column] < lower_bound) | (house[column] > upper_bound)]
        outliers = pd.concat([outliers, outliers_in_column])
    return outliers

# Deteksi outlier
outliers = detect_outliers_iqr(house)

# Visualisasi outlier
for column in house.select_dtypes(include=['int64']).columns:
    plt.figure(figsize=(10, 5))
    plt.boxplot(house[column], vert=False)
    plt.title(f'Box plot of {column}')
    plt.xlabel(column)
    plt.show()

"""INSIGHT:

Karena jumlah outliernya tidak terlalu banyak, dan dikhawatirkan outlier ini nantinya akan mengganggu proses pelatihan pada model, maka data yang mengandung outlier nantinya bisa dihapus menggunakan IQR method.

Univariate analysis fitur categorical
"""

# Visualisasi presentase dari masing masing fitur categori
categorical_features = house.select_dtypes(include=['object']).columns

for feature in categorical_features:
    count = house[feature].value_counts()
    percent = 100 * house[feature].value_counts(normalize=True)
    dfc = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    count.plot(kind='bar', title=feature)

    # Anotasi presentase ke barchart
    for idx, value in enumerate(count):
        plt.text(idx, value, f'{percent[idx]:.1f}%', ha='center', va='bottom')

    plt.xlabel(feature)
    plt.ylabel('Jumlah Sampel')
    plt.show()

    # Print dataframe dengan jumlah dan presentase
    print(dfc)

"""INSIGHT:

Berdasarkan visualisasi, rumah yang ada pada dataset ini cenderung banyak yang berada di jalan utama. Meskipun demikian rumah-rumah tersebut didominasi oleh rumah yang tidak memiliki kamar tamu, basement, penghangat air, pendingin ruangan, serta area yang bukan favorit pembeli. Selain itu terdapat lebih banyak rumah yang semi furnished dan unfurnised dibandingkan full furnished.

Univariate analysis fitur numeric
"""

# Visualisasi histogram dari masing masing fitur numeric
numerical_features = house.select_dtypes(include=['int64']).columns

for feature in numerical_features:
    plt.figure(figsize=(10, 5))
    plt.hist(house[feature], bins=30, edgecolor='k')
    plt.title(f'Histogram {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

"""INSIGHT:

Berdasarkan histogram dapat diketahui bahwa kebanyakan rumah yang ada memiliki 3 kamar tidur, 1 kamar mandi, 1 lantai dan tidak punya tempat parkir. Selain itu bisa dilihat juga bahwa lebih banyak rumah yang memiliki luas area yang kecil sampai menangah yaitu berpusat di sekitar 3000-4000. Faktor-faktor tersebut mengakibatkan  harga rumah lebih banyak berada harga yang rendah, yaitu range 3 juta sampai 4 juta, (positif skewness) hal ini tentunya akan berimplikasi ke model nantinya.

Multivariate analysis fitur categorical
"""

cat_features = house.select_dtypes(include='object').columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="price", kind="bar", dodge=False, height = 4, aspect = 3,  data=house, palette="Set3")
  plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))

"""INSIGHT:

Dapat diketahui secara jelas bahwa rumah dengan posisi yang berada di jalan utama, memiliki kamar tamu, memiliki basement, pemanas air, pendingin ruangan,  berada di area yang diinginkan, dan full furnished konsisten memiliki harga yang lebih tinggi dibandingkan yang tidak memiliki faktor-faktor tersebut.

Multivariate analysis fitur numeric
"""

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(house, diag_kind = 'kde')

"""INSIGHT:

terlihat jelas hubungan antara harga dengan area konsisten meningkat secara linear, sedangkan dengan fitur fitur lainnya cenderung diskrit, ini bisa dilihat dari visualisasi garis-garis vertikal karena banyak data menumpuk di titik x yang sama. Selain itu meskipun outlier sudah di atasi, distribusi price masih terlihat tidak normal, mencuat tinggi di kisaran 8 juta dan menurun tajam ke kanan. Artinya masih ada sebagian kecil rumah dengan harga sangat tinggi.

Beberapa algoritma seperti linear regression bisa terlalu terdorong oleh outlier rumah yang terlalu mahal dan dapat disimpulkan bahwa perlu adanya transformasi agar mendekati distribusi normal pada kolom target. Selain itu kondisi fitur lainnya yang diskrit membuat terkadang model linier kesulitan kalau tidak ada variasi cukup, atau jika makna angka tidak linier (misal, selisih antara 1 dan 2 parkir tidak sama dampaknya dengan 3 dan 4). Dari kondisi tersebut model seperti tree based cenderung lebih cocok seperti random forest maupun adaboost. Meskipun demikian, linear regression akan tetap dicoba dengan proses scaling dan akan dibandingkan hasilnya dengan model random forest serta adaboost. Selain itu, berdasarkan analisis yang dilakukan pada univariate maupun multivariate analysis, dapat diketahui bahwa dataset yang digunakan cenderung memiliki harga rumah yang rendah dan sangat sedikit rumah yang memiliki harga sangat tinggi (outlier), hal ini dapat disimpulkan bahwa MAE lebih cocok digunakan untuk matriks evaluasinya, sebab cenderung lebih adil dan dapat menunjukan apakah model ini lebih baik secara rata-rata umum atau tidak.
"""

plt.figure(figsize=(10, 8))
correlation_matrix = house[numerical_features].corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""INSIGHT:

Semua fitur numeric yang ada terlihat memiliki kontribusi atau korelasi terhadap target, meskipun tidak ada yang sangat dominan korelasinya, namun tidak ada yang berada dibawah 0. dan apabila dilihat dari visualisasi barchart sebelumnya pada multivariate analysis fitur categorical bisa dilihat juga bahwa fitur fitur tersebut selaras dengan harga, semakin lengkap fasilitas terpenuhi, semakin tinggi juga harga rumahnya. Maka dapat diputuskan bahwa semua fitur akan digunakan untuk melatih model kecuali kolom price (target)

# Data preparation

Menghapus outlier
"""

# Ambil hanya kolom numerikal
numeric_cols = house.select_dtypes(include='number').columns
# Hitung Q1, Q3, dan IQR hanya untuk kolom numerikal
Q1 = house[numeric_cols].quantile(0.25)
Q3 = house[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
# Buat filter untuk menghapus baris yang mengandung outlier di kolom numerikal
filter_outliers = ~((house[numeric_cols] < (Q1 - 1.5 * IQR)) |
                    (house[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
# Terapkan filter ke dataset asli (termasuk kolom non-numerikal)
house = house[filter_outliers]
# Cek ukuran dataset setelah outlier dihapus
house.shape

"""Encoding fitur categorical"""

house = pd.concat([house, pd.get_dummies(house['mainroad'], prefix='mainroad')],axis=1)
house = pd.concat([house, pd.get_dummies(house['guestroom'], prefix='guestroom')],axis=1)
house = pd.concat([house, pd.get_dummies(house['basement'], prefix='basement')],axis=1)
house = pd.concat([house, pd.get_dummies(house['hotwaterheating'], prefix='hotwaterheating')],axis=1)
house = pd.concat([house, pd.get_dummies(house['airconditioning'], prefix='airconditioning')],axis=1)
house = pd.concat([house, pd.get_dummies(house['prefarea'], prefix='prefarea')],axis=1)
house = pd.concat([house, pd.get_dummies(house['furnishingstatus'], prefix='furnishingstatus')],axis=1)

house.drop(['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], axis=1, inplace=True)

house.head()

for col in house.columns:
    if house[col].dtype == 'bool':
        house[col] = house[col].astype(int)

house.info()

"""Split dataset"""

X = house.drop(["price"],axis =1)
y = house["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""Standarisasi data latih"""

numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

y_train_log = np.log1p(y_train)

"""# Model development"""

models = pd.DataFrame(index=['train_mae','test_mae'],
                      columns=['LinearRegression', 'RandomForest', 'AdaBoost'])

"""Logistic Regression"""

LR = LinearRegression()
LR.fit(X_train, y_train_log)
y_pred_log = LR.predict(X_train)
y_pred = np.expm1(y_pred_log)
mae = mean_absolute_error(y_train, y_pred)

models.loc['train_mae','LinearRegression'] = mae

"""Random forest dengan hyperparameter tunning"""

base_model = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train_log)

best_rf = random_search.best_estimator_

y_pred_log = best_rf.predict(X_train)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_train, y_pred)

models.loc['train_mae', 'RandomForest'] = mae

print("Best Parameters for Random Forest:", random_search.best_params_)

"""AdaBoost dengan hyperparameter tunning"""

base_model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(),
    random_state=42
)


param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3, 5, 7],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
}


random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train_log)

best_adaboost = random_search.best_estimator_

y_pred_log = best_adaboost.predict(X_train)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_train, y_pred)

models.loc['train_mae', 'AdaBoost_Tuned'] = mae

print("Best Parameters for AdaBoost:", random_search.best_params_)

"""# Evaluasi"""

X_test[numerical_features] = scaler.transform(X_test[numerical_features])

models = pd.DataFrame(index=[ 'train_mae', 'test_mae'],
                      columns=['LinearRegression', 'RandomForest', 'AdaBoost'])

model_dict = {
    'LinearRegression': LR,
    'RandomForest': best_rf,
    'AdaBoost': best_adaboost
}

for name, model in model_dict.items():

    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_test = np.expm1(y_pred_test_log)

    models.loc['train_mae', name] = mean_absolute_error(y_train, y_pred_train)
    models.loc['test_mae', name] = mean_absolute_error(y_test, y_pred_test)

print(models)

# Visualisasi train vs test MAE
fig, ax = plt.subplots(figsize=(8,4))
models.loc[['train_mae', 'test_mae']].T.plot(kind='barh', ax=ax)
ax.set_title("Train vs Test MAE")
ax.set_xlabel("MAE")
ax.grid(True)
plt.tight_layout()
plt.show()

"""INSIGHT:

Berdasarkan hasil tersebut, model AdaBoost ternyata merupakan model terbaik dengan nilai mean absolut error (MAE) pada proses training sebesar 433662.30811 dan testing sebesar 548786.736431, nilai ini lebih rendah dibandingkan dua model lainnya, baik pada saat proses training maupun proses testing. Model random forest memiliki nilai MAE sebesar 538718.73195 pada training dan 596380.0920 pada testing, sedangkan model linear regression memiliki nilai MAE sebesar 685947.065865 pada training dan 622031.295643 pada testing. Sehingga dapat diketahui bahwa model AdaBoost mampu memprediksi harga rumah dengan error rata rata secara umum yang paling baik dibandingkan random forest maupun linear regression, serta model Adaboost inilah yang akan dipilih untuk digunakan dalam memprediksi harga rumah.

Pengujian beberapa harga pada datsaset
"""

# Ambil satu data untuk membandingkan hasil prediksi
prediksi = X_test.iloc[:5].copy()
y_true_subset = y_test.iloc[:5]

# Buat dictionary untuk menyimpan hasil prediksi
pred_dict = {'y_true': y_true_subset.values}

for name, model in model_dict.items():
    y_pred_log = model.predict(prediksi)
    y_pred = np.expm1(y_pred_log)
    pred_dict['prediksi_' + name] = np.round(y_pred, 1)

# Tampilkan sebagai DataFrame
pd.DataFrame(pred_dict)

"""Terbukti, dari 5 harga yang ada pada dataset,  model AdaBoost berhasil memprediksi 3 harga asli dengan selisih paling mendekati dibandingkan dengan model lainnya."""