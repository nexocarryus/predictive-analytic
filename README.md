
# Laporan Proyek Machine Learning - Naufal Dzakwan Zakianto

## Domain Proyek

Properti merupakan kebutuhan primer setiap individu, khususnya rumah sebagai tempat tinggal. Dalam praktiknya, harga rumah sangat bervariasi dan bergantung pada banyak faktor seperti lokasi, luas bangunan, dan fasilitas. Variasi harga ini menimbulkan tantangan bagi pembeli dan penjual untuk menentukan harga yang ideal. 

Memprediksi harga rumah secara akurat memiliki nilai penting dalam berbagai aspek, termasuk investasi properti, hingga keputusan pembelian individu. Oleh karena itu, pendekatan berbasis machine learning menjadi solusi potensial dalam mengotomatisasi dan meningkatkan akurasi prediksi harga rumah.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga rumah dengan harga yang baik berdasarkan fitur-fitur yang tersedia?
2. Fitur apa saja yang berpengaruh terhadap prediksi harga rumah pada dataset yang digunakan?

### Goals

1. Membangun model regresi yang mampu memprediksi harga rumah berdasarkan data.
2. Menentukan fitur-fitur yang berkontribusi terhadap harga rumah pada dataset.

### Solution Statements

- Menggunakan algoritma regresi seperti Linear Regression, Random Forest Regressor, dan AdaBoost Regressor.
- Melakukan hyperparameter tunning untuk meningkatkan performa model.
- Melakukan exploratory data analysis untuk melihat kondisi data dan fitur penting yang berkontribusi terhadap harga rumah.
- Mengukur performa model menggunakan metrik **Mean Absolute Error (MAE)**.

## Data Understanding
Dataset yang digunakan bersumber dari situs kaggle yang dapat di akses di link berikut: https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

Data tersebut berisi 545 baris data dan 12 kolom, yang terdiri dengan rincian kolom sebagai berikut:

- Price: Harga rumah. (Target)
- Area: Total luas rumah.
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

Tahapan EDA (Explatory Data Analysis) juga dilakukan dalam proyek ini, dengan rincian sebagai berikut:

- **Memeriksa deskripsi variabel dan missing value**
  
  ![image](https://github.com/user-attachments/assets/0f972845-2efe-4367-b774-102d260c0eb5)

  ![image](https://github.com/user-attachments/assets/bae39fda-1307-47ce-a0c4-94b2722ac51d)
  
  Berdasarkan pemeriksaan tidak ditemukan ke anehan dalam deskripsi variabel, semua nilai yang ada pada variabel terlihat wajar, tidak ada error value serta tidak ditemukan missing value
  
- **Memeriksa outlier**
  ![image](https://github.com/user-attachments/assets/0877ea32-d5c8-4ea8-bf2f-a057500375cf)

  ![image](https://github.com/user-attachments/assets/c7a857da-bf6e-4d67-a6cf-d17b967d95e8)

  ![image](https://github.com/user-attachments/assets/59660c6f-e4b1-4f37-ad6d-efc4c0d35310)

  Setelah diperiksa, ditemukan beberapa outlier dan karena jumlah outliernya tidak terlalu banyak, dan dikhawatirkan outlier ini nantinya akan mengganggu proses pelatihan pada model, maka data yang mengandung outlier bisa dihapus.

- Univariate analysis fitur categorical
  
  ![image](https://github.com/user-attachments/assets/01ff113a-04f8-426d-a74a-1c5250c298eb)

  ![image](https://github.com/user-attachments/assets/d5e087d7-d005-4fb3-a4b1-b899ad187c21)

  ![image](https://github.com/user-attachments/assets/39a23262-259d-4122-a0c8-969797a8a36c)

  ![image](https://github.com/user-attachments/assets/bec571ca-29f4-445f-a7f1-32de28d82fbd)

  Berdasarkan visualisasi, rumah yang ada pada dataset ini cenderung banyak yang berada di jalan utama. Meskipun demikian rumah-rumah 
  tersebut didominasi oleh rumah yang tidak memiliki kamar tamu, basement, penghangat air, pendingin ruangan, serta area yang bukan 
  favorit pembeli. Selain itu terdapat lebih banyak rumah yang semi furnished dan unfurnised dibandingkan full furnished.

- Univariate analysis fitur numeric

  ![image](https://github.com/user-attachments/assets/399ec3aa-32e0-4edf-9540-8fdb53dcd10f)

  ![image](https://github.com/user-attachments/assets/3f41aae8-61d7-4259-b6dc-63598570337e)

  ![image](https://github.com/user-attachments/assets/3e9d4ed0-3ca5-472f-8314-26dfaeb39578)

  Berdasarkan histogram dapat diketahui bahwa kebanyakan rumah yang ada memiliki 3 kamar tidur, 1 kamar mandi, 1 lantai dan tidak punya 
  tempat parkir. Selain itu bisa dilihat juga bahwa lebih banyak rumah yang memiliki luas area yang kecil sampai menangah yaitu 
  berpusat di sekitar 3000-4000. Faktor-faktor tersebut mengakibatkan harga rumah lebih banyak berada harga yang rendah, yaitu range 3 
  juta sampai 4 juta, (positif skewness) hal ini tentunya akan berimplikasi ke model nantinya.

- Multivariate analysis fitur categorical

  ![image](https://github.com/user-attachments/assets/923cbd44-7483-482a-8fd0-013cc6c31e19)

  ![image](https://github.com/user-attachments/assets/ff921ebf-286a-46d5-b31b-5ccda2201b12)

  ![image](https://github.com/user-attachments/assets/3dcc31fe-0d18-40c2-ade6-3eb913d638ed)

  ![image](https://github.com/user-attachments/assets/aae89140-153d-4b6d-a4f7-d5d1e7f65c62)

  Dari visualisasi dapat diketahui secara jelas bahwa rumah dengan posisi yang berada di jalan utama, memiliki kamar tamu, memiliki 
  basement, pemanas air, pendingin ruangan, berada di area yang diinginkan, dan full furnished konsisten memiliki harga yang lebih 
  tinggi dibandingkan yang tidak memiliki faktor-faktor tersebut. **Sehingga dapat disimpulkan bahwa semua fitur kategorical memiliki kontribusi penting terhadap harga rumah**.

- Multivariate analysis fitur numeric menggunakan pairplot

  ![image](https://github.com/user-attachments/assets/6b817b59-556b-4ae9-8209-005e627b035e)
  ![image](https://github.com/user-attachments/assets/1aa6c1fe-f2b8-4ae8-9cae-07dfb2c7480e)

  terlihat jelas hubungan antara harga dengan area konsisten meningkat secara linear, sedangkan dengan fitur fitur lainnya cenderung diskrit, ini bisa dilihat dari visualisasi garis-garis vertikal karena banyak 
  data menumpuk di titik x yang sama. Selain itu meskipun outlier sudah di atasi, distribusi price masih terlihat tidak normal, mencuat tinggi di kisaran 8 juta dan menurun tajam ke kanan. Artinya masih ada 
  sebagian kecil rumah dengan harga sangat tinggi.

  Beberapa algoritma seperti linear regression bisa terlalu terdorong oleh outlier rumah yang terlalu mahal dan dapat disimpulkan bahwa perlu adanya transformasi agar mendekati distribusi normal pada kolom 
  target. Selain itu kondisi fitur lainnya yang diskrit membuat terkadang model linier kesulitan kalau tidak ada variasi cukup, atau jika makna angka tidak linier (misal, selisih antara 1 dan 2 parkir tidak 
  sama dampaknya dengan 3 dan 4). Dari kondisi tersebut model seperti tree based cenderung lebih cocok seperti random forest maupun adaboost. Meskipun demikian, linear regression akan tetap dicoba dengan proses 
  scaling dan akan dibandingkan hasilnya dengan model random forest serta adaboost. Selain itu, berdasarkan analisis yang dilakukan pada univariate maupun multivariate analysis, dapat diketahui bahwa dataset 
  yang digunakan cenderung memiliki harga rumah yang rendah dan sangat sedikit rumah yang memiliki harga sangat tinggi (outlier), hal ini dapat disimpulkan bahwa MAE lebih cocok digunakan untuk matriks 
  evaluasinya, sebab cenderung lebih adil dan dapat menunjukan apakah model ini lebih baik secara rata-rata umum atau tidak.

- Multivariate analysis menggunakan correlation matrix
  
  ![image](https://github.com/user-attachments/assets/ab4bbaee-7dc0-44d4-bb35-f6160d94d085)

  Semua fitur numeric yang ada terlihat memiliki kontribusi atau korelasi terhadap target, meskipun tidak ada yang sangat dominan korelasinya, namun tidak ada yang berada dibawah 0. dan apabila dilihat dari 
  visualisasi barchart sebelumnya pada multivariate analysis fitur categorical bisa dilihat juga bahwa fitur fitur tersebut selaras dengan harga, semakin lengkap fasilitas terpenuhi, semakin tinggi juga harga 
  rumahnya. **Maka dapat diputuskan bahwa semua fitur akan digunakan untuk melatih model kecuali kolom price (target), karena semua fitur yang ada pada dataset ini memiliki pengaruh dan kontribusi terhadap harga rumah.**

## Data Preparation

Beberapa tahapan data preparation yang dilakukan pada proyek ini adalah:

- **Menghapus outlier**: proses penghapusan outlier dilakukan dengan menggunakan iqr method. Penghapusan outlier adalah proses mengidentifikasi dan menghapus nilai-nilai yang secara signifikan berbeda dari nilai-nilai lain dalam dataset. Outlier dapat disebabkan oleh kesalahan pengukuran, variasi alami, atau faktor eksternal. Menghapus outlier dapat memberikan berbagai manfaat, salah satunya adalah meningkatkan akurasi model, karena outlier dapat mengganggu hasil analisis dan memberikan gambaran yang salah tentang pola dalam data. Adapun iqr method adalah teknik statistik untuk mendeteksi outlier. IQR adalah rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3) dalam dataset. Nilai-nilai yang berada di luar rentang ini dianggap sebagai outlier dan bisa difilter untuk dihapus dari data.

- **Encoding fitur kategorikal**: Menggunakan One-Hot Encoding untuk fitur seperti `Furnishingstatus`, `Mainroad`, dll. One-hot encoding adalah teknik yang digunakan dalam pemrosesan data untuk mengubah data kategorikal menjadi bentuk yang dapat dipahami oleh algoritma machine learning. Teknik ini merepresentasikan setiap kategori sebagai vektor biner yang bernilai 0 dan 1, di mana hanya satu elemen yang bernilai 1 dan sisanya bernilai 0. Hal ini perlu dilakukan karena algoritma machine learning tidak dapat memproses data kategorikal secara langsung. One-hot encoding mengubah data tersebut menjadi bentuk numerik yang dapat diproses oleh algoritma

  
- **Split Dataset**: Split dataset adalah proses membagi dataset menjadi beberapa bagian yang berbeda, pada proyek ini dibagi menjadi dua, yaitu data training (pelatihan) dan data testing (pengujian). Hal ini perlu dilakukan agar model dapat diukur kinerjanya secara adil, dengan menguji performa model menggunakan data yang belum pernah dipelajari sebelumnya. Selain itu itu teknik ini juga dapat menghindari overfitting dimana model 
sangat terbiasa dengan data yang sudah dipelajarinya, tetapi tidak bekerja dengan baik pada data baru.

- **Standarisasi data latih**: Merupakan proses mengubah nilai fitur dalam dataset sehingga memiliki rata-rata (mean) 0 dan simpangan baku (standard deviation). Ini dilakukan dengan mengurangi rata-rata dari setiap nilai fitur dan kemudian membaginya dengan simpangan baku. Teknik ini perlu dilakukan untuk emngatasi skala yang berbeda, terkadang fitur A memiliki nilai rentang nilai yang jauh lebih besar dibandingkan fitur B tanpa standarisasi, fitur dengan skala lebih besar akan mendominasi perhitungan dalam algoritma machine learning.
  
- **Transformasi Log**: Merupakan teknik yang digunakan untuk mengubah data dengan mengambil logaritma dari nilai-nilai aslinya. Transformasi log membantu mengurangi atau menghilangkan kemiringan (skewness) dalam data. Data yang sangat miring dapat menyebabkan masalah dalam analisis statistik dan pemodelan.

## Modeling

Tahapan modeling di awali dengan membuat dataframe yang akan digunakan untuk analisis model. Selanjutnya dibangun 3 buah model berbeda yaitu Linear Regression, Random Forest, dan AdaBoost.

1. **Model development Linear Regression**: Linear regression adalah metode statistik yang digunakan untuk memodelkan hubungan antara satu variabel dependen (yang ingin diprediksi) dan satu atau lebih variabel independen (yang digunakan untuk memprediksi). Metode ini berusaha menemukan garis lurus yang paling sesuai dengan data, yang dapat digunakan untuk memprediksi nilai variabel dependen berdasarkan nilai variabel independen. Kelebihan dari model ini adalah sederhana dan mudah dipahami serta efisien untuk hubungan linear, sebab pada saat dilakukan EDA terdapat variabel area yang linear dengan variabel price (target) maka dari itu model ini patut untuk dicoba, akan tetapi model ini memiliki kekurangan yaitu sensitif terhadap outlier dan tidak efektif untuk hubungan non linear.
   
   Adapun fungsi dan parameter yang digunakan dalam membangun model ini adalah sebagai berikut:
   
   Fungsi:

   - LinearRegression(): Membuat objek model regresi linear.
   - LR.fit(X_train, y_train_log): Melatih model regresi linear menggunakan data pelatihan X_train dan y_train_log.
   - LR.predict(X_train): Memprediksi nilai target menggunakan model yang telah dilatih.
   - np.expm1(y_pred_log): Mengubah prediksi logaritma kembali ke skala asli menggunakan fungsi eksponensial minus satu (expm1).
   - mean_absolute_error(y_train, y_pred): Menghitung Mean Absolute Error (MAE) antara nilai target asli y_train dan prediksi y_pred.
   - models.loc['train_mae','LinearRegression'] = mae: Menyimpan nilai MAE ke dalam DataFrame models pada lokasi yang ditentukan.
     
   Parameter:
   
   - X_train: Data fitur untuk pelatihan.
   - y_train_log: Data target yang telah ditransformasi menggunakan logaritma.
   - y_pred_log: Prediksi yang masih dalam skala logaritma.
   - y_train: Nilai target asli.
   - y_pred: Nilai prediksi yang telah dikembalikan ke skala asli.
   - 'train_mae': Indeks baris.
   - 'LinearRegression': Indeks kolom.
   - mae: Nilai MAE yang dihitung.
         
    
3. **Random Forest Regressor**: Random Forest Regressor adalah algoritma machine learning berbasis ensemble yang digunakan untuk memprediksi nilai numerik atau kontinu. Algoritma ini bekerja dengan membangun beberapa decision trees (pohon keputusan) dari subset data dan fitur yang dipilih secara acak. Hasil prediksi akhir diperoleh dengan menghitung rata-rata dari semua prediksi pohon yang dibangun. kelebihan dari algoritma ini adalah tahan terhadap overfitting dan kuat terhadap outlier. Namun kekurangannya adalah kompleksitas dan memori yang diperlukan bisa sangat banyak akrena membangun banyak pohon.

   Pada model ini terdapat beberapa fungsi dan parameter yang digunakan, dengan rincian sebagai berikut:

   Fungsi:
   
   - RandomForestRegressor(random_state=42): Membuat objek model Random Forest Regressor.
   - param_dist: Dictionary yang berisi parameter dan nilai yang akan dicoba dalam pencarian acak.
   - RandomizedSearchCV: Melakukan pencarian acak untuk menemukan kombinasi parameter terbaik.
   - random_search.fit(X_train, y_train_log): Melatih model menggunakan pencarian acak dengan data pelatihan X_train dan y_train_log.
   - random_search.best_estimator_: Mendapatkan model terbaik berdasarkan pencarian acak.
   - best_rf.predict(X_train): Memprediksi nilai target menggunakan model terbaik yang telah dilatih.
   - np.expm1(y_pred_log): Mengubah prediksi logaritma kembali ke skala asli menggunakan fungsi eksponensial minus satu (expm1).
   - mean_absolute_error(y_train, y_pred): Menghitung Mean Absolute Error (MAE) antara nilai target asli y_train dan prediksi y_pred.
   - models.loc['train_mae', 'RandomForest'] = mae: Menyimpan nilai MAE ke dalam DataFrame models pada lokasi yang ditentukan.
   - random_search.best_params_: memanggil parameter terbaik yang ditemukan oleh pencarian acak.
     
   Parameter:
   - random_state=42: Menetapkan seed untuk memastikan hasil yang dapat direproduksi.
   - n_estimators: jumlah pohon dalam hutan yang dibuat
   - max_depth : kedalaman maksimum masing masing pohon
   - min_samples_split : jumlah minimum sampel untuk membagi node
   - min_samples_leaf : jumlah minimum sampel pada daun pohon
   - max_features : jumlah fitur yang dipertimbangkan saat mencari split terbaik
   - estimator=base_model: Model dasar yang akan dioptimalkan.
   - param_distributions=param_dist: Distribusi parameter yang akan dicoba.
   - n_iter=30: Jumlah iterasi pencarian acak.
   - cv=5: Jumlah fold dalam cross-validation.
   - scoring='neg_mean_absolute_error': Metrik yang digunakan untuk mengevaluasi model.
   - random_state=42: Menetapkan seed untuk memastikan hasil yang dapat direproduksi.
   - n_jobs=-1: Menggunakan semua core prosesor yang tersedia.
   - verbose=1: Menampilkan informasi proses pencarian.
   - X_train: Data fitur untuk pelatihan.
   - y_train_log: Data target yang telah ditransformasi menggunakan logaritma.
   - y_pred_log: Prediksi yang masih dalam skala logaritma.
   - y_train: Nilai target asli.
   - y_pred: Nilai prediksi yang telah dikembalikan ke skala asli.
   - 'train_mae': Indeks baris.
   - 'RandomForest': Indeks kolom.
   - mae: Nilai MAE yang dihitung.

   Setelah proses hyperparameter tunning selesai dilakukan, dapat diketahui bahwa nilai parameter terbaik untuk model yang sudah dicari
   sebelumnya adalah sebagai berikut:
   - n_estimators: 150
   - max_depth : 10
   - min_samples_split : 10
   - min_samples_leaf : 1
   - max_features : log2

3. **AdaBoost Regressor**: Merupakan algoritma machine learning berbasis ensemble yang digunakan untuk meningkatkan akurasi model prediktif dengan menggabungkan beberapa model sederhana (weak learners) menjadi satu model yang lebih kuat (strong learner). Algoritma ini bekerja dengan memberikan bobot lebih tinggi pada data yang salah diprediksi oleh model sebelumnya, sehingga model berikutnya lebih fokus pada kesalahan tersebut. Kelebihannya adalah dapat mengatasi data kompleks dan interaksi antar fitur serta dapat mencegah overfitting. Namun kekurangannya memerlukan waktu komputasi yang lebih lama dan kurang efisien untuk dataset sangat besar karena kompleksitas komputasinya.

   Adapun fungsi dan parameter yang digunakan dalam model ini adalah sebagai berikut:

   Fungsi:
   - AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42): Membuat objek model AdaBoost Regressor dengan Decision Tree
     sebagai estimator dasar.
   - param_dist: Dictionary yang berisi parameter dan nilai yang akan dicoba dalam pencarian acak.
   - RandomizedSearchCV: Melakukan pencarian acak untuk menemukan kombinasi parameter terbaik.
   - random_search.fit(X_train, y_train_log): Melatih model menggunakan pencarian acak dengan data pelatihan X_train dan y_train_log.
   - andom_search.best_estimator_: Mendapatkan model terbaik berdasarkan pencarian acak.
   - best_adaboost.predict(X_train): Memprediksi nilai target menggunakan model terbaik yang telah dilatih.
   - np.expm1(y_pred_log): Mengubah prediksi logaritma kembali ke skala asli menggunakan fungsi eksponensial minus satu (expm1).
   - mean_absolute_error(y_train, y_pred) : Menghitung Mean Absolute Error (MAE) antara nilai target asli y_train dan prediksi y_pred.
   - models.loc['train_mae', 'AdaBoost_Tuned'] = mae: Menyimpan nilai MAE ke dalam DataFrame models pada lokasi yang ditentukan
   - random_search.best_params_: Memanggil parameter terbaik yang ditemukan oleh pencarian acak.

   Parameter:
   - estimator=DecisionTreeRegressor(): Menetapkan Decision Tree sebagai estimator dasar.
   - random_state=42: Menetapkan seed untuk memastikan hasil konsisten yang dapat direproduksi.
   - n_estimators: Jumlah estimator (pohon) dalam ensemble.
   - learning_rate: Tingkat pembelajaran yang mengontrol kontribusi setiap estimator.
   - estimator__max_depth: Kedalaman maksimum pohon dalam estimator dasar.
   - estimator__min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi node dalam estimator dasar.
   - estimator__min_samples_leaf: Jumlah minimum sampel yang diperlukan di setiap daun node dalam estimator dasar.
   - estimator=base_model: Model dasar yang akan dioptimalkan.
   - param_distributions=param_dist: Distribusi parameter yang akan dicoba.
   - n_iter=30: Jumlah iterasi pencarian acak.
   - cv=5: Jumlah fold dalam cross-validation.
   - scoring='neg_mean_absolute_error': Metrik yang digunakan untuk mengevaluasi model.
   - n_jobs=-1: Menggunakan semua core prosesor yang tersedia.
   - verbose=1: Menampilkan informasi proses pencarian.
   - X_train: Data fitur untuk pelatihan.
   - y_train_log: Data target yang telah ditransformasi menggunakan logaritma.
   - y_pred_log: Prediksi yang masih dalam skala logaritma.
   - y_train: Nilai target asli.
   - y_pred: Nilai prediksi yang telah dikembalikan ke skala asli.
   - train_mae': Indeks baris.
   - 'AdaBoost_Tuned': Indeks kolom.
   - mae: Nilai MAE yang dihitung.

   Setelah dilakukan hyperparameter tunning, ditemukan parameter terbaik untuk model ini adalah sebagai berikut:
   - n_estimators: 50
   - learning_rate: 0.01
   - estimator__max_depth: 7
   - estimator__min_samples_split: 10
   - estimator__min_samples_leaf: 1

## Evaluation

Metrik yang digunakan dalam proyek ini adalah **Mean Absolute Error (MAE)**, metrik ini dipilih karena pertimbangan hasil EDA yang menunjukan sebaran harga rumah yang cenderung berkumpul di range harga rendah dan hanya sedikit rumah yang memiliki harga sangat tinggi (adanya outlier ekstrim), oleh karena itu metrik MAE cocok digunakan karena akan lebih adil dengan menilai performa model berdasarkan error rata rata secara umum dan lebih tahan dari error yang terjadi akibat adanya outlier.

Formulai MAE:

![image](https://github.com/user-attachments/assets/9465166c-ffeb-4e74-babd-e9784e237421)

Di mana:
- n adalah jumlah sampel dalam data
- yi adalah nilai aktual
- yi^ adalah nilai prediksi

Cara kerja MAE:
1. Menghitung selisih absolut: Untuk setiap sampel, akan dihitung selisih absolut antara nilai aktual dan nilai prediksi.
2. Rata-rata selisih absolut: Hitung rata rata dari semua selisih aboulut yang telah dihitung 

Setelah dilakukan tahap evaluasi terhadap ketiga model, hasil evaluasinya adalah sebagai berikut:

- **Linear Regression**: train MAE = 685947.065865, test MAE =  622031.295643
- **Random Forest**: train MAE = 538718.73195,  test MAE = 596380.0920
- **Random Forest Regressor**: train MAE = 433662.30811, test MAE = 548786.736431

![image](https://github.com/user-attachments/assets/3d1988e6-27a0-45df-8098-584d7f35cb61)

Berdasarkan hasil tersebut, model AdaBoost ternyata merupakan model terbaik dengan nilai mean absolut error (MAE) yang konsisten lebih rendah dibandingkan dua model lainnya, baik pada saat proses training maupun proses testing. Sehingga dapat diketahui bahwa model AdaBoost mampu memprediksi harga rumah dengan error rata rata secara umum yang paling baik, dan model inilah yang akan dipilih untuk digunakan.
