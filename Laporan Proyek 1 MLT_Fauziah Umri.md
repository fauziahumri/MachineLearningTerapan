# Laporan Proyek 1 Machine Learning Terapan - Fauziah Umri
## Domain Proyek
Anggur merupakan salah satu buah yang banyak di konsumsi dan cukup populer di berbagai wilayah. Buah anggur biasanya di konsumsi secara langsung dan juga diolah menjadi suatu produk seperti makanan dan minuman yang difermentasi yang akan menjadi minuman beralkohol seperti _wine_. Biasanya jangka waktu yang dibutuhkan untuk fermentasi anggur untuk menjadi _wine_ bervariasi, ada yang membutuhkan waktu singkat dan adapula yang membutuhkan waktu yang lama.
Beberapa jenis-jenis _wine_ yaitu _Rose Wine, Sweet Wine, Sparkling Wine, Red Wine, White Wine,_ dan _Fortified Wine_. 

Anggur memiliki berbagai karakteristik seperti kepadatan, nilai pH, alkohol dan asam lainnya. Dalam perkembanganya _wine_ semakin bermacam varianya. Hal itu pula yang membuat _wine_ di bagi berdasarkan kualitasnya untuk memnetukan harga jual di pasaran. Kulaitas pada _wine_ dipengaruhi oleh bebrapa faktor, contohnya komposisi yang terdapat di dalamnya. Untuk menentukan kualitas _wine_ tentu harus ada ahli yang bertugas untuk mencicipi sampel dari minuman anggur tersebut.

Dari permasalah diatas yang akan dilakukan terhadap dataset _Red wine Quality_ adalah melakukan pengujian dengan menggunakan _Random Forest Classsifier_ yang akan dilakukan dengan _tools_ Python.

## Pendefinisian Bisnis

Suatu perusahaan ingin meningkatkan pengetahuan tentang kualitas _wine_ untuk memenuhi kebutuhan dan keinginan konsumen dan memberi petunjuk tentang kemungkinan, dan kesediaan konsumen untuk membeli anggur dengan campuran bahan-bahan tertentu serta memberikan keunggulan bagi produsen dibandingkan pesaing lainnya. Untuk meningkatkan kualitas teresebut perusahaan menggunakan teknologi machine learning untuk memprediksi kualitas wine tersebut. sehingga prediksi dilakukan dengan metode klasifikasi jenis _wine_ mulai dari kualitas yang rendah hingga kualitas yang paling tinggi.

## Masalah

Berdasarkan latar belakang yang telah diuraikan diatas, maka dapat dirumuskan rincian masalah apa saja yang dapat diselesaikan pada proyek ini :
* Bagaimana melakukan pra-pemrosesan data agar bisa digunakan pada model machine learning ?
* Bagaimana membuat model machine learning agar dapat mengklasifikasikan kualitas dari _wine_ ?

## Tujuan

Adapun tujuan dari proyek ini yaitu :
* Melakukan _pra-pemrosesan_ data agar bisa digunakan pada model machine learning
* Membuat model macbine learning untuk mengklasifikasi kualitas _wine_

## Solusi

Adapun solusi untuk mencapai tujuan diatas yaitu :

* _Pra-pemrosesan_ dapat dilakukan dengan beberapa teknik, yaitu
  * Melakukan _Categorical Encoding_ sebagai proses untuk mengubah data numerik menjadi data kategori menggunakan One-Hot Encoding
  * Melakukan _Split Data_ dengan membagi 2 dataset sebagai data latih (train data) dan data test (test data) dengan perbandingan rasio 80% : 20%.
  * Melakukan standardisasi data pada fitur numerik dengan _StandarScaler_.

* Untuk pembuatan model proyek ini menggunakan algoritma *Support Vector Machine* (SVM) sebagai model baseline. Konsep SVM dapat dijelaskan secara sederhana sebagai usaha mencari hyperplane terbaik yang berfungsi sebagai pemisah dua buah kelas pada input space. Pattern merupakan anggota dari dua buah kelas: +1 dan -1 dan berbagi alternatif garis pemisah (discrimination boundaries). Margin adalah jarak antara hyperplane tersebut dengan pattern terdekat dari masing-masing kelas. Pattern yang paling dekat ini disebut sebagai support vector. Usaha untuk mencari lokasi hyperplane ini merupakan inti dari proses pembelajaran pada SVM 
 <img width="596" alt="image" src="https://user-images.githubusercontent.com/96508690/196625493-76f16037-f2e3-468d-a12c-f98c97e8d11e.png">


  Dalam proyek ini menggunakan SVM Klasifikasi Non-Linier. Adapun cara kerjanya yaitu : 
  * Data dimuat
  * Mentransformasikan data menjadi ruang baru
  * Memisahkan data dengan mengimplementasikan beberapa fungsi kernel, antara lain yaitu:
    1. Polynomial
    
          <img width="217" alt="image" src="https://user-images.githubusercontent.com/96508690/196625157-e29ddd80-2c3d-411d-9817-9aa43851e204.png">

       
    2. Gaussian 
    
          <img width="233" alt="image" src="https://user-images.githubusercontent.com/96508690/196625250-40c350e4-9d3c-4135-b3af-01c1fb048207.png">

       
    3. Sigmoid 
    
          <img width="250" alt="image" src="https://user-images.githubusercontent.com/96508690/196625352-8f5a94ca-dc51-4526-8c9c-9f3ce883001f.png">

   
   Adapun kelebihan dan kekurangan dari SVM, antara lain :
   * Kelebihan :
     * Pengklasifikasi SVM menawarkan akurasi yang tinggi dan dapat bekerja dengan baik dengan ruang dimensi tinggi. SVM melakukan klasifikasi pada dasarnya menggunakan subset dari poin pelatihan sehingga hasilnya menggunakan memori yang sangat sedikit.
     * Landasan teori Sebagai metode yang berbasis statistik, SVM memiliki landasan teori yang dapat dianalisa dengan jelas, dan tidak bersifat black box.
     * Feasibility SVM bisa diimplementasikan dengan relatif mudah, karena proses penentuan support vector dapat dirumuskan dalam QP problem.
   * Untuk keekurangan-nya sendiri yaitu :
     * Sulit dipakai dalam problem berskala besar. Yang mana skala besar dalam hal ini dimaksudkan dengan jumlah sample yang diolah.
     * SVM secara teoritik dikembangkan untuk problem klasifikasi dengan dua class
     * Memiliki waktu pelatihan yang tinggi sehingga pada saat praktiknya tidak cocok untuk kumpulan data yang besar

## Data Understanding

Dataset yang digunakan pada proyek ini adalah dataset untuk memprediksi kulalitas _Red Wine_ yang diunduh melalui kaggle : [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/)
Pada dataset yang diunduh terdapat 1599 baris dan memiliki 12 kolom. Berdasarkan informasi dari dataset, variabel yang ada didalam dataset _Red Wine Quality_ sebagai berikut :
 
 (berdasarkan _physicochemical tests_):
  1. fixed acidity
  2. volatile acidity
  3. citric acid
  4. residual sugar
  5. chlorides
  6. free sulfur dioxide
  7. total sulfur dioxide
  8. densty
  9. pH
  10. sulphates
  11. alcohol

 (berdasarkan _sensory data_) :
  
  12. quality

   <img width="417" alt="image" src="https://user-images.githubusercontent.com/96508690/196632795-a8de7606-67ab-468e-a1ac-ffa63dcd734e.png">


Pada gambar yang tertera diatas dijelaskan bahwa pada data hanya memiliki 1 data kategori bertipe object dan data lainnya merupakan data numerik bertipe float64.
Berikut Visualisasi data kategori, yaitu:

![download](https://user-images.githubusercontent.com/96508690/196654702-3edbfdd2-5d6a-4860-bd89-30f25e81d12d.png)

Selanjutnya untuk visualisasi numeriknya dapat dilihat sebagai berikut :

![download](https://user-images.githubusercontent.com/96508690/196655619-833be0fe-aebc-4ce8-9474-aeba27a7d890.png)

Lalu terdapat visualisasi distribusi data pada kolom dengan numerik features dan antar numeric features, yang dapat dilihat sebagai berikut :

![image](https://user-images.githubusercontent.com/96508690/196657155-5e3ab751-57fd-4298-90b9-3d4b51586997.png)

Dan berikut untuk visualisasi heatmap atau kolerasi numeric features :

![image](https://user-images.githubusercontent.com/96508690/196657513-97406276-27ee-4e8f-93ea-655ea7e05892.png)

 - Jika heatmap mendekati 1 maka semakin tinggi pula kolerasi antar fitur numerik
 - Jika heatmap mendekati -1 maka kolerasi antar fitur numerik semakin rendah
 - Jika heatmap mendekati 0 maka kolerasi antar fitur numerik mendekati netral
 
 
 ## Data Preparation
 
 Seperti yang sudah diketahui sebelumnya pada bagian Solution statements ada beberapa tahap-tahap dalam melakukan pra-pemrosesan, yaitu sebagai berikut :
  1. Melakukan _Categorical Encoding_ yang digunakan sebagai proses untuk mengubah data numerik ke data kategori. Untuk teknik Encoding fitur kategori menggunakan One-Hot Encoding. One-Hot Encoding berfungsi untuk data nominal. yang mana data nominal diklasifikasikan tanpa urutan atau peringkat.
  2. _Split Data_ yang merupakan pembagian dataset menjadi 2, yaitu data latih (_train data_) dan data tes (_test data_). Data latih berguna untuk pelatihan model dan data tes untuk menguji model.
  3. Standarisasi data pada _numeric features_ yang memiliki tujuan yaitu agar membuat data numerik pada variabel independen memiliki rentang nilai yang sama.
  
 
 ## Modeling
 
 Setelah melakukan pra-pemrosesan data yang baik, pada tahap modeling akan melakukan 2 hal yaitu tahap pembuatan model baseline dan tahap pembuatan model yang dikembangkan.
 * Model baseline pada tahap ini akan membuat model dasar dengan menggunakan modul dari scikit-learn yaitu SVC dengan parameter default lalu selanjutnya akan melakukan prediksi pada data tes.
 * Model yang dikembangkan akan dilakukan setelah melihat kinerja dari model _baseline_, agar model dapat bekerja lebih optimal maka membutuhkan _Hyper Parameter Tuning_.
 _Hyper Parameter Tuning_ digunakan untuk mencari parameter terbaik yang nanti akan diterpakan pada model _baseline_. Pada analis proyek kali ini akan menggunakan _Grid Search Cross Validation_ dan _Grid Search Cross Validation_ yang mana merupakan metode pemilihan kombinasi model dan hyperparameter dengan cara menguji 1/1 kombinasi dan melakukan validasi untuk setiap kombinasi, tujuannya agar dapat digunakan untuk jadi model saat prediksi.
 
 
 ## Evaluasi
 
 Pada proses evaluasi proyek ini menggunakan _confussion matriks_ .
  * _Confussion matriks_ yaitu pengukur performa untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih.

Berikut perbandingan dari _confussion matriks_ pada analisa kedua model:

 * Model _baseline_


    ![image](https://user-images.githubusercontent.com/96508690/196663921-405ca076-8920-415b-b029-3ede1bef1f5b.png)

* Model yang dikembangkan


    ![image](https://user-images.githubusercontent.com/96508690/196664084-e1b5b04c-2759-4868-9cea-21d84420e8ed.png)

Dari 2 gambar diatas bisa dilihat bahwa nilai _False Positif_ dan _False Negatif_ yang terlihat di model _baseline_ lebih besar daripada model yang dikembangkan.
