import pysd
import matplotlib.pyplot as plt

def run_simulation():
    # 1. Modeli yükle
    print("Vensim modeli yükleniyor...")
    model = pysd.read_vensim('heat_adjustment_vensim.mdl')
    
    # 2. Kullanıcıdan parametreleri al
    print("\nLütfen simülasyon parametrelerini girin:")
    
    try:
        adj_time = float(input("Adjustment Time (Örn: 5): "))
        meas_delay = float(input("Measurement Delay (Örn: 2): "))
        des_temp = float(input("Desired Temperature (Örn: 25): "))
    except ValueError:
        print("Hata: Lütfen geçerli sayısal değerler girin.")
        return

    # 3. Parametre sözlüğünü oluştur (Vensim'deki isimlerle eşleşmeli)
    new_params = {
        'adjustment time': adj_time,
        'measurement delay': meas_delay,
        'desired temp': des_temp
    }

    # 4. Modeli yeni parametrelerle çalıştır
    print("\nSimülasyon çalıştırılıyor...")
    results = model.run(params=new_params)

    # 5. Sonuçları görselleştir
    # (Modeldeki isimler: 'actual temp' ve 'merasured temp')
    plt.figure(figsize=(10, 6))
    
    plt.plot(results.index, results['actual temp'], 
             label='Actual Temperature', color='blue', linewidth=2)
             
    plt.plot(results.index, results['merasured temp'], 
             label='Measured Temperature', color='red', linestyle='--', linewidth=2)
             
    # Grafik ayarları
    plt.title(f'Sıcaklık Değişimi\n(Desired: {des_temp}, Adj Time: {adj_time}, Delay: {meas_delay})')
    plt.xlabel('Zaman (Month)')
    plt.ylabel('Sıcaklık (Celcius)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Grafiği ekranda göster
    plt.show()

# Fonksiyonu çağır
if __name__ == "__main__":
    run_simulation()