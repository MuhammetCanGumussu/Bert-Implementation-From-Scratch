# Global değişken
sayi = 10

def sayiyi_guncelle(yeni_deger):
    global sayi  # Global değişkeni kullanmak için
    sayi = yeni_deger  # Global değişkeni güncelle

# Fonksiyonu çağır
print("Önceki değer:", sayi)
sayiyi_guncelle(20)
print("Güncellenmiş değer:", sayi)