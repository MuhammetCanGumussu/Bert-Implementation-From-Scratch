import pandas as pd
from multiprocessing import Pool, Manager


def process_item(item, shared_list):
    # İşlem yap ve sonucu paylaşılmış listeye ekle
    result = item * item  # Örnek işlem: sayının karesini alma
    print(shared_list)
    shared_list.append(result)

if __name__ == "__main__":
    with Manager() as manager:
        shared_list = manager.list()  # Paylaşılan liste oluştur
        items = [1, 2, 3, 4, 5]  # İşlem yapılacak öğeler

        with Pool() as pool:
            # Argümanları ile birlikte process_item fonksiyonunu çağır
            pool.starmap(process_item, [(item, shared_list) for item in items])

        print(shared_list)  # Paylaşılan sonuçları yazdır
