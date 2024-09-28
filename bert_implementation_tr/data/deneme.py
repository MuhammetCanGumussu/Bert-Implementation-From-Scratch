import multiprocessing
import ctypes


print(__file__)

def worker(shared_dict):
    # ctypes ile bellek adresini al
    address = ctypes.cast(id(shared_dict), ctypes.py_object).value
    print(f"Worker Process Address: {address}")
    shared_dict['key'] = 'value from worker'

def main():
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()  # Paylaşılan sözlük oluştur

    # ctypes ile bellek adresini al
    address = ctypes.cast(id(shared_dict), ctypes.py_object).value
    print(f"Main Process Address: {address}")

    processes = []
    for _ in range(3):
        p = multiprocessing.Process(target=worker, args=(shared_dict,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Paylaşılan sözlüğü yazdır
    print("Paylaşılan Sözlük:", dict(shared_dict))

if __name__ == "__main__":
    main()