# CUDA doğrulama — 27 Eylül 2026

Çalıştırılan makine: `cuda-box`, Ubuntu 24.04.5, GTX 1050 Ti Max-Q,
compute capability 6.1, NVIDIA sürücü 580.178.04, CUDA Toolkit 12.9.86,
GCC 13.3.0. Derleme: `-O2 -lineinfo -std=c++17 -arch=sm_61`.

## Sonuç

- `make -C cuda test`: başarılı. Altı laboratuvarda 78 CPU referans
  karşılaştırması geçti. PMPP vector add için 1.048.576, matmul için 65.536
  elemanın tamamı doğru çıktı.
- `make -C cuda sanitize`: sekiz program, her birinde 0 bellek hatası.
- `make -C cuda racecheck`: sekiz program, her birinde 0 hata / 0 uyarı.
- `make -C cuda synccheck`: sekiz program, her birinde 0 senkronizasyon hatası.
- Sanitizer çalışmaları `--test` ile benchmark'ı atlar: her çalışmada
  77 CPU karşılaştırması ve iki PMPP doğrulaması bulunur. Normal çalışmadaki
  ilave kontrol 128 bloklu grid-stride benchmark çıktısını doğrular.

Loglar her iki makinede proje altındaki `build/cuda/test.log`, `memcheck.log`,
`racecheck.log`, `synccheck.log` dosyalarında bulunur. `build/` aktarım dışında
bırakılır; logların yerel kopyası doğrulama sonrasında ayrıca alınmıştır.

## Ölçülen süreler

Aşağıdaki kayıt normal test çalışmasındandır, sanitizer altında ölçülmemiştir.
Metot ve boyutlar `cuda/README.md` ve laboratuvar kaynaklarında belirtilir.
Bunlar kernel süreleridir; CPU–GPU aktarımı ve allocation dahil değildir.

```text
BENCH vector add               0.1274 ms | 98.78 effective GB/s
BENCH grid stride / 128 blocks 0.1293 ms | 97.29 effective GB/s
BENCH transpose naive          0.4914 ms | 25.61 effective GB/s
BENCH transpose tiled          0.1299 ms | 96.83 effective GB/s
BENCH hierarchical reduction   0.1294 ms | 32.42 effective GB/s
BENCH matmul naive             0.3242 ms | 103.50 GFLOP/s
BENCH matmul tiled             0.1602 ms | 209.45 GFLOP/s
filter=3x3 image=256x320
BENCH convolution naive        0.0362 ms
BENCH convolution tiled        0.0292 ms
filter=5x5 image=256x320
BENCH convolution naive        0.0949 ms
BENCH convolution tiled        0.0687 ms
filter=9x9 image=256x320
BENCH convolution naive        0.2780 ms
BENCH convolution tiled        0.1888 ms
BENCH exclusive scan           0.5210 ms | 16.10 effective GB/s
```

Bu çalışmada tiled transpose yaklaşık 3,78×, tiled matmul 2,02× daha hızlıydı.
Tek makine oturumunun ölçümleridir; farklı güç/ısı koşullarıyla yeniden ölçün.
Metal laboratuvarları değiştirilmedi ve bu geçiş sırasında Mac testleri yeniden
çalıştırılmadı. CUDA örnekleri eğitim amaçlıdır; bütün girdi boyutları ve sayısal
koşullar için genel amaçlı kütüphane garantisi vermez.
