# CUDA laboratuvarları — Ubuntu / GTX 1050 Ti

**Güncel çalışma programı:** [8 Ekim–1 Kasım planı](../docs/08-cuda-inference-study-plan.md).
Kendi kernel'lerini [study/](../study/README.md) altında yaz;
[ilerleme çizelgesini](../study/progress.md) kanıtlarla güncelle.


Bu dizin Python veya MLX gerektirmeyen CUDA C++ öğrenme yoludur.
Mac'teki `labs/`, `gpuk/` ve Metal örnekleri ayrı öğrenme yolu olarak korunur.
Altı konunun CUDA başlangıç uygulamaları burada bulunur; Metal laboratuvarlarındaki
bütün optimizasyon varyantlarının birebir portu değildir.

## Çalıştırma

Mac terminalinden:

```bash
ssh cuda 'cd ~/GPU-Training && make -C cuda test'
```

Ubuntu üzerinde:

```bash
cd ~/GPU-Training
make -C cuda test
# Tek bir laboratuvar:
./build/cuda/lab01_vector_add
# Sadece doğruluk testleri, benchmark olmadan:
./build/cuda/lab01_vector_add --test
```

`make` varsayılanları bu makine için `/usr/local/cuda-12.9/bin/nvcc` ve `sm_61`.
SSH oturumunun PATH'ine CUDA eklemek gerekmiyor. Derleme çıktıları `build/cuda/`
altına yazılır. Başka bir GPU kullanırken `ARCH` ve gerekirse `CUDA_HOME` verilebilir.
Derleyici/flag/hedef değiştirildiğinde yeniden derlemek için `make -B` kullanın.

## İçerik ve çalışma sırası

| Dosya | Öğrenilecek konu | Uygulama |
|---|---|---|
| `lab01_vector_add.cu` | Launch geometrisi, global bellek | Her elemana bir thread, grid-stride loop |
| `lab02_coalescing.cu` | Coalescing, shared memory bank conflicts | Naive transpose ve 32x33 shared tile |
| `lab03_reduction.cu` | Blok içi ve bloklar arası toplama | 256-thread tree, özyinelemeli partial sum seviyeleri |
| `lab04_matmul.cu` | Veri tekrar kullanımı ve tiling | Naive ve 16x16 shared-memory matmul |
| `lab05_convolution.cu` | Halo ve sınır koşulları | Naive ve halo tile; 3x3, 5x5, 9x9 filtre |
| `lab06_scan.cu` | Prefix sum, global koordinasyon | Blelloch exclusive scan, blok toplamları ve offset ekleme |

Her dosyada CPU referansı ve farklı boyutlarda doğruluk kontrolleri var.
Girdiler deterministik, karışık işaretli sayılardan oluşur. Matmul referansı double
birikim kullanır. İki uygulamalı testlerde çıktı ikinci kernel'den önce bozulur;
böylece eksik yazan bir kernel önceki doğru sonuçla testi geçemez.
Scan testleri 512²'yi aşarak üç seviyeyi de çalıştırır.

Convolution burada sıfır dolgulu **cross-correlation** anlamındadır: filtre
ters çevrilmez. Scan **exclusive**: çıktı elemanı, kendinden önceki girdilerin toplamıdır.
Örnekler pozitif boyutlar içindir; genel amaçlı bir tensör kütüphanesi API'si değildir.

PMPP'nin mevcut iki `.cu` örneği de `make test` içinde derlenip çalıştırılır.
Ortak `include/common.cuh`, CUDA hatalarını dosya/satır bilgisiyle bildirir ve
başarısız doğruluk kontrolünde programı sıfır olmayan kodla bitirir.

## GPU hata denetimi

```bash
make -C cuda sanitize   # Out-of-bounds ve geçersiz bellek erişimleri
make -C cuda racecheck  # Shared-memory veri yarışları
make -C cuda synccheck  # Geçersiz bariyer/senkronizasyon kullanımı
```

Bu hedefler NVIDIA Compute Sanitizer kullanır ve hata bulursa başarısız olur.
Sanitizer altında benchmark çalıştırılmaz. Kapsam, test edilen boyutlar ve kernel'lerle
sınırlıdır; her olası girdi için doğruluk kanıtı değildir.

## Ölçüm yöntemi

- Önce CPU referansı ile doğruluk, sonra ölçüm.
- 5 ısınma çağrısı, ardından her birinde 20 çağrı bulunan 5 ölçüm; raporlanan süre
  bu 5 ortalama sürenin medyanıdır.
- CUDA events yalnızca GPU çalışmasının süresini ölçer. Bellek ayırma ve CPU–GPU
  aktarımları dışarıdadır; sonuçlar uçtan uca uygulama gecikmesi değildir.
- Reduction/scan ölçümüne bütün kernel seviyeleri dahil, çalışma tamponlarının
  allocation maliyeti hariçtir.
- GB/s yararlı veri boyutuna göre etkin bant genişliğidir; özellikle scan/reduction
  için bütün ara bellek trafiğini saymaz. Matmul işlemleri `2*M*N*K` olarak sayılır.
- Laptop güç/ısı durumu ve kernel'ler arasındaki host launch boşlukları ölçümleri
  etkileyebilir. Bunları GPU'nun teorik tavanı veya M4 ile kontrollü karşılaştırma
  sonucu olarak yorumlamayın.

## Mac'ten değişiklik aktarma

Proje kökünden:

```bash
bash tools/sync_cuda.sh --dry-run
bash tools/sync_cuda.sh
ssh cuda 'cd ~/GPU-Training && make -C cuda test'
```

Aktarım yönü **Mac → Ubuntu**. Script aynı adlı uzak kaynak dosyalarını günceller;
Ubuntu'da elle değişiklik yaptıysanız önce onları Mac'e alın. Uzakta fazladan bulunan
dosyaları silmez. `.venv`, önbellekler, Git metadata ve derleme çıktıları aktarılmaz.

Kurulu CUDA 12.9 bu Pascal kartını destekliyor. Derleme sırasında eski GPU hedefi
hakkında verilen deprecation uyarısı beklenir. CUDA 13'e yükseltmeyin; `sm_61`
için derleme desteği kaldırılmıştır. Ayrıntılar ve makine envanteri:
[`../docs/06-ubuntu-cuda-inventory.md`](../docs/06-ubuntu-cuda-inventory.md).

Bu makinedeki ilk doğrulama ve ölçüm kaydı: [sonuçlar](../docs/07-cuda-validation.md).
