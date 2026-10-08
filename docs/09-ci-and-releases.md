# CI ve sürüm yayımlama

[GitHub Actions](https://github.com/omcal/GPU-Training/actions) push/PR'da iki kontrol çalıştırır:

| Kontrol | Kapsam | GPU gerekir mi? |
|---|---|---|
| Repository checks | Python syntax, yerel Markdown dosya linkleri, benchmark CSV yapısı, shell syntax, workflow lint | Hayır |
| CUDA 12.9 / sm_61 compile | Altı CUDA lab + iki PMPP örneğinin derlemesi | Hayır |

CUDA job resmi `nvidia/cuda:12.9.1-devel-ubuntu24.04` image'ını kullanır;
çıktı `cuda-sm61-linux-x64` artifact'ında 7 gün tutulur. Python job yalnız
stdlib kullanır; Linux'a MLX/Metal kurulmaz. Link kontrolü dış URL'lere veya
sayfa içi anchor'lara ağ isteği yapmaz.

GitHub'ın standart runner'ları derleme için kullanılır; GPU çalışma sonucu
kendi Mac/CUDA ortamında doğrulanır. Bu CI, kişisel CUDA makinesine SSH yapmaz.
[Runner kapsamı](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Yerel ve gerçek donanım kontrolleri

Proje kökünden:

```bash
python3 tools/check_repo.py
bash -n tools/sync_cuda.sh
# Apple Silicon / Metal:
.venv/bin/python -m pytest -q
# NVIDIA / CUDA 12.9:
ssh cuda 'cd ~/GPU-Training && make -C cuda test'
```

CUDA testleri CPU referanslarını ve benchmark'ları çalıştırır; hız için pass/fail
sınırı yoktur. Sanitizer hedefleri mevcut toolkit/GPU desteklediğinde ayrıca çalıştırılabilir:
`make -C cuda sanitize`, `racecheck`, `synccheck`.
Güncel donanım kontrolü [8 Ekim kaydında](10-validation-2026-10-08.md).

## Release akışı

Sürüm yayımlamaya hazır olduğunda (bu kurulum otomatik ilk tag oluşturmaz):

```bash
git tag v0.1.0
git push origin v0.1.0
```

`v*` tag workflow'u aynı CI'ı önce çalıştırır. Kontroller geçerse
`vMAJOR.MINOR.PATCH` (opsiyonel suffix) formatını doğrular ve tagged commit'in
kaynak arşivlerini + `SHA256SUMS` dosyasını GitHub Releases'a yayımlar.
Bu repoda dağıtılacak bir servis olmadığından CD çıktısı kaynak release'idir.
GPU-specific binaries CI artifact olarak kalır; portable Python wheel veya
CUDA GPU runtime garantisi olarak yayımlanmaz.

Workflow yalnız release job'ında `contents: write` kullanır. Diğer job'larda
`contents: read`; action referansları commit SHA ile sabitlenmiştir.
Dependabot action güncellemeleri için haftalık PR açar.

Repo adresi değişirse README badge'ini ve bu rehberdeki repo linklerini güncelle.
