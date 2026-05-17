import os
import zipfile
import shutil
import urllib.request

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR      = "data"
NON_MAKEUP    = os.path.join(DATA_DIR, "non_makeup")
MAKEUP        = os.path.join(DATA_DIR, "makeup")

# MT-Dataset via BeautyGAN repo (most stable public mirror)
DATASET_URL   = "https://github.com/wtjiang98/BeautyGAN_pytorch/archive/refs/heads/master.zip"
ZIP_NAME      = "beautygan.zip"
EXTRACT_DIR   = "beautygan_tmp"
# ─────────────────────────────────────────────────────────────────────────

def download(url, dest):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()

def progress(count, block, total):
    pct = min(int(count * block * 100 / total), 100)
    print(f"\r  {pct}%", end="", flush=True)

def setup_dirs():
    os.makedirs(NON_MAKEUP, exist_ok=True)
    os.makedirs(MAKEUP,     exist_ok=True)
    print(f"Created: {NON_MAKEUP}")
    print(f"Created: {MAKEUP}")

def extract_and_organize():
    print("Extracting zip ...")
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(EXTRACT_DIR)

    # Locate image folders inside the extracted repo
    # BeautyGAN repo structure: .../data/images/non-makeup / makeup
    non_mk_src, mk_src = None, None
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if os.path.basename(root) in ("non-makeup", "non_makeup"):
            non_mk_src = root
        if os.path.basename(root) == "makeup" and "non" not in root:
            mk_src = root

    if not non_mk_src or not mk_src:
        print("\n[!] Could not auto-locate image folders.")
        print(f"    Please manually copy images into:\n    {NON_MAKEUP}\n    {MAKEUP}")
        return

    # Copy images
    for src, dst in [(non_mk_src, NON_MAKEUP), (mk_src, MAKEUP)]:
        imgs = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for img in imgs:
            shutil.copy2(os.path.join(src, img), os.path.join(dst, img))
        print(f"Copied {len(imgs)} images → {dst}")

def cleanup():
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    print("Cleaned up temp files.")

def verify():
    nm = len([f for f in os.listdir(NON_MAKEUP) if f.endswith(".jpg")])
    mk = len([f for f in os.listdir(MAKEUP)     if f.endswith(".jpg")])
    print(f"\nDataset ready:")
    print(f"  non_makeup/ : {nm} images")
    print(f"  makeup/     : {mk} images")
    if nm == 0:
        print("\n[WARNING] non_makeup folder is empty!")
        print("  Download the MT-Dataset manually from:")
        print("  https://github.com/wtjiang98/BeautyGAN_pytorch")
        print("  and place images in: data/non_makeup/")

if __name__ == "__main__":
    setup_dirs()
    try:
        download(DATASET_URL, ZIP_NAME)
        extract_and_organize()
        cleanup()
    except Exception as e:
        print(f"\n[!] Auto-download failed: {e}")
        print("  Manual fallback instructions:")
        print("  1. Go to https://github.com/wtjiang98/BeautyGAN_pytorch")
        print("  2. Download the repo as ZIP")
        print("  3. Copy the images into data/non_makeup/ and data/makeup/")
    verify()