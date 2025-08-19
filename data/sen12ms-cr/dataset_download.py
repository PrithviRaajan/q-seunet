import os
import shutil
import random
import tarfile
import glob
import subprocess

# Path where you want to store the processed dataset
output_folder = "/path/to/your/sen12ms_subset"

# Number of samples per season
samples_per_season = 15000  

# ROIs per season used in the example (can add/remove more as needed)
download_links = {
    "spring": {
        "s2_cloudy": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs1158_spring_s2_cloudy.tar.gz",
        "s2_clear": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs1158_spring_s2.tar.gz"
    },
    "summer": {
        "s2_cloudy": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs1868_summer_s2_cloudy.tar.gz",
        "s2_clear": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs1868_summer_s2.tar.gz"
    },
    "fall": {
        "s2_cloudy": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs1970_fall_s2_cloudy.tar.gz",
        "s2_clear": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs1970_fall_s2.tar.gz"
    },
    "winter": {
        "s2_cloudy": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs2017_winter_s2_cloudy.tar.gz",
        "s2_clear": "https://dataserv.ub.tum.de/s/m1554803/download?path=%2F&files=ROIs2017_winter_s2.tar.gz"
    }
}

os.makedirs(output_folder, exist_ok=True)

def download_file(url, outpath):
    if not os.path.exists(outpath):
        print(f"Downloading {outpath} ...")
        subprocess.run(["wget", "--no-check-certificate", "-O", outpath, url])
    else:
        print(f"Already downloaded: {outpath}")

def extract_tar(tar_path, extract_dir):
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

def process_season(season, cloudy_tar, clear_tar):
    print(f"\n=== Processing {season.upper()} ===")

    season_dir = os.path.join(output_folder, season)
    cloudy_dir = os.path.join(season_dir, "s2_cloudy")
    clear_dir   = os.path.join(season_dir, "s2_cloudFree")
    os.makedirs(cloudy_dir, exist_ok=True)
    os.makedirs(clear_dir, exist_ok=True)

    # Temporary extraction folder
    tmp_extract = os.path.join(season_dir, "tmp")
    os.makedirs(tmp_extract, exist_ok=True)

    # Extract archives
    extract_tar(cloudy_tar, tmp_extract)
    extract_tar(clear_tar, tmp_extract)

    # Find files
    cloudy_files = glob.glob(os.path.join(tmp_extract, "**/*"), recursive=True)
    clear_files  = glob.glob(os.path.join(tmp_extract, "**/*"), recursive=True)

    # Keep only files (skip dirs)
    cloudy_files = [f for f in cloudy_files if os.path.isfile(f) and "cloudy" in f]
    clear_files  = [f for f in clear_files if os.path.isfile(f) and "cloudy" not in f]

    print(f"Found {len(cloudy_files)} cloudy and {len(clear_files)} clear images")

    # Match pairs by patch ID
    cloudy_dict = {os.path.basename(f).split("_p")[-1]: f for f in cloudy_files}
    clear_dict  = {os.path.basename(f).split("_p")[-1]: f for f in clear_files}

    common_keys = list(set(cloudy_dict.keys()) & set(clear_dict.keys()))
    print(f"Common pairs: {len(common_keys)}")

    # Randomly sample N pairs
    selected_keys = random.sample(common_keys, min(samples_per_season, len(common_keys)))

    for key in selected_keys:
        c_file = cloudy_dict[key]
        f_file = clear_dict[key]

        new_name = f"{season}_{key}"
        shutil.copy(c_file, os.path.join(cloudy_dir, new_name + ".tif"))
        shutil.copy(f_file, os.path.join(clear_dir, new_name + ".tif"))

    print(f"Saved {len(selected_keys)} pairs for {season}")

    # Cleanup tmp
    shutil.rmtree(tmp_extract)


for season, urls in download_links.items():
    cloudy_tar = os.path.join(output_folder, f"{season}_s2_cloudy.tar.gz")
    clear_tar  = os.path.join(output_folder, f"{season}_s2.tar.gz")

    # Download archives
    download_file(urls["s2_cloudy"], cloudy_tar)
    download_file(urls["s2_clear"], clear_tar)

    # Process season subset
    process_season(season, cloudy_tar, clear_tar)

print("\n✅ Finished building SEN12MS-CR subset (optical only)")
