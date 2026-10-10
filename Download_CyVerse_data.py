import os
import requests
import urllib.parse
import time

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

IRODS_PATH = (
    "iplant/home/shared/commons_repo/curated/"
    "EHTC_M87-2018_Mar2024/csv"
)

# URL-encode the iRODS path for CyVerse REST API
ENCODED_PATH = urllib.parse.quote(IRODS_PATH, safe="")

# CyVerse REST API endpoints
LIST_URL = f"https://datacommons.cyverse.org/irods-rest/1.0.0/collection/{ENCODED_PATH}"
DOWNLOAD_BASE = "https://datacommons.cyverse.org/irods-rest/1.0.0/file/"

# Local directory to save files
LOCAL_DIR = "C:/Users/lanto/Documents/EHT_2018_CSVs"
os.makedirs(LOCAL_DIR, exist_ok=True)

# ------------------------------------------------------------
# FETCH DIRECTORY LISTING
# ------------------------------------------------------------

print("Fetching directory listing from CyVerse API...")
r = requests.get(LIST_URL)

if r.status_code != 200:
    raise RuntimeError(f"Directory listing failed: HTTP {r.status_code}")

listing = r.json()

if "children" not in listing:
    raise RuntimeError("CyVerse API returned no 'children' field.")

files = [item["path"] for item in listing["children"]
         if item["path"].endswith(".csv")]

print(f"Found {len(files)} CSV files.")

# ------------------------------------------------------------
# DOWNLOAD FILES
# ------------------------------------------------------------

for path in files:
    filename = os.path.basename(path)
    encoded_file_path = urllib.parse.quote(path, safe="")
    url = DOWNLOAD_BASE + encoded_file_path
    local_path = os.path.join(LOCAL_DIR, filename)

    print(f"\nDownloading: {filename}")
    print(f"From: {url}")

    for attempt in range(3):
        try:
            resp = requests.get(url, stream=True)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Saved to: {local_path}")
                break
            else:
                print(f"HTTP {resp.status_code}, retrying...")
        except Exception as e:
            print(f"Error: {e}, retrying...")

        time.sleep(1)

    else:
        print(f"FAILED to download {filename} after 3 attempts.")

print("\nAll downloads complete.")
