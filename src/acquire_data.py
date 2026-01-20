import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import re

# Configuration
FFMPEG_URL = "https://evermeet.cx/ffmpeg/getrelease/zip"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(BASE_DIR, "bin")
DATA_DIR = os.path.join(BASE_DIR, "data")
FFMPEG_BIN = os.path.join(BIN_DIR, "ffmpeg")

# Archive.org Item IDs
ARCHIVE_ITEMS = [
    "lp_arabian-music-maqam_various",
    "AliJihadRacySimonShaheenTaqasimTheArtOfImprovisationIn" # Found likely correct ID via search pattern guessing (usually stripping special chars)
    # If that fails, it will just skip.
]

# Maqam Mapping (Filename Keyword -> Directory Name)
MAQAM_MAP = {
    "bayati": "Bayati",
    "bayati": "Bayati",
    "rast": "Rast",
    "hijaz": "Hijaz",
    "hidjaz": "Hijaz",
    "kurd": "Kurd",
    "nahawand": "Nahawand",
    "nihavend": "Nahawand", # Turkish spelling
    "saba": "Saba",
    "sikah": "Sikah",
    "sigah": "Sikah", # Common spelling
    "suznak": "Suznak",
    "ajam": "Ajam", 
    "ushaq": "Bayati", # Often related
}

def install_ffmpeg():
    if os.path.exists(FFMPEG_BIN):
        print(f"[+] ffmpeg found at {FFMPEG_BIN}")
        return

    print(f"[*] Downloading ffmpeg from {FFMPEG_URL}...")
    zip_path = os.path.join(BIN_DIR, "ffmpeg.zip")
    
    try:
        # User-Agent needed sometimes
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)
        print("[*] Extracting ffmpeg...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BIN_DIR)
            
        # Cleanup
        os.remove(zip_path)
        
        # Make executable
        os.chmod(FFMPEG_BIN, 0o755)
        print("[+] ffmpeg installed successfully.")
        
    except Exception as e:
        print(f"[-] Failed to install ffmpeg: {e}")

def get_file_links(archive_id):
    url = f"https://archive.org/download/{archive_id}"
    print(f"[*] Scanning {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')
            
        # Simple regex to find href="...mp3"
        links = re.findall(r'href=["\'](.*?\.mp3)["\']', html)
        return [f"{url}/{link}" for link in links if not link.startswith("/")]
    except Exception as e:
        print(f"[-] Failed to scan {archive_id}: {e}")
        return []

def process_files():
    # Update PATH
    os.environ["PATH"] += os.pathsep + BIN_DIR
    
    for item_id in ARCHIVE_ITEMS:
        links = get_file_links(item_id)
        print(f"[*] Found {len(links)} mp3 files in {item_id}")
        
        for link in links:
            filename = urllib.parse.unquote(os.path.basename(link)).lower()
            
            # Identify Maqam
            target_maqam = None
            for keyword, maqam in MAQAM_MAP.items():
                if keyword in filename:
                    target_maqam = maqam
                    break
            
            if not target_maqam:
                continue
                
            # Prepare paths
            target_dir = os.path.join(DATA_DIR, target_maqam)
            if not os.path.exists(target_dir):
                target_dir = os.path.join(DATA_DIR, target_maqam) # Retrying or skipping if folder usually anticipated
                # We only download if the folder exists (meaning it's a target class)
                continue
                
            local_mp3 = os.path.join(target_dir, filename)
            local_wav = local_mp3.replace(".mp3", ".wav")
            
            if os.path.exists(local_wav):
                print(f"[.] Skipping {filename} (already exists)")
                continue
                
            print(f"[*] Downloading {filename} to {target_maqam}...")
            try:
                urllib.request.urlretrieve(link, local_mp3)
                
                # Convert
                print(f"[*] Converting to WAV...")
                subprocess.run([FFMPEG_BIN, "-i", local_mp3, "-ar", "22050", "-ac", "1", local_wav], 
                             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Cleanup MP3
                os.remove(local_mp3)
                print(f"[+] Processed {filename}")
                
            except Exception as e:
                print(f"[-] Error processing {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(BIN_DIR):
        os.makedirs(BIN_DIR)
        
    install_ffmpeg()
    process_files()
