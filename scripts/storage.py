"""Preview legacy storage, or explicitly delete generated files in one named folder."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--delete-folder',choices=['streamlit_videos','streamlit_images','rtsp_recordings','_uploads','csv','reports'])
args=parser.parse_args()
root=Path(config.OUTPUTS_ROOT).resolve()
for folder in root.iterdir() if root.exists() else []:
    if folder.is_dir():
        files=[p for p in folder.rglob('*') if p.is_file()]
        print(f'{folder.name}: {len(files)} files, {sum(p.stat().st_size for p in files)/1024**2:.2f} MB')
if args.delete_folder:
    target=(root/args.delete_folder).resolve()
    if target.parent != root or target.is_symlink():
        raise SystemExit('Unsafe cleanup path; nothing deleted.')
    # Deliberately non-recursive: only the chosen legacy generated folder, never source/input media elsewhere.
    for path in target.iterdir() if target.exists() else []:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    print('Removed generated files in',target.name)
