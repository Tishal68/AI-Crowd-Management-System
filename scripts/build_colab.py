"""Build a self-contained, reviewable Colab notebook from the local training sources."""
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parents[1]
sources={str(p.relative_to(ROOT)).replace('\\','/'):p.read_text(encoding='utf-8') for p in (ROOT/'src').rglob('*.py')}
for name in ['config.py','scripts/prepare_scut.py','scripts/model_experiment.py']:
    sources[name]=(ROOT/name).read_text(encoding='utf-8')
cells=[]
def markdown(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text):
    ast.parse(text)
    cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=text.splitlines(True)))

markdown('''# CrowdVision Campus — cloud GPU training
Improve head counting for recorded campus occupancy review using SCUT-HEAD Part B (academic research only).

Select **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. CPU training is refused. No Drive mounting or local footage uploads are needed.

This notebook downloads the official dataset, converts its head boxes, excludes exact duplicate images across splits, and fine-tunes the existing YOLO11m detector. Validation data tunes confidence; the separate test split compares counting MAE/RMSE and detection precision/recall. Lower MAE is better, not an accuracy percentage.

The original checkpoint's training exposure is unknown. This comparison does not establish independence from its original training set or accuracy at a particular campus.

Download the output ZIP before disconnecting. Colab resources and runtime lifetime are limited.

Dataset source: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
''')
code('''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.90', 'gdown==6.4.2', 'streamlit>=1.59,<2', 'reportlab>=4,<6', 'scipy>=1.12,<2', 'imageio-ffmpeg>=0.6,<1'], check=True)
import torch
assert torch.cuda.is_available(), 'No GPU allocated. Select Runtime > Change runtime type > T4 GPU.'
print('Cloud GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__)
''')
code('''from pathlib import Path
import json, os, hashlib, urllib.request
ROOT=Path('/content/crowdvision_gpu')
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)
SOURCES='''+repr(sources)+'''
for name,content in SOURCES.items():
    path=ROOT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8')
(ROOT/'models').mkdir(exist_ok=True)
baseline=ROOT/'models/head_detector.pt'
if not baseline.exists():
    urllib.request.urlretrieve('https://raw.githubusercontent.com/Tishal68/AI-Crowd-Management-System/af246a2/models/head_detector.pt',baseline)
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='667b4a6da2b8739b4a11e2070e281506e6303de76589cb2832d80c261b99004c','Baseline checksum mismatch'
print('Pinned baseline and source code ready.')
''')
code('''import gdown
archive=ROOT/'SCUT_HEAD_Part_B.zip'
if not archive.exists():
    gdown.download(id='1LZ_KlTPStDEcqycfqUkDkqQ-aNMMC3cl',output=str(archive),quiet=False)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='02afedc58065aca9cb5f28333963a818e37564d75d8420f9bcb94d1e6c340a98','Dataset checksum mismatch'
dataset=ROOT/'scut_head'
if not (dataset/'provenance.json').exists():
    if dataset.exists(): raise RuntimeError('Partial conversion exists. Use a new dataset folder or a fresh runtime.')
    subprocess.run([sys.executable,'scripts/prepare_scut.py',str(archive),str(dataset)],check=True)
print((dataset/'provenance.json').read_text())
''')
markdown('''## Train and compare
Use all deduplicated official splits. Training stops at 30 epochs or after 8 non-improving epochs. If GPU memory runs out, change BATCH to 4 and rerun with a new run name. Exact duplicates are excluded; near-duplicates may remain. Evaluation uses the application's 640-pixel inference setting for both models.
''')
code('''from datetime import datetime,timezone
EPOCHS=30
BATCH=8
RUN_NAME='scut_gpu_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
command=[sys.executable,'-u','scripts/model_experiment.py','--dataset',str(dataset),
         '--name',RUN_NAME,'--epochs',str(EPOCHS),'--train-images','0','--val-images','0',
         '--test-images','0','--device','0','--batch',str(BATCH),'--freeze','0','--imgsz','768','--patience','8']
with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
    for output_line in process.stdout:
        print(output_line,end='',flush=True)
    if process.wait(): raise RuntimeError('Training or evaluation failed. Inspect the output above.')
''')
code('''import zipfile
import pandas as pd
from IPython.display import display
experiment=ROOT/'runs/experiments'/RUN_NAME
comparison=json.loads((experiment/'comparison.json').read_text())
display(pd.DataFrame({'Previous model':comparison['baseline'],'Trained candidate':comparison['candidate']}))
print('Eligible for replacement:',comparison['eligible_for_promotion'])
print(comparison['limitation'])
package=ROOT/(RUN_NAME+'_results.zip')
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for relative in ['comparison.json','manifest.json','training/weights/best.pt','training/weights/last.pt','training/results.csv','training/args.yaml']:
        path=experiment/relative
        if path.exists(): z.write(path,relative)
    z.write(dataset/'provenance.json','dataset_provenance.json')
print('Keep this download:',package.name)
from google.colab import files
files.download(str(package))
''')
for i,cell in enumerate(cells): cell['id']=f'crowdvision-{i:02d}'
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'colab':{'name':'CrowdVision_Campus_GPU_Training.ipynb','provenance':[]},'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'},'accelerator':'GPU'},cells=cells)
destination=ROOT/'notebooks/CrowdVision_Campus_GPU_Training.ipynb'
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
print('Created notebook; every Python cell parses:',destination)
