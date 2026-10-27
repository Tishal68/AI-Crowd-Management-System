"""Convert the official SCUT-HEAD Part B archive to YOLO, preserving its splits.

Academic research only. Obtain the archive from HCIILAB/SCUT-HEAD-Dataset-Release.
Raw images, XML and generated labels are local data, not repository content.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def convert(archive, output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Choose a new output folder.')
    output.mkdir(parents=True)
    prefix='SCUT_HEAD_Part_B/'
    summary={}; seen=set(); corrections=0; missing_sizes=[]
    with zipfile.ZipFile(archive) as z:
        image_names={Path(n).stem:n for n in z.namelist() if '/JPEGImages/' in n and n.lower().endswith(('.jpg','.jpeg','.png'))}
        for split in ('train','val','test'):
            ids=z.read(prefix+f'ImageSets/Main/{split}.txt').decode().split()
            images=output/'images'/split; labels=output/'labels'/split
            images.mkdir(parents=True); labels.mkdir(parents=True)
            count=0
            for name in ids:
                if name in seen: raise ValueError('Overlapping official splits: '+name)
                seen.add(name)
                raw=z.read(image_names[name]); frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if frame is None: raise ValueError('Unreadable image '+name)
                h,w=frame.shape[:2]
                xml=ET.fromstring(z.read(prefix+'Annotations/'+name+'.xml'))
                declared=(int(xml.findtext('size/width')),int(xml.findtext('size/height')))
                if declared == (0,0):
                    missing_sizes.append(name)
                elif declared != (w,h):
                    raise ValueError('Image/XML dimension mismatch: '+name)
                boxes=[]
                for obj in xml.findall('object'):
                    b=obj.find('bndbox')
                    x1,y1,x2,y2=[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
                    # VOC has one-based inclusive coordinates; clamp documented out-of-image boxes.
                    original=(x1-1,y1-1,x2,y2)
                    x1,y1,x2,y2=max(0,x1-1),max(0,y1-1),min(w,x2),min(h,y2)
                    if (x1,y1,x2,y2)!=original: corrections+=1
                    if x2<=x1 or y2<=y1: raise ValueError('Degenerate box: '+name)
                    boxes.append(f'0 {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                if len(boxes)!=len(set(boxes)): raise ValueError('Duplicate original boxes: '+name)
                (images/(name+Path(image_names[name]).suffix.lower())).write_bytes(raw)
                (labels/(name+'.txt')).write_text('\n'.join(boxes),encoding='utf-8')
                count+=len(boxes)
            summary[split]=dict(images=len(ids),heads=count)
    summary.update(source='https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release',
                   license='Academic research only',archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                   clamped_boxes=corrections,missing_xml_sizes_read_from_image=missing_sizes,
                   conversion='VOC one-based inclusive to clipped YOLO head boxes')
    (output/'provenance.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('archive'); parser.add_argument('output')
    args=parser.parse_args(); convert(args.archive,args.output)
