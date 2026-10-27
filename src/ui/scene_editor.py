"""Local, dependency-free Streamlit canvas component for scene geometry."""
from pathlib import Path
import base64
import cv2
import streamlit.components.v1 as components

_editor=components.declare_component('crowdvision_scene',path=str(Path(__file__).parent/'scene_component'))

def scene_editor(frame, key, geometry=None):
    height,width=frame.shape[:2]
    scale=min(1,960/width)
    preview=cv2.resize(frame,(int(width*scale),int(height*scale)))
    ok,encoded=cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok: raise ValueError('Cannot prepare source preview.')
    geometry=geometry or {'polygon':None,'line':None}
    return _editor(image='data:image/jpeg;base64,'+base64.b64encode(encoded).decode(),
                   geometry=geometry,key=key,default=geometry)
