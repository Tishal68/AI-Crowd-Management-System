"""Scene geometry, sampled capacity episodes, and experimental local flow tracking."""
import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def validate_polygon(points):
    if points is None:
        return [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 12:
        raise ValueError('Choose 3 to 12 corners for the counting area.')
    if not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
        raise ValueError('Area points must lie inside the image.')
    # Convex zones make selection unambiguous and reject crossing/degenerate outlines.
    if not cv2.isContourConvex(polygon.astype(np.float32)) or abs(cv2.contourArea(polygon.astype(np.float32))) < .005:
        raise ValueError('Draw a convex area in clockwise order, without crossing edges (minimum 0.5% of image).')
    return polygon.tolist()


def validate_line(points):
    arr = np.asarray(points, dtype=float)
    if arr.shape != (2, 2) or not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any() or np.linalg.norm(arr[1]-arr[0]) < .05:
        raise ValueError('Set two different flow-line endpoints inside the image, at least 5% apart.')
    return arr.tolist()


def in_zone(detections, shape, polygon):
    h, w = shape[:2]
    contour = np.asarray(validate_polygon(polygon),np.float32) * [w,h]
    return [d for d in detections if cv2.pointPolygonTest(contour.astype(np.float32),
            ((d.x1+d.x2)/2,(d.y1+d.y2)/2),False) >= 0]


def draw_scene(frame, polygon, line=None):
    h,w=frame.shape[:2]
    contour=(np.asarray(validate_polygon(polygon))*[w-1,h-1]).astype(np.int32)
    cv2.polylines(frame,[contour],True,(210,195,65),2,cv2.LINE_AA)
    if line is not None:
        a,b=(np.asarray(validate_line(line))*[w-1,h-1]).astype(int)
        cv2.line(frame,tuple(a),tuple(b),(70,170,255),2,cv2.LINE_AA)
        cv2.putText(frame,'A',tuple(a),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,170,255),2)
        cv2.putText(frame,'B',tuple(b),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,170,255),2)
        vector=b-a
        midpoint=(a+b)/2
        endpoint=midpoint+np.array([-vector[1],vector[0]])/np.linalg.norm(vector)*min(w,h)*.07
        cv2.arrowedLine(frame,tuple(midpoint.astype(int)),tuple(endpoint.astype(int)),(70,170,255),2,cv2.LINE_AA,tipLength=.25)
    return frame


def capacity_events(counts, timestamps, capacity, minimum_seconds=3., max_gap_seconds=1.):
    """Confirmed sampled episodes; never extrapolate beyond the final observed sample."""
    if len(counts)!=len(timestamps) or capacity <= 0 or minimum_seconds < 0:
        raise ValueError('Invalid event series or capacity.')
    events=[]; start=None; end=None; peak=0; peak_time=0
    def finish():
        if start is not None and end-start+1e-6 >= minimum_seconds:
            events.append(dict(start_seconds=start,end_seconds=end,duration_seconds=end-start,
                               peak_count=peak,peak_seconds=peak_time,capacity=capacity))
    previous=None
    for count,t in zip(counts,timestamps):
        if not math.isfinite(t) or (previous is not None and t <= previous):
            raise ValueError('Event timestamps must increase.')
        if previous is not None and t-previous > max_gap_seconds+1e-6:
            finish(); start=None
        if count > capacity:
            if start is None:
                start=t; peak=count; peak_time=t
            if count > peak:
                peak=count; peak_time=t
            end=t
        elif start is not None:
            finish(); start=None
        previous=t
    finish()
    return events


class FlowTracker:
    """Conservative centroid association, not biometric identity or unique visitors.

    Matches nearby observed heads using Hungarian assignment and short-lived IDs.
    Crossings require two observations on each side and intersection with the finite line.
    """
    def __init__(self, line, max_age=1., max_distance=.08):
        self.line=np.asarray(validate_line(line),float)
        self.max_age=max_age; self.max_distance=max_distance
        self.tracks={}; self.next_id=1; self.forward=0; self.reverse=0; self.events=[]

    def side(self,p):
        a,b=self.line; vector=b-a
        distance=(vector[0]*(p[1]-a[1])-vector[1]*(p[0]-a[0]))/np.linalg.norm(vector)
        return 1 if distance>.01 else -1 if distance<-.01 else 0

    def crosses_segment(self,p,q):
        a,b=self.line; v=b-a; move=q-p
        determinant=v[0]*move[1]-v[1]*move[0]
        if abs(determinant)<1e-9: return False
        delta=p-a
        u=(delta[0]*move[1]-delta[1]*move[0])/determinant
        return 0 <= u <= 1

    def update(self,detections,shape,timestamp):
        h,w=shape[:2]
        points=[np.array([(d.x1+d.x2)/(2*w),(d.y1+d.y2)/(2*h)]) for d in detections]
        self.tracks={k:v for k,v in self.tracks.items() if timestamp-v['time'] <= self.max_age}
        ids=list(self.tracks); assigned={}
        if ids and points:
            costs=np.array([[np.linalg.norm(self.tracks[k]['point']-p) for p in points] for k in ids])
            rows,cols=linear_sum_assignment(np.where(costs<=self.max_distance,costs,1000.))
            assigned={int(c):ids[int(r)] for r,c in zip(rows,cols) if costs[r,c]<=self.max_distance}
        for i,(det,p) in enumerate(zip(detections,points)):
            tid=assigned.get(i)
            if tid is None:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]=dict(point=p,time=timestamp,stable_side=0,anchor=p,candidate=0,hits=0)
            track=self.tracks[tid]; side=self.side(p)
            if side:
                track['hits']=track['hits']+1 if track['candidate']==side else 1
                track['candidate']=side
                if track['hits']>=2:
                    if track['stable_side'] and side!=track['stable_side'] and self.crosses_segment(track['anchor'],p):
                        direction='along arrow' if side>0 else 'against arrow'
                        if side>0: self.forward+=1
                        else: self.reverse+=1
                        self.events.append(dict(time_seconds=timestamp,direction=direction,track_id=tid))
                        if len(self.events)>10000:
                            raise ValueError('Too many flow crossings for one clip. Use a shorter recording.')
                    track['stable_side']=side; track['anchor']=p
            track.update(point=p,time=timestamp)
            det.track_id=tid
        return detections
