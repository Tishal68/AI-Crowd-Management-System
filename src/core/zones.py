"""Bounded spatial summaries; observations are not physical density or unique people."""
import numpy as np
from src.core.scene import validate_polygon, in_zone, capacity_events


class ZoneAnalytics:
    def __init__(self, zones):
        if not 1 <= len(zones) <= 5:
            raise ValueError('Configure between one and five zones.')
        self.zones = []
        for z in zones:
            name = str(z['name']).strip()
            capacity = int(z['capacity'])
            if not name or len(name) > 40 or not 1 <= capacity <= 10000:
                raise ValueError('Zone names need 1–40 characters and capacities 1–10,000 heads.')
            if name.casefold() in [v['name'].casefold() for v in self.zones]:
                raise ValueError('Zone names must be unique.')
            self.zones.append(dict(name=name, capacity=capacity, polygon=validate_polygon(z['polygon']), counts=[]))
        self.grid = np.zeros((36, 64), dtype=np.int64)
        self.samples = 0

    def observe(self, detections, shape):
        selected = set()
        for zone in self.zones:
            inside = in_zone(detections, shape, zone['polygon'])
            zone['counts'].append(len(inside))
            selected.update(id(d) for d in inside)
        union = [d for d in detections if id(d) in selected]
        h, w = shape[:2]
        for d in union:
            x = min(63, max(0, int((d.x1+d.x2)/2/w*64)))
            y = min(35, max(0, int((d.y1+d.y2)/2/h*36)))
            self.grid[y, x] += 1
        self.samples += 1
        return union

    def finish(self, times, seconds, gap):
        zones = []
        for z in self.zones:
            zones.append(dict(z, events=capacity_events(z['counts'], times, z['capacity'], seconds, gap),
                              peak=max(z['counts'], default=0)))
        return zones, dict(grid=self.grid.tolist(), samples=self.samples,
                           measurement='Head-center detections per cell across sampled frames; not unique visitors, dwell time or physical density.')
