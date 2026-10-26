"""CSV export uses stored source timestamps and the run's original classification."""
import csv
import io

def csv_bytes(record):
    buffer=io.StringIO(newline='')
    writer=csv.writer(buffer)
    writer.writerow(['sample','source_time_seconds','count','crowd_level'])
    counts=record['per_frame_counts'];times=record['timestamps'];levels=record['per_frame_levels']
    if not len(counts)==len(times)==len(levels):
        raise ValueError('Frame data and source timestamps must have matching lengths.')
    writer.writerows(zip(range(1,len(counts)+1),times,counts,levels))
    return buffer.getvalue().encode('utf-8')
