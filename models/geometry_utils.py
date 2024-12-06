import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict


def line_overlap(line, object_bbox):

    if line[2] < object_bbox[0] or object_bbox[2] < line[0]:
        return False
    if line[3] < object_bbox[1] or object_bbox[3] < line[1]:
        return False

    return True

def process_frames_line(detection_db, line_points):
    total_count = [0, 0, 0]  # People, Cars, Bikes
    tID_set = {0: {}, 1: {}, 2: {}}
    class_map = {"person": 0, "car": 1, "bike": 2}

    for frame in detection_db['frame']:
        for obj in frame['objects']:
            cls_id = class_map[obj['class']]
            for i, bbox in enumerate(obj['bbox']):
                x1, y1, x2, y2 = bbox
                detected_bbox = [x1, y1, x2, y2]
                
                if line_overlap(line_points, detected_bbox):
                    tra_id = obj['tra_id'][i]
                    if tra_id not in tID_set[cls_id]:
                        total_count[cls_id] += 1
                        tID_set[cls_id][tra_id] = []
                        tID_set[cls_id][tra_id].append(frame["timestamp"])
                    else:
                        tID_set[cls_id][tra_id].append(frame["timestamp"])

    return total_count, tID_set


def process_frames_polygon(detection_db, polygon_points):
    total_count = [0, 0, 0]  # People, Cars, Bikes
    tID_set = {0: {}, 1: {}, 2: {}}
    class_map = {"person": 0, "car": 1, "bike": 2}

    for frame in detection_db['frame']:
        for obj in frame['objects']:
            cls_id = class_map[obj['class']]
            for i, bbox in enumerate(obj['bbox']):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if point_in_polygon(center_x, center_y, polygon_points):
                    tra_id = obj['tra_id'][i]
                    if tra_id not in tID_set[cls_id]:
                        total_count[cls_id] += 1
                        tID_set[cls_id][tra_id] = []
                        tID_set[cls_id][tra_id].append(frame["timestamp"])
                    else:
                        tID_set[cls_id][tra_id].append(frame["timestamp"])

    return total_count, tID_set

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside



def marker_result(tid_dic):
    tracK_answer = ""
    class_map = {0: "person", 1: "car", 2: "bike"}
    for cls in [0,1,2]:
        for key in tid_dic[cls].keys():
            frame_id_list = tid_dic[cls][key]
            frame_min = min(frame_id_list)
            frame_max = max(frame_id_list)
            tracK_answer += f'<br>Object class: {class_map[cls]}, Track ID: {key}, View_frame {frame_min} ~ {frame_max}'
    return tracK_answer


def visualize_data(video_id, tid_dic, line=True):
    matplotlib.use('Agg')
    class_map = {0: "Person", 1: "Car", 2: "Bike"}
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    
    # Prepare data
    all_frames = []
    track_frames = defaultdict(lambda: defaultdict(set))
    for cls in [0, 1, 2]:
        for track_id, frames in tid_dic[cls].items():
            all_frames.extend(frames)
            for frame in frames:
                track_frames[cls][frame].add(track_id)
    
    bin_size = 30

    # Determine bin edges
    min_frame, max_frame = min(all_frames), max(all_frames)
    num_bins = (max_frame - min_frame) // bin_size + 1
    bin_edges = [min_frame + i * bin_size for i in range(num_bins + 1)]
    
    # Aggregate data into bins
    binned_data = defaultdict(lambda: defaultdict(set))
    for cls in [0, 1, 2]:
        for frame, track_ids in track_frames[cls].items():
            bin_index = (frame - min_frame) // bin_size
            binned_data[bin_index][cls].update(track_ids)
    
    # Convert to lists for plotting
    bins = list(range(num_bins))
    counts = {cls: [len(binned_data[bin][cls]) for bin in bins] for cls in [0, 1, 2]}
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.25
    index = np.arange(len(bins))
    
    for i, cls in enumerate([0, 1, 2]):
        ax.bar(index + i*bar_width, counts[cls], bar_width, 
                label=class_map[cls], color=colors[cls], alpha=0.7)
    
    ax.set_xlabel('Frame Bins')
    ax.set_ylabel('Unique Object Count')

    ax.set_xticks(index + bar_width)
    ax.set_xticklabels([f'{bin_edges[i]}-{bin_edges[i+1]}' for i in bins])
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    if line:
        plt.savefig(f'./public/frontend/static/images/line_{video_id}.png')
    else:
        plt.savefig(f'./public/frontend/static/images/polygon_{video_id}.png')
    plt.close(fig)

