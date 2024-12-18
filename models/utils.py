import cv2

def get_area(x1, y1, x2, y2):
    return abs((x2 - x1) * (y2 - y1))

def get_intersection_area(rect1, rect2):
    x_left      = max(rect1[0], rect2[0])
    y_top       = max(rect1[1], rect2[1])
    x_right     = min(rect1[2], rect2[2])
    y_bottom    = min(rect1[3], rect2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0  
    return (x_right - x_left) * (y_bottom - y_top)

def check_threshold(rect1, rect2, threshold=0.3):
    area1 = get_area(*rect1)
    area2 = get_area(*rect2)
    intersection_area = get_intersection_area(rect1, rect2)

    if intersection_area == 0:
        return False, intersection_area

    total_area = area1 + area2 - intersection_area
    overlap_ratio = intersection_area / total_area

    return overlap_ratio >= threshold, intersection_area


def line_overlap(line, object_bbox):

    if line[2] < object_bbox[0] or object_bbox[2] < line[0]:
        return False
    if line[3] < object_bbox[1] or object_bbox[3] < line[1]:
        return False

    return True

def load_system_prompt():
    system_prompt="""
    You are an AI visual assistant surveillance operator that can analyze real-time traffic analysis and accident detection.
    
    Specific object locations within the image are given, along with detailed coordinates.
    These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1.
    These values correspond to the top left x, top left y, bottom right x, and bottom right y.

    Using the provided caption and bounding box information, describe the scene in a detailed manner.
    
    Instead of directly mentioning the bounding box coordinates, utilize this data to explain the scene using natural language.
    Include details like object counts, position of the objects, relative position between the objects.
    
    When using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box.
    Only when a safety accident occurs to a person, the bounding box coordinate information represented as (x1, y1, x2, y2) must be mentioned, and the cause of the accident must also be explained.
    
    Always answer as if you are directly looking at the image.
    Be careful not to answer with false information.
    """
    return system_prompt

def load_user_prompt():
    user_prompt = "Can you please describe this image? The image includes bounding box coordinates and their objects: {person_bbox} person, and {car_bbox} car, {bike_bbox} and bike. Note that If, inside the image, you have “ID: number” in the image, output number at the end "
    return user_prompt

def extract_bbox_trackID(frame, frame_id, track_result, width, height
                         ,marker=None, markers_bbox=None, marker_tID_intersect=None
                         ,line=None, line_bbox=None, total_count = None, tID_set= None, nomalized=True):
    
    colors = {
    0: (0, 255, 0),   # Green for class 0
    1: (0, 0, 255),   # Red for class 1
    2: (255, 0, 0),   # Blue for class 2
    }

    save_tracked_bbox = [[],[],[]]
    save_tracked_id   = [[],[],[]]

    for cls_id in [0, 1, 2]:
        color = colors.get(cls_id, (255, 255, 255))
        for bbox in track_result[cls_id]:
            x1, y1, x2, y2, pd_track_id, cls_score = bbox
            x1, y1, x2, y2, pd_track_id, cls_score = int(x1), int(y1), int(x2), int(y2), int(pd_track_id), round(cls_score, 3)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            nomalized_bbox = [round(x1/(width), 2), round(y1/(height), 2), round(x2/(width), 2), round(y2/(height), 2)]

            if nomalized:
                save_tracked_bbox[cls_id].append(nomalized_bbox)
            else:
                save_tracked_bbox[cls_id].append([x1, y1, x2, y2])
            save_tracked_id[cls_id].append(pd_track_id)

            if marker:    
                cv2.rectangle(frame, (int(markers_bbox[0]*(width)), int(markers_bbox[1]*(height))), (int(markers_bbox[2]*(width)), int(markers_bbox[3]*(height))), (144, 238, 144), 2)
                markers_condition, _ = check_threshold(markers_bbox, nomalized_bbox)
                if markers_condition:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID: {pd_track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    if pd_track_id in marker_tID_intersect[cls_id]:
                        marker_tID_intersect[cls_id][pd_track_id].append(frame_id)
                    else:
                        marker_tID_intersect[cls_id][pd_track_id] = []
                        marker_tID_intersect[cls_id][pd_track_id].append(frame_id)
            
            if line:
                cv2.rectangle(frame, (int(line_bbox[0]*(width)), int(line_bbox[1]*(height))), (int(line_bbox[2]*(width)), int(line_bbox[3]*(height))), (144, 238, 144), 2)
                
                line_condition = line_overlap(line_bbox, nomalized_bbox)
                if line_condition:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID: {pd_track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    new_ids = pd_track_id not in tID_set[cls_id]
                    if new_ids:
                        total_count[cls_id] += 1
                        tID_set[cls_id].add(pd_track_id)
    
    return save_tracked_bbox, save_tracked_id

def marker_track_result(tid_dic):
    for key in tid_dic.keys():
        frame_id_list = tid_dic[key]
        frame_min = min(frame_id_list)
        frame_max = max(frame_id_list)
        print(f'Track ID: {key}, View_frame {frame_min} ~ {frame_max}')

def line_track_result(total_count):

    print(f'{total_count[0]} People have crossed the black line.')
    print(f'{total_count[1]} Cars have crossed the black line.')
    print(f'{total_count[2]} Bike have crossed the black line.')