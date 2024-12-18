import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import subprocess
import cv2

import torch
# from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, VipLlavaForConditionalGeneration

from bytetrack_yolo.module import YoloByteTrack

import argparse

np.random.seed(0)


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_id", default="cam_04.mp4", type=str)
    parser.add_argument("--video_path", default="raw_video/", type=str)
    parser.add_argument("--output_dir_path", default="output_video/", type=str)
    parser.add_argument("--config_path", default="./bytetrack_yolo/configs/yolov8m_bytetrack_skt.json", type=str)
    parser.add_argument("--json_save_path", default="data/", type=str)
    
    parser.add_argument("--vqa_model_name", default="llava-hf/vip-llava-7b-hf", type=str)
    # parser.add_argument("--vqa_model_name", default="llava-hf/llama3-llava-next-8b-hf", type=str)
    # parser.add_argument("--vqa_model_name", default="llava-hf/llava-v1.6-vicuna-7b-hf", type=str)
    # parser.add_argument("--vqa_model_name", default="llava-hf/llava-v1.6-mistral-7b-hf", type=str) # Not System Prompt
    
    parser.add_argument("--use_bbox_image", default=False, type=bool)
    parser.add_argument("--use_track_id_image", default=False, type=bool)
    
    parser.add_argument("--index", default='verb', type=str)
    
    return parser.parse_args()


if __name__ == '__main__':

    ## Setting

    args = parse_args()

    ## System-th
    system_prompt = """
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

    ## user-th
    user_prompt = "Can you please describe this image? The image includes bounding box coordinates and their objects: {person_bbox} person, and {car_bbox} car, {bike_bbox} and bike"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_path = args.video_path + args.video_id
    output_path = args.output_dir_path + "oup_" + args.video_id
    final_output_path = args.output_dir_path + "final_oup_" + args.video_id

    with open(args.config_path,'r') as f :
        config = json.load(f)

    # Load the model
    wrapper = YoloByteTrack(cfg=config)
    wrapper.model = wrapper.model.to(device)

    processor = AutoProcessor.from_pretrained(args.vqa_model_name)
    processor.tokenizer.padding_side = "left"
    vqa_model = VipLlavaForConditionalGeneration.from_pretrained(
        args.vqa_model_name,
        # torch_dtype=torch.float16,
        low_cpu_mem_usage= True
    ).to(device)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # # Define the codec and create VideoWriter object for half-sized frames
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width // 2, height // 2))

    # Colors for different classes
    colors = {
        0: (0, 255, 0),   # Green for class 0
        1: (0, 0, 255),   # Red for class 1
        2: (255, 0, 0),   # Blue for class 2
    }

    frame_id = 0
    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Convert frame to appropriate format and move to device
        input_frame = frame.copy()

        # Perform inference
        track_result = wrapper(input_frame, (frame.shape[1], frame.shape[0]))

        save_tracked_bbox = [[],[],[]]
        save_tracked_id   = [[],[],[]]

        # Draw bounding boxes on the frame
        for cls_id in [0, 1, 2]:
            color = colors.get(cls_id, (255, 255, 255))  # Default to white if class not in colors
            for bbox in track_result[cls_id]:
                x1, y1, x2, y2, pd_track_id, cls_score = bbox
                x1, y1, x2, y2, pd_track_id, cls_score = int(x1), int(y1), int(x2), int(y2), int(pd_track_id), round(cls_score, 3)

                # save_tracked_bbox[cls_id].append([x1, y1, x2, y2])
                save_tracked_bbox[cls_id].append(
                    [round(x1/(width), 2), round(y1/(height), 2),
                    round(x2/(width), 2), round(y2/(height), 2)]
                )
                save_tracked_id[cls_id].append(pd_track_id)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if args.use_track_id_image:
                    # Put the class id and score on the box
                    cv2.putText(frame, f'ID: {pd_track_id}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2) # Score: {cls_score}

        # # Resize the frame to half its size
        # half_frame = cv2.resize(frame, (width // 2, height // 2))

        # # Write the half-sized frame to the output video
        # out.write(half_frame)

        ## fps 조건

        # if (frame_id % fps == 0) and (frame_id >= 120) and (frame_id < 600): # cam_04
        # if (frame_id % fps == 0) and (frame_id >= 472) and (frame_id < 1770): # cam_05
        # if (frame_id % fps == 0) and (frame_id >= 472) and (frame_id < 1770): # cam_06
        # if (frame_id % fps == 0) and (frame_id >= 360) and (frame_id < 600): # cam_07
        if (frame_id % fps == 0):

            print(frame_id)

            ## Save Json
            object_list = []

            person_bbox   = save_tracked_bbox[0]
            car_bbox      = save_tracked_bbox[1]
            bike_bbox     = save_tracked_bbox[2]

            person_max_id = max(save_tracked_id[0]) if len(save_tracked_id[0]) > 0 else 0
            car_max_id    = max(save_tracked_id[1]) if len(save_tracked_id[1]) > 0 else 0
            bike_max_id   = max(save_tracked_id[2]) if len(save_tracked_id[2]) > 0 else 0

            person_dict = {
                "class"  : "person",
                "num"    : len(save_tracked_id[0]),
                "max_id" : person_max_id,
                "tra_id" : save_tracked_id[0],
                "bbox"   : person_bbox,
            }

            car_dict = {
                "class"  : "car",
                "num"    : len(save_tracked_id[1]),
                "max_id" : car_max_id,
                "tra_id" : save_tracked_id[1],
                "bbox"   : car_bbox,
            }

            bike_dict = {
                "class"  :"bike",
                "num"    : len(save_tracked_id[2]),
                "max_id" : bike_max_id,
                "tra_id" : save_tracked_id[2],
                "bbox"   : bike_bbox,
            }

            object_list.append(person_dict)
            object_list.append(car_dict)
            object_list.append(bike_dict)

            ## Captioning
            formatted_prompt = user_prompt
            if "{person_bbox}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{person_bbox}", str(person_bbox))
            if "{car_bbox}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{car_bbox}", str(car_bbox))
            if "{bike_bbox}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{bike_bbox}", str(bike_bbox))
            
            if "{person_max_id}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{person_max_id}", str(person_max_id))
            if "{car_max_id}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{car_max_id}", str(car_max_id))
            if "{bike_max_id}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{bike_max_id}", str(bike_max_id))
            
            if "{person_num}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{person_num}", str(len(save_tracked_id[0])))
            if "{car_num}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{car_num}", str(len(save_tracked_id[1])))
            if "{bike_num}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{bike_num}", str(len(save_tracked_id[2])))

            if "{person_track_id}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{person_track_id}", str(save_tracked_id[0]))
            if "{car_track_id}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{car_track_id}", str(save_tracked_id[1]))
            if "{bike_track_id}" in user_prompt:
                formatted_prompt = formatted_prompt.replace("{bike_track_id}", str(save_tracked_id[2]))


            ## 바운딩 박스 이미지를 사용할 경우
            if args.use_bbox_image:
                image = frame
            else:
                image = input_frame

            # image_save_path = f"data/frame_{frame_id}.jpg"
            # cv2.imwrite(image_save_path, image)

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": formatted_prompt},
                    ],
                },
            ]

            text = processor.apply_chat_template(conversation, add_generation_prompt=True)
            print(text)

            inputs = processor(text=text, images=image, return_tensors="pt").to(device)
            prompt_length = inputs['input_ids'].shape[1]

            output = vqa_model.generate(**inputs, max_new_tokens=4096)
            caption = processor.decode(output[0][prompt_length:], skip_special_tokens=True)

            frame_dict = {
                "image_id" : frame_id,
                "timestamp" : frame_id,
                "objects" : object_list,
                "caption" : caption,
            }
            frame_list.append(frame_dict)

    ## chema save

    schema = {
        "video_id" : args.video_id,
        "fps" : config["bytetrack"]["fps"],
        "frame" : frame_list
    }

    # # Release the video capture and writer objects
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
 
    # # Convert the output video to H.264 format using ffmpeg
    # subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'libx264', '-crf', '23', final_output_path])

    json_name = args.json_save_path + args.video_id + '-' + str(args.index) + '.json'
    with open(json_name, 'w') as f:
        json.dump(schema, f, indent=4)

    print("done.")