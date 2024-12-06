import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json
import numpy as np
import subprocess
import cv2

import torch
import torch.nn as nn

from huggingface_hub import snapshot_download
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from accelerate import cpu_offload

# from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, VipLlavaForConditionalGeneration

from bytetrack_yolo.module import YoloByteTrack

import argparse
from utils import *

np.random.seed(0)


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_id", default="demo_1.mp4", type=str)   # demo_1, cam_07
    parser.add_argument("--video_path", default="raw_video/", type=str)
    parser.add_argument("--output_dir_path", default="output_video/", type=str)
    parser.add_argument("--config_path", default="./bytetrack_yolo/configs/yolov8m_bytetrack_skt.json", type=str)
    parser.add_argument("--json_save_path", default="data/", type=str)
    
    # parser.add_argument("--vqa_model_name", default="llava-hf/vip-llava-13b-hf", type=str)
    parser.add_argument("--vqa_model_name", default="llava-hf/vip-llava-7b-hf", type=str)
    # parser.add_argument("--vqa_model_name", default="llava-hf/llama3-llava-next-8b-hf", type=str)
    # parser.add_argument("--vqa_model_name", default="llava-hf/llava-v1.6-vicuna-7b-hf", type=str)
    # parser.add_argument("--vqa_model_name", default="llava-hf/llava-v1.6-mistral-7b-hf", type=str) # Not System Prompt
    
    parser.add_argument("--use_vis_prompt", default=False, type=bool)
    
    parser.add_argument("--index", default='verb', type=str)
    parser.add_argument("--ROI", default='verb', type=str)
    parser.add_argument("--marker_bbox", default=[0.37, 0.55, 0.43, 0.65], nargs='+', type=float)
    parser.add_argument("--line_bbox", default=[0.3, 0.45, 0.5, 0.45], nargs='+', type=float)

    parser.add_argument("--use_marker", default=False, type=bool)
    parser.add_argument("--use_line", default=False, type=bool)
    
    return parser.parse_args()


if __name__ == '__main__':
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ## Setting
    args = parse_args()

    ## Load Prompt
    system_prompt   = load_system_prompt()
    user_prompt     = load_user_prompt()
    # user_prompt     = '10 text output'
    ## Load Path
    video_path          = args.video_path + args.video_id
    output_path         = args.output_dir_path + "oup_" + args.video_id
    final_output_path   = args.output_dir_path + "final_oup_" + args.video_id

    with open(args.config_path,'r') as f :
        config = json.load(f)

    # Load the model
    wrapper         = YoloByteTrack(cfg=config)
    wrapper.model   = wrapper.model.to(device)

    processor = AutoProcessor.from_pretrained(args.vqa_model_name)
    processor.tokenizer.padding_side = "left"

    checkpoint = args.vqa_model_name
    weights_location = snapshot_download(repo_id=checkpoint)

    with init_empty_weights():
        vqa_model = VipLlavaForConditionalGeneration.from_pretrained(
            args.vqa_model_name,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage= True
        )
    # print(vqa_model)
    
    vqa_model = load_checkpoint_and_dispatch(
        vqa_model, checkpoint=weights_location, 
        device_map = "auto", 
        no_split_module_classes=["CLIPEncoderLayer", "LlamaDecoderLayer"]
        )
    vqa_model.tie_weights()
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_id = 0
    frame_list = []
    caption_frame_id = []

    detection_list = []
    detection_frame_id = []

    if args.use_marker:
        markers = True
        marker_tID_intersect  = [{}, {}, {}]
    else:
        markers = False
        marker_tID_intersect = None

    if args.use_line:
        line = True
        total_count  = [0, 0, 0]
        tID_set = [set(), set(), set()]
        
    else:
        line = False
        tID_set = None
        total_count  = None

    while cap.isOpened():
        ret, frame = cap.read()
        print(frame.dtype)
        if not ret:
            break

        frame_id += 1

        # Convert frame to appropriate format and move to device
        input_frame = frame.copy()

        # Perform inferencev
        track_result = wrapper(input_frame, (frame.shape[1], frame.shape[0]))

        # ALL 
        save_tracked_bbox = [[],[],[]]
        save_tracked_id   = [[],[],[]]
        
        # markers_bbox = [0.37, 0.55, 0.43, 0.65] # 실험 3번에 bbox
        markers_bbox = args.marker_bbox
        save_tracked_bbox, save_tracked_id = extract_bbox_trackID(frame, frame_id, track_result, width, height, 
                                                                    args.use_marker, markers_bbox, marker_tID_intersect,
                                                                    args.use_line, args.line_bbox, total_count, tID_set)
        
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
        
        frame_dict = {
                        "image_id"  : frame_id,
                        "timestamp" : frame_id,
                        "objects"   : object_list,
                    }
        
        detection_list.append(frame_dict)

        out.write(frame)
        if (frame_id % fps == 0):
            
            print(f'Frame_id: {frame_id}')
            caption_frame_id.append(frame_id)

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
            if args.use_vis_prompt:
                image = frame
            else:
                image = input_frame

            # os.makedirs(f"output_images/{args.video_id}/", exist_ok=True)
            # image_save_path = f"output_images/{args.video_id}/frame_{frame_id}.jpg"

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

            inputs = processor(text=text, images=image, return_tensors="pt").to("cuda:0")

            prompt_length = inputs['input_ids'].shape[1]
            
            output = vqa_model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            
            caption = processor.decode(output[0][prompt_length:], skip_special_tokens=True)

            frame_dict = {
                "image_id" : frame_id,
                "timestamp" : frame_id,
                "objects" : object_list,
                "caption" : caption,
            }
            frame_list.append(frame_dict)

    ## chema save
    # marker_tID_intersect
    if args.use_marker:
        for cls in [0,1,2]:
            marker_track_result(marker_tID_intersect[cls])

    if args.use_line:
        line_track_result(total_count)

    schema = {
        "video_id" : args.video_id,
        "fps" : config["bytetrack"]["fps"],
        "width": width,
        "height": height,
        "frame" : frame_list
    }

    detection_schema = {
        "video_id" : args.video_id,
        "fps"      : config["bytetrack"]["fps"],
        "width"    : width,
        "height"   : height,
        "frame"    : detection_list
        }

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Convert the output video to H.264 format using ffmpeg
    # subprocess.run(['ffmpeg', '-y', '-i', output_path, '-vcodec', 'libx264', '-crf', '23', final_output_path])

    json_name = args.json_save_path + args.video_id + '-' +'meta_db.json'
    with open(json_name, 'w') as f:
        json.dump(schema, f, indent=4)

    json_name = args.json_save_path + args.video_id + '-' + 'detection_db.json'
    with open(json_name, 'w') as f:
        json.dump(detection_schema, f, indent=4)

    print("done.")
