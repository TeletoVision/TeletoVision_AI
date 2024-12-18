from flask import Flask, render_template, request, jsonify, send_file
from models.video_retrieval import setup_llm_pipeline
import os

app = Flask(__name__, static_folder='public/frontend/static', template_folder='public/frontend/templates')

app.config['IMAGE_FOLDER'] = 'public/frontend/static/images'
app.config['VIDEO_FOLDER'] = 'public/frontend/static/videos'
app.config['VIDEO_DB']     = 'public/frontend/static/db'

# Global variable to store the current video filename
video_filename = ""
last_search_response = None

llm = setup_llm_pipeline()

def get_available_videos():
    video_folder = app.config['VIDEO_FOLDER']
    return [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

@app.route('/')
def index():
    return render_template('index.html', video_filename=video_filename)

@app.route('/chat', methods=['POST'])
def chat():
    global video_filename
    global llm
    user_message = request.json.get('message')
    bot_response = ""

    if "polygon" in user_message.lower() and "graph" in user_message.lower():
        import time
        # Return the path of the image
        bot_response = "Here is polygon graph."
        timestamp = int(time.time())
        image_url = f"/get_image?filename=polygon_{video_filename}.png&t={timestamp}"
        return jsonify({'response': bot_response, 'image_url': image_url})
    
    elif "line" in user_message.lower() and "graph" in user_message.lower():
        import time
        # Return the path of the image
        bot_response = "Here is line graph."
        timestamp = int(time.time())
        image_url = f"/get_image?filename=line_{video_filename}.png&t={timestamp}"
        return jsonify({'response': bot_response, 'image_url': image_url})

    elif "line" in user_message.lower():
        bot_response = "Please draw the line on the video."
        return jsonify({'response': bot_response})

    elif "polygon" in user_message.lower():
        bot_response = "Please draw the polygon on the video by clicking multiple points, then submit."
        return jsonify({'response': bot_response})
    
    elif "select" in user_message.lower():
        global selected_video_filename
        from models.video_retrieval import process_total_json_from_dataframe
        
        base_directory = app.config['VIDEO_DB']
        db_list        = [filename for filename in os.listdir(base_directory) if 'meta' in filename]
        json_databases = process_total_json_from_dataframe(db_list, base_directory)
        
        # user_message = user_message[len('select'):].strip()
        # question = f"Find the video with {user_message}"
        question = user_message

        docs = json_databases['retriever'].invoke(question)

        if docs:
            full_path = docs[0].metadata['source']
            # Extract only the video filename (e.g., 'cam_06.mp4') from the full path
            selected_video_filename = os.path.basename(full_path)
            # Remove all extensions and add back only .mp4
            selected_video_filename = os.path.splitext(selected_video_filename)[0].split('-')[0]
            bot_response = f"Video changed to: {selected_video_filename}"
            # global video_filename  # 이 줄을 추가
            video_filename = selected_video_filename  # 전역 변수 업데이트
        else:
            bot_response = "No matching video found."
            selected_video_filename = ""
        print(video_filename)

        return jsonify({
            "response": bot_response,
            "filename": selected_video_filename
        })
    elif "find" in user_message.lower():
        global last_search_response
        import re
        from langchain.prompts import PromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser
        from models.video_retrieval import (process_jsons_from_dataframe,
                                        format_docs, PROMPT_TEMPLATE)
        
        json_databases = process_jsons_from_dataframe(f'./public/frontend/static/db/{video_filename}-meta_db.json', video_filename) 
        
        # user_message = user_message[len('search'):].strip()
        #'Find the frame with a person lying on the floor'
        
        question = f'Find the frame with {user_message}'
        # question = user_message

        source = video_filename
        
        retriever = json_databases[source]['retriever']

        template = PROMPT_TEMPLATE
        prompt = PromptTemplate.from_template(template)

        # RAG 체인 정의
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        bot_response = rag_chain.invoke(question)
        bot_response = bot_response.replace('*', '')
        last_search_response = bot_response

        return jsonify({'response': bot_response})
    
    elif "when" in user_message.lower():
        import re
        if last_search_response:
            # 프레임 번호 추출
            frame_numbers = re.findall(r'\d+', last_search_response)
            frame_numbers = [int(num) for num in frame_numbers]

            if frame_numbers:
                min_frame = min(frame_numbers)
                max_frame = max(frame_numbers)

                # 프레임을 초로 변환 (1초당 1프레임 가정, 필요시 조정)
                min_timestamp = f"{min_frame // 60:02d}:{min_frame % 60:02d}"
                max_timestamp = f"{max_frame // 60:02d}:{max_frame % 60:02d}"
                if min_timestamp == max_timestamp:
                    bot_response = f"Based on the previous search, the relevant events occur {min_timestamp}."
                    return jsonify({
                        'response': bot_response,
                        'timestamp': "min_timestamp"
                    })
                else:
                    bot_response = f"Based on the previous search, the relevant events occur between {min_timestamp} and {max_timestamp}."
                    return jsonify({
                        'response': bot_response,
                        'timestamps': [min_timestamp, max_timestamp]
                    })
            else:
                return jsonify({'response': "No specific timestamps found in the previous search result."})
        else:
            return jsonify({'response': "Please perform a search first before asking about when."})
    
    elif "which" in user_message.lower():
        bot_response = "Matched videos are (1.) v1.mp4 (2.) v2.mp4 Choose the video what you want!"
        return jsonify({'response': bot_response})
    
    elif "video" in user_message.lower():
        # Extract the video filename from user input, e.g., "I want to see the video v2.mp4"
        parts = user_message.split()
        new_video_filename = None  # Initialize the variable
        for part in parts:
            if part.endswith(".mp4"):
                new_video_filename = part

        if new_video_filename is None or new_video_filename not in set(get_available_videos()):
            available_videos = get_available_videos()
            if available_videos:
                bot_response = "Please provide a valid video filename."
                video_list = "Available videos are:<br>" + "<br>".join(available_videos)
                return jsonify({'response': bot_response, 'video_list': video_list})
            else:
                bot_response = "No videos are currently available in the video folder."
                return jsonify({'response': bot_response})
        video_filename = new_video_filename
        
        bot_response = f"Video changed to {new_video_filename}"
        return jsonify({
                    "response": bot_response,
                    "filename": new_video_filename
                })


# Route to handle line submission
@app.route('/submit_line', methods=['POST'])
def submit_line():

    import json
    from models.geometry_utils import process_frames_line, marker_result, visualize_data

    with open(f'./public/frontend/static/db/{video_filename}-detection_db.json', 'r') as f:
        detection_db = json.load(f)
    
    points = request.json.get('points')
    print(f"Line drawn with points: {points}")

    line_points = [points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']]

    total_count, tID_set = process_frames_line(detection_db, line_points)
    answer = f'{total_count[0]} People have crossed the black line.<br>{total_count[1]} Cars have crossed the black line.<br>{total_count[2]} Bike have crossed the black line.'
    track_answer = marker_result(tID_set)
    visualize_data(detection_db['video_id'], tID_set)

    return jsonify({'status': 'success', 'message': f'Results of passed the line: <br> {answer}<br>{track_answer}'})

@app.route('/submit_polygon', methods=['POST'])
def submit_polygon():

    import json
    from models.geometry_utils import process_frames_polygon, marker_result, visualize_data

    with open(f'./public/frontend/static/db/{video_filename}-detection_db.json', 'r') as f:
        detection_db = json.load(f)
    
    points = request.json.get('points')
    print(f"Polygon drawn with points: {points}")

    # Convert points to the format expected by process_frames_polygon
    polygon_points = [(point['x'], point['y']) for point in points]

    total_count, tID_set = process_frames_polygon(detection_db, polygon_points)
    answer = f'{total_count[0]} People have entered the polygon area.<br>{total_count[1]} Cars have entered the polygon area.<br>{total_count[2]} Bikes have entered the polygon area.'

    track_answer = marker_result(tID_set)
    visualize_data(detection_db['video_id'], tID_set, False)

    return jsonify({'status': 'success', 'message': f'Results for objects entering the polygon area: <br> {answer}<br>{track_answer}'})


# Route to update video filename
@app.route('/updateVideoSource', methods=['POST'])
def updateVideoSource(filename=None):
    global video_filename
    if filename:
        video_filename = filename
    else:
        video_filename = request.json.get('filename')
    print(f"Video filename updated to: {video_filename}")
    return jsonify({'status': 'success', 'filename': video_filename})

@app.route('/get_image')
def get_image():
    filename = request.args.get('filename')
    file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
    return send_file(file_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
