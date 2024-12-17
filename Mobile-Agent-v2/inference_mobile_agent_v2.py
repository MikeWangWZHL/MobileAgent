import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw

from MobileAgentE.api import inference_chat
from MobileAgentE.text_localization import ocr
from MobileAgentE.icon_localization import det
from MobileAgentE.controller import get_screenshot, tap, swipe, type, back, home, enter, switch_app, clear_background_and_back_to_home, clear_notes, clear_processes, reset_everything
from MobileAgentE.prompt_v2 import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from MobileAgentE.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

####################################### Edit your Setting #########################################
from inference_agent_E import (
    ADB_PATH, INIT_TIPS, INIT_SHORTCUTS,
    OPENAI_API_URL, OPENAI_API_KEY, QWEN_API_KEY,
    REASONING_MODEL, CAPTION_MODEL, 
    FORCE_ADDED, CAPTION_CALL_METHOD
)
# Your ADB path
adb_path = ADB_PATH
# Your instruction

# instruction = "Find the best reviewed Korean restaurant in my area that is within 10min drive and opens late until 10pm."
# run_name = "test"

# Your GPT-4o API URL
API_url = OPENAI_API_URL

# Your GPT-4o API Token
# token = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school", "r").read()
# token = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school_ecole", "r").read()
token = OPENAI_API_KEY

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
caption_call_method = CAPTION_CALL_METHOD

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
caption_model = CAPTION_MODEL

# If you choose the api caption call method, input your Qwen api here
qwen_api = QWEN_API_KEY
# qwen_api = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/qwen_key_from_xi", "r").read()

# You can add operational knowledge to help Agent operate more accurately.
# add_info = "If you want to tap an icon of an app, use the action \"Open_App\". If you want to exit an app, use the action \"Home\""
add_info = FORCE_ADDED + INIT_TIPS

# Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
reflection_switch = True
# reflection_switch = False

# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = True
# memory_switch = False

reasoning_model_name = REASONING_MODEL
###################################################################################################


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size, coord[0] + point_size, coord[1] + point_size), fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./temp/{i}.jpg")


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def process_image(image, query):
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': image
            },
            {
                'text': query
            },
        ]
    }]
    response = MultiModalConversation.call(model=caption_model, messages=messages)
    
    try:
        response = response['output']['choices'][0]['message']['content'][0]["text"]
    except:
        response = "This is an icon."
    
    return response


def generate_api(images, query):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    
    return icon_map


from inference_agent_E import Perceptor, DEFAULT_PERCEPTION_ARGS

def run_single_task(
    instruction,
    run_name="test",
    log_root="log/agent_E",
    task_id=None,
    reset_phone_state=True,
    perceptor: Perceptor = None,
    perception_args=DEFAULT_PERCEPTION_ARGS,
    max_itr=40,
    max_consecutive_failures=3,
    max_repetitive_actions=3,
    overwrite_log_dir=False,
    **kwargs
):
    ### set up log dir ###
    if task_id is None:
        task_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_root}/{run_name}/{task_id}"
    if os.path.exists(log_dir) and not overwrite_log_dir:
        print("The log dir already exists. And overwrite_log_dir is set to False. Skipping...")
        return
    os.makedirs(f"{log_dir}/screenshots", exist_ok=True)
    log_json_path = f"{log_dir}/steps.json"


    ### reset the phone status ###
    if reset_phone_state:
        print("INFO: Reseting the phone states to Home and clear Notes and background apps ...\n")
        reset_everything(ADB_PATH)

    thought_history = []
    summary_history = []
    action_history = []
    action_outcomes = []
    summary = ""
    action = ""
    completed_requirements = ""
    memory = ""
    insight = ""
    temp_file = "temp"
    screenshot = "screenshot"
    if not os.path.exists(temp_file):
        os.mkdir(temp_file)
    else:
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)
    if not os.path.exists(screenshot):
        os.mkdir(screenshot)
    error_flag = False

    # init perceptor if not initialized
    if perceptor is None:
        # if perceptor is not initialized, create the perceptor
        perceptor = Perceptor(ADB_PATH, perception_args=perception_args)

    steps = []
    task_start_time = time.time()
    steps.append({
        "step": 0,
        "operation": "init",
        "instruction": instruction,
        "task_id": task_id,
        "run_name": run_name,
        "max_itr": max_itr,
        "max_consecutive_failures": max_consecutive_failures,
        "max_repetitive_actions": max_repetitive_actions,
        "log_root": log_root,
        "reset_phone_state": reset_phone_state,
        "perception_args": perception_args
    })
    with open(log_json_path, "w") as f:
        json.dump(steps, f, indent=4)

    iter = 0
    while True:

        ## max iteration stop ##
        if max_itr is not None and iter > max_itr:
            print("Max iteration reached. Stopping...")
            task_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "max_iteration",
                "max_itr": max_itr,
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            return
        
        ## consecutive failures stop ##
        if len(action_outcomes) >= max_consecutive_failures:
            last_k_aciton_outcomes = action_outcomes[-max_consecutive_failures:]
            err_flags = [1 if outcome in ["B", "C"] else 0 for outcome in last_k_aciton_outcomes]
            if sum(err_flags) == max_consecutive_failures:
                print("Consecutive failures reaches the limit. Stopping...")
                task_end_time = time.time()
                steps.append({
                    "step": iter,
                    "operation": "finish",
                    "finish_flag": "max_consecutive_failures",
                    "max_consecutive_failures": max_consecutive_failures,
                    "task_duration": task_end_time - task_start_time,
                })
                with open(log_json_path, "w") as f:
                    json.dump(steps, f, indent=4)
                return
        
        ## repetitive actions stop ##
        if len(action_history) >= max_repetitive_actions:
            last_k_actions = action_history[-max_repetitive_actions:]
            last_k_actions_set = set(last_k_actions)
            if len(last_k_actions_set) == 1:
                repeated_action_key = last_k_actions_set.pop()
                if "Swipe" not in repeated_action_key and "Back" not in repeated_action_key:
                    print("Repetitive actions reaches the limit. Stopping...")
                    task_end_time = time.time()
                    steps.append({
                        "step": iter,
                        "operation": "finish",
                        "finish_flag": "max_repetitive_actions",
                        "max_repetitive_actions": max_repetitive_actions,
                        "task_duration": task_end_time - task_start_time,
                    })
                    with open(log_json_path, "w") as f:
                        json.dump(steps, f, indent=4)
                    return

        iter += 1
        if iter == 1:
            screenshot_file = "./screenshot/screenshot.jpg"
            perception_start_time = time.time()
            perception_infos, width, height = perceptor.get_perception_infos(screenshot_file, temp_file=temp_file)
            shutil.rmtree(temp_file)
            os.mkdir(temp_file)
            
            keyboard = False
            keyboard_height_limit = 0.9 * height
            for perception_info in perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    keyboard = True
                    break
            
            ## log ##
            perception_end_time = time.time()
            save_screen_shot_path = f"{log_dir}/screenshots/{iter}.jpg"
            Image.open(screenshot_file).save(save_screen_shot_path)
            steps.append({
                "step": iter,
                "operation": "perception",
                "screenshot": save_screen_shot_path,
                "perception_infos": perception_infos,
                "duration": perception_end_time - perception_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            ##
        
        ## action prompt ##
        action_decision_start_time = time.time()
        prompt_action = get_action_prompt(instruction, perception_infos, width, height, keyboard, summary_history, action_history, summary, action, add_info, error_flag, completed_requirements, memory)
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, screenshot_file)

        output_action = inference_chat(chat_action, reasoning_model_name, API_url, token)
        thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
        summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        chat_action = add_response("assistant", output_action, chat_action)
        status = "#" * 50 + " Decision " + "#" * 50
        print(f"$$$ prompt_action $$$:\n {prompt_action}")
        print(status)
        print(output_action)
        print('#' * len(status))
        action_decision_end_time = time.time()


        ## update memory of current state before executing the action ##
        if memory_switch:
            notetaking_start_time = time.time()
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            output_memory = inference_chat(chat_action, reasoning_model_name, API_url, token)
            chat_action = add_response("assistant", output_memory, chat_action)
            status = "#" * 50 + " Memory " + "#" * 50
            print("$$$ prompt_memory:$$$\n", prompt_memory)
            print(status)
            print(output_memory)
            print('#' * len(status))
            output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory

            ## log ##
            notetaking_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "notetaking",
                "prompt_note": prompt_memory,
                "raw_response": output_memory,
                "important_notes": memory,
                "duration": notetaking_end_time - notetaking_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            ##

        ## action execution ##
        action_execution_start_time = time.time()
        if "Open_App" in action:
            app_name = action.split("(")[-1].split(")")[0]
            text, coordinate = ocr(screenshot_file, perceptor.ocr_detection, perceptor.ocr_recognition)
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [int((coordinate[ti][0] + coordinate[ti][2])/2), int((coordinate[ti][1] + coordinate[ti][3])/2)]
                    tap(adb_path, name_coordinate[0], name_coordinate[1]- int(coordinate[ti][3] - coordinate[ti][1]))# 
                    break
            if app_name in ['Fandango', 'Walmart', 'Best Buy']:
                # additional wait time for app loading
                time.sleep(10)
            time.sleep(10)
        
        elif "Tap" in action:
            coordinate = action.split("(")[-1].split(")")[0].split(", ")
            x, y = int(coordinate[0]), int(coordinate[1])
            tap(adb_path, x, y)
            time.sleep(5)
        
        elif "Swipe" in action:
            coordinate1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
            coordinate2 = action.split("), (")[-1].split(")")[0].split(", ")
            x1, y1 = int(coordinate1[0]), int(coordinate1[1])
            x2, y2 = int(coordinate2[0]), int(coordinate2[1])
            swipe(adb_path, x1, y1, x2, y2)
            time.sleep(5)
            
        elif "Type" in action:
            if "(text)" not in action:
                text = action.split("(")[-1].split(")")[0]
            else:
                text = action.split(" \"")[-1].split("\"")[0]
            type(adb_path, text)
            time.sleep(3)
        
        elif "Enter" in action:
            enter(adb_path)
            time.sleep(10)
        
        elif "Back" in action:
            back(adb_path)
            time.sleep(3)
        
        elif "Home" in action:
            home(adb_path)
            time.sleep(3)
        
        elif "Switch_App" in action:
            switch_app(adb_path)
            time.sleep(3)
        
        elif "Wait" in action:
            time.sleep(10)
            
        elif "Stop" in action:
            print("Stop action is triggered. Finishing...")
            task_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "success",
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            return
        
        
        ## log ##
        action_execution_end_time = time.time()
        steps.append({
            "step": iter,
            "operation": "action",
            "prompt_action": prompt_action,
            "raw_response": output_action,
            "action_object": action,
            "action_thought": thought,
            "action_description": summary,
            "duration": action_decision_end_time - action_decision_start_time,
            "execution_duration": action_execution_end_time - action_execution_start_time,
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)


        ## perception on the next step ##
        perception_start_time = time.time()
        last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = "./screenshot/last_screenshot.jpg"
        last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)
        
        perception_infos, width, height = perceptor.get_perception_infos(screenshot_file, temp_file=temp_file)
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)
        
        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break

        ## log ##
        Image.open(screenshot_file).save(f"{log_dir}/screenshots/{iter+1}.jpg")
        perception_end_time = time.time()
        steps.append({
            "step": iter+1,
            "operation": "perception",
            "screenshot": f"{log_dir}/screenshots/{iter+1}.jpg",
            "perception_infos": perception_infos,
            "duration": perception_end_time - perception_start_time
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        ##

        ## reflection ##
        if reflection_switch:
            action_reflection_start_time = time.time()
            prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height, last_keyboard, keyboard, summary, action, add_info)
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image("user", prompt_reflect, chat_reflect, [last_screenshot_file, screenshot_file])

            output_reflect = inference_chat(chat_reflect, reasoning_model_name, API_url, token)
            reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
            chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            status = "#" * 50 + " Reflcetion " + "#" * 50
            print("$$$ prompt_reflect $$$:\n", prompt_reflect)
            print(status)
            print(output_reflect)
            print('#' * len(status))
            action_reflection_end_time = time.time()
        
            if 'A' in reflect:
                action_outcome = "A"
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)
                
                planning_start_time = time.time()
                prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)
                output_planning = inference_chat(chat_planning, 'gpt-4-turbo', API_url, token)
                chat_planning = add_response("assistant", output_planning, chat_planning)
                status = "#" * 50 + " Planning " + "#" * 50
                print(status)
                print(output_planning)
                print('#' * len(status))
                completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
                
                error_flag = False

                planning_end_time = time.time()
                steps.append({
                    "step": iter+1,
                    "operation": "planning",
                    "raw_response": output_planning,
                    "prompt_planning": prompt_planning,
                    "progress_status": completed_requirements,
                    "duration": planning_end_time - planning_start_time,
                })
                with open(log_json_path, "w") as f:
                    json.dump(steps, f, indent=4)

            
            elif 'B' in reflect:
                action_outcome = "B"
                error_flag = True
                back(adb_path)
                
            elif 'C' in reflect:
                action_outcome = "C"
                error_flag = True

            action_outcomes.append(action_outcome)
            
            ## log ##
            steps.append({
                "step": iter,
                "operation": "action_reflection",
                "prompt_action_reflect": prompt_reflect,
                "raw_response": output_reflect,
                "outcome": action_outcome,
                "duration": action_reflection_end_time - action_reflection_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            ##
        
        else:
            error_flag = False
            
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)
            action_outcomes.append("A")

            
            planning_start_time = time.time()
            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(chat_planning, 'gpt-4-turbo', API_url, token)
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print("$$$ prompt_planning:$$$\n", prompt_planning)
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()

            planning_end_time = time.time()
            steps.append({
                "step": iter+1,
                "operation": "planning",
                "prompt_planning": prompt_planning,
                "raw_response": output_planning,
                "progress_status": completed_requirements,
                "duration": planning_end_time - planning_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

        os.remove(last_screenshot_file)
        
        print("sleeping for 5 before next iteration")
        time.sleep(5)
        ##