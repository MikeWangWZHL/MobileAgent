import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw

from MobileAgentE.api import inference_chat
from MobileAgentE.text_localization import ocr
from MobileAgentE.icon_localization import det
from MobileAgentE.controller import get_screenshot, back, clear_background_and_back_to_home, clear_notes
from MobileAgentE.agents import InfoPool, Manager, Executor, Notetaker, ActionReflector, KnowledgeReflector, INIT_SHORTCUTS
from MobileAgentE.agents import add_response, add_response_two_image
from MobileAgentE.agents import ATOMIC_ACTION_SIGNITURES

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent
import json
from dataclasses import dataclass, field, asdict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

####################################### Edit your Setting #########################################
# Your ADB path
ADB_PATH = "/Users/wangz3/Desktop/vlm_agent_project/platform-tools/adb"

# Your GPT-4o API URL
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Your GPT-4o API Token
# OPENAI_API_KEY = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school", "r").read()
# OPENAI_API_KEY = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school_ecole", "r").read()
OPENAI_API_KEY = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_taobao", "r").read()

USAGE_TRACKING_JSONL = "usage/from_dec_5.jsonl"

# Reasoning GPT model
REASONING_MODEL = "gpt-4o-2024-11-20"

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
CAPTION_CALL_METHOD = "api"

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
CAPTION_MODEL = "qwen-vl-plus"

# If you choose the api caption call method, input your Qwen api here
QWEN_API_KEY = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/qwen_key_mine", "r").read()
# QWEN_API_KEY = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/qwen_key_from_xi", "r").read()

# # You can add operational knowledge to help Agent operate more accurately.
INIT_TIPS = """1. By default, no APPs are opened in the background.
2. Screenshots may show partial text in text boxes from your previous input; this does not count as an error.
3. When creating new Notes, you do not need to enter a title unless the user specifically requests it.
"""
# 2. If you want to type new text in a search box that already contains text you previously entered, make sure to clear it first by tapping the 'X' button.

EVOLVE_FREQ = 5 # how often the agent evolves its knowledge; in iterations

TEMP_DIR = "temp"
SCREENSHOT_DIR = "screenshot"

# # Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
# reflection_switch = True
# # reflection_switch = False

# # Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
# memory_switch = True
# # memory_switch = False

###################################################################################################
### Perception related functions ###

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


def crop(image, box, i, temp_file=TEMP_DIR):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    save_path = os.path.join(temp_file, f"{i}.jpg")
    cropped_image.save(save_path)


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def process_image(image, query, caption_model=CAPTION_MODEL):
    dashscope.api_key = QWEN_API_KEY
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


def generate_api(images, query, caption_model=CAPTION_MODEL):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query, caption_model=caption_model): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    
    return icon_map


def merge_text_blocks(
    text_list,
    coordinates_list,
    x_distance_threshold=45,
    y_distance_min=-20,
    y_distance_max=30,
    height_difference_threshold=20,
    # x_distance_threshold=10,
    # y_distance_min=-10,
    # y_distance_max=30,
    # height_difference_threshold=10,
):
    merged_text_blocks = []
    merged_coordinates = []

    # Sort the text blocks based on y and x coordinates
    sorted_indices = sorted(
        range(len(coordinates_list)),
        key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]),
    )
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue

        anchor = i
        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i + 1, num_blocks):
            if merge[j]:
                continue

            # Calculate differences and thresholds
            x_diff_left = abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0])
            x_diff_right = abs(sorted_coordinates_list[anchor][2] - sorted_coordinates_list[j][2])

            y_diff = sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3]
            height_anchor = sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1]
            height_j = sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1]
            height_diff = abs(height_anchor - height_j)

            if (
                (x_diff_left + x_diff_right) / 2 < x_distance_threshold
                and y_distance_min <= y_diff < y_distance_max
                and height_diff < height_difference_threshold
            ):
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = "\n".join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])
    print(text_list, len(text_list))
    print(merged_text_blocks, len(merged_text_blocks))
    for t in merged_text_blocks:
        if "\n" in t:
            print("merged text:", t)
    return merged_text_blocks, merged_coordinates

###################################################################################################


def load_perception_models(
    device="cuda",
    caption_call_method=CAPTION_CALL_METHOD,
    caption_model=CAPTION_MODEL,
    groundingdino_model="AI-ModelScope/GroundingDINO",
    groundingdino_revision="v1.0.0",
    ocr_detection_model="iic/cv_resnet18_ocr-detection-db-line-level_damo",
    ocr_recognition_model="iic/cv_convnextTiny_ocr-recognition-document_damo",
    ):

    ### Load caption model ###
    if caption_call_method == "local":
        if caption_model == "qwen-vl-chat":
            model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
            vlm_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
            vlm_model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif caption_model == "qwen-vl-chat-int4":
            qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
            vlm_model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True,use_safetensors=True).eval()
            vlm_model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True, do_sample=False)
        else:
            print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
            exit(0)
        vlm_tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
    elif caption_call_method == "api":
        vlm_model = None
        vlm_tokenizer = None
        pass
    else:
        print("You must choose the caption model call function from \"local\" and \"api\"")
        exit(0)


    ### Load ocr and icon detection model ###
    groundingdino_dir = snapshot_download(groundingdino_model, revision=groundingdino_revision)
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
    # ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    ocr_detection = pipeline(Tasks.ocr_detection, model=ocr_detection_model) # dbnet (no tensorflow)
    # ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
    ocr_recognition = pipeline(Tasks.ocr_recognition, model=ocr_recognition_model)

    print("INFO: Loaded perception models:")
    print("\t- Caption model method:", caption_call_method, "| caption vlm model:", caption_model)
    print("\t- Grounding DINO model:", groundingdino_model)
    print("\t- OCR detection model:", ocr_detection_model)
    print("\t- OCR recognition model:", ocr_recognition_model)
    return ocr_detection, ocr_recognition, groundingdino_model, vlm_model, vlm_tokenizer


DEFAULT_PERCEPTION_ARGS = {
    "device": "cuda",
    "caption_call_method": CAPTION_CALL_METHOD,
    "caption_model": CAPTION_MODEL,
    "groundingdino_model": "AI-ModelScope/GroundingDINO",
    "groundingdino_revision": "v1.0.0",
    "ocr_detection_model": "iic/cv_resnet18_ocr-detection-db-line-level_damo",
    "ocr_recognition_model": "iic/cv_convnextTiny_ocr-recognition-document_damo",
}

class Perceptor:
    def __init__(self, adb_path, perception_args = DEFAULT_PERCEPTION_ARGS):
        self.ocr_detection, self.ocr_recognition, self.groundingdino_model, \
            self.vlm_model, self.vlm_tokenizer = load_perception_models(**perception_args)
        self.adb_path = adb_path

    def get_perception_infos(self, screenshot_file, temp_file=TEMP_DIR):
        get_screenshot(self.adb_path)
        
        width, height = Image.open(screenshot_file).size
        
        text, coordinates = ocr(screenshot_file, self.ocr_detection, self.ocr_recognition)
        text, coordinates = merge_text_blocks(text, coordinates)
        
        center_list = [[(coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2] for coordinate in coordinates]
        draw_coordinates_on_image(screenshot_file, center_list)
        
        perception_infos = []
        for i in range(len(coordinates)):
            perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
            perception_infos.append(perception_info)
            
        coordinates = det(screenshot_file, "icon", self.groundingdino_model)
        
        for i in range(len(coordinates)):
            perception_info = {"text": "icon", "coordinates": coordinates[i]}
            perception_infos.append(perception_info)
            
        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            if perception_infos[i]['text'] == 'icon':
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)

        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i], temp_file=temp_file)

        images = get_all_files_in_folder(temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
            if CAPTION_CALL_METHOD == "local":
                for i in range(len(images)):
                    image_path = os.path.join(temp_file, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = generate_local(self.vlm_tokenizer, self.vlm_model, image_path, prompt)
                    icon_map[i+1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(temp_file, images[i])
                icon_map = generate_api(images, prompt, caption_model=CAPTION_MODEL)
            for i, j in zip(image_id, range(1, len(image_id)+1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] = "icon: " + icon_map[j]

        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
            
        return perception_infos, width, height

###################################################################################################

def finish(
        info_pool: InfoPool,
        persistent_knowledge_path=None,
        persistent_shortcuts_path=None
    ):
    
    print("Plan:", info_pool.plan)
    print("Progress Logs:")
    for i, p in enumerate(info_pool.progress_status_history):
        print(f"Step {i}:", p, "\n")
    print("Important Notes:", info_pool.important_notes)
    print("Finish Thought:", info_pool.finish_thought)
    if persistent_knowledge_path:
        print("Update persistent knowledge:", persistent_knowledge_path)
        with open(persistent_knowledge_path, "w") as f:
            f.write(info_pool.additional_knowledge)
    if persistent_shortcuts_path:
        print("Update persistent shortcuts:", persistent_shortcuts_path)
        with open(persistent_shortcuts_path, "w") as f:
            json.dump(info_pool.shortcuts, f)
    # exit(0)

def run_single_task(
    instruction,
    future_tasks=[],
    run_name="test",
    log_root="log/agent_E",
    task_id=None,
    knowledge_path=None,
    shortcuts_path=None,
    persistent_knowledge_path=None, # cross tasks
    persistent_shortcuts_path=None, # cross tasks
    reset_phone_state=True,
    perceptor: Perceptor = None,
    perception_args=DEFAULT_PERCEPTION_ARGS,
    max_itr=None,
    overwrite_log_dir=False,
):

    ### reset the phone status ###
    if reset_phone_state:
        print("INFO: Reseting the phone states to Home and clear background apps ...\n")
        clear_background_and_back_to_home(ADB_PATH)
        # clear_notes(ADB_PATH)

    ### set up log dir ###
    if task_id is None:
        task_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_root}/{run_name}/{task_id}"
    if os.path.exists(log_dir) and not overwrite_log_dir:
        raise ValueError("The log dir already exists. And overwrite_log_dir is set to False.")
    os.makedirs(f"{log_dir}/screenshots", exist_ok=True)
    log_json_path = f"{log_dir}/steps.json"
    
    # local experience save paths
    local_shortcuts_save_path = f"{log_dir}/shortcuts.json" # single-task setting
    local_knowledge_save_path = f"{log_dir}/knowledge.txt" # single-task setting

    ### Init Information Pool ###
    if shortcuts_path is not None and persistent_shortcuts_path is not None and shortcuts_path != persistent_shortcuts_path:
        raise ValueError("You cannot specify different shortcuts_path and persistent_shortcuts_path.")
    if knowledge_path is not None and persistent_knowledge_path is not None and knowledge_path != persistent_knowledge_path:
        raise ValueError("You cannot specify different knowledge_path and persistent_knowledge_path.")
    
    if shortcuts_path:
        initial_shortcuts = json.load(open(shortcuts_path, "r")) # load agent collected shortcuts
    elif persistent_shortcuts_path:
        initial_shortcuts = json.load(open(persistent_shortcuts_path, "r"))
    else:
        initial_shortcuts = INIT_SHORTCUTS
    print("INFO: Initial shortcuts:", initial_shortcuts)
        
    if knowledge_path:
        additional_knowledge = open(knowledge_path, "r").read() # load agent updated knowledge
    elif persistent_knowledge_path:
        additional_knowledge = open(persistent_knowledge_path, "r").read()
    else:
        additional_knowledge = INIT_TIPS # user provided initial knowledge
    print("INFO: Initial knowledge:", additional_knowledge)

    info_pool = InfoPool(
        instruction = instruction,
        shortcuts = initial_shortcuts,
        additional_knowledge = additional_knowledge, # initial knowledge from user
        future_tasks = future_tasks,
    )

    ### temp dir ###
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    else:
        shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)
    if not os.path.exists(SCREENSHOT_DIR):
        os.mkdir(SCREENSHOT_DIR)

    ### Init Agents ###
    if perceptor is None:
        # if perceptor is not initialized, create the perceptor
        perceptor = Perceptor(ADB_PATH, perception_args=perception_args)
    manager = Manager()
    executor = Executor(adb_path=ADB_PATH)
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    knowledge_reflector = KnowledgeReflector()

    # save initial knowledge and shortcuts
    with open(local_knowledge_save_path, "w") as f:
        f.write(additional_knowledge)
    with open(local_shortcuts_save_path, "w") as f:
        json.dump(initial_shortcuts, f)

    ### Start the agent ###
    steps = []
    steps.append({
        "step": 0,
        "operation": "init",
        "instruction": instruction,
        "task_id": task_id,
        "run_name": run_name,
        "max_itr": max_itr,
        "future_tasks": future_tasks,
        "log_root": log_root,
        "knowledge_path": knowledge_path,
        "shortcuts_path": shortcuts_path,
        "persistent_knowledge_path": persistent_knowledge_path,
        "persistent_shortcuts_path": persistent_shortcuts_path,
        "reset_phone_state": reset_phone_state,
        "perception_args": perception_args,
        "init_info_pool": asdict(info_pool),
    })

    iter = 0
    while True:
        iter += 1

        ## max iteration stop ##
        if max_itr is not None and iter > max_itr:
            print("Max iteration reached. Stopping...")
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "max_iteration",
                "final_info_pool": asdict(info_pool),
            })
            return

        ## do perception if (1) the first perception; (2) previous action has error ##
        if iter == 1 or info_pool.action_outcomes[-1] in ["B", "C"]:
            screenshot_file = "./screenshot/screenshot.jpg"
            print("\n### Perceptor ... ###\n")
            perception_infos, width, height = perceptor.get_perception_infos(screenshot_file, temp_file=TEMP_DIR)
            shutil.rmtree(TEMP_DIR)
            os.mkdir(TEMP_DIR)
            
            keyboard = False
            keyboard_height_limit = 0.9 * height
            for perception_info in perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    keyboard = True
                    break
            
            info_pool.width = width
            info_pool.height = height

            ## log ##
            Image.open(screenshot_file).save(f"{log_dir}/screenshots/{iter}.jpg")
            steps.append({
                "step": iter,
                "operation": "perception",
                "screenshot": f"screenshots/{iter}.jpg",
                "perception_infos": perception_infos,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            ##
        ## otherwise reuse the previous perception info of the updated state ##
        
        ### get perception infos ###
        info_pool.perception_infos_pre = perception_infos
        info_pool.keyboard_pre = keyboard

        ### Manager: High-level Planning ###
        print("\n### Manager ... ###\n")
        ## check if stuck with errors for a long time ##
        # if so need to think about the plan again
        info_pool.error_flag_plan = False
        if len(info_pool.action_outcomes) > 3:
            # check if the last 3 actions are all errors
            latest_outcomes = info_pool.action_outcomes[-3:]
            count = 0
            for outcome in latest_outcomes:
                if outcome in ["B", "C"]:
                    count += 1
            if count == 3:
                info_pool.error_flag_plan = True
        ## 
        info_pool.prev_subgoal = info_pool.current_subgoal

        prompt_planning = manager.get_prompt(info_pool)
        chat_planning = manager.init_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning, image=screenshot_file)
        output_planning = inference_chat(chat_planning, REASONING_MODEL, OPENAI_API_URL, OPENAI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL)
        parsed_result_planning = manager.parse_response(output_planning)
        
        info_pool.plan = parsed_result_planning['plan']
        info_pool.current_subgoal = parsed_result_planning['current_subgoal']

        ## log ##
        steps.append({
            "step": iter,
            "operation": "planning",
            "prompt_planning": prompt_planning,
            "thought": parsed_result_planning['thought'],
            "plan": parsed_result_planning['plan'],
            "current_subgoal": parsed_result_planning['current_subgoal'],
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        pdb_hook()
        ###

        ### Knowledge Reflection: Periodicly Update Knowledge / Creating Shortcuts for Self-Evolving ###
        # if a subgoal is completed and every EVOLVE_FREQ iterations, or at the end of each task, update the knowledge
        if len(info_pool.action_outcomes) > 0:
            if (info_pool.action_outcomes[-1] == "A" and iter % EVOLVE_FREQ == 0) or ("Finished" in info_pool.current_subgoal.strip()):
                print("\n### Knowledge Reflector ... ###\n")
                print("Updating knowledge...")
                prompt_knowledge = knowledge_reflector.get_prompt(info_pool)
                chat_knowledge = knowledge_reflector.init_chat()
                chat_knowledge = add_response("user", prompt_knowledge, chat_knowledge, image=None)
                output_knowledge = inference_chat(chat_knowledge, REASONING_MODEL, OPENAI_API_URL, OPENAI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL)
                parsed_result_knowledge = knowledge_reflector.parse_response(output_knowledge)
                new_shortcut_str, updated_tips = parsed_result_knowledge['new_shortcut'], parsed_result_knowledge['updated_tips']
                if new_shortcut_str != "None" and new_shortcut_str is not None:
                    knowledge_reflector.add_new_shortcut(new_shortcut_str, info_pool)
                info_pool.additional_knowledge = updated_tips
                steps.append({
                    "step": iter,
                    "operation": "knowledge_reflection",
                    "prompt_knowledge": prompt_knowledge,
                    "new_shortcut": new_shortcut_str,
                    "updated_tips": updated_tips,
                })
                ## save the updated knowledge and shortcuts ##
                with open(local_knowledge_save_path, "w") as f:
                    f.write(info_pool.additional_knowledge)
                with open(local_shortcuts_save_path, "w") as f:
                    json.dump(info_pool.shortcuts, f)
                pdb_hook()
        
        ### Stopping by planner ###
        if "Finished" in info_pool.current_subgoal.strip():
            info_pool.finish_thought = parsed_result_planning['thought']
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "success",
                "final_info_pool": asdict(info_pool),
            })
            finish(
                info_pool,
                persistent_knowledge_path = persistent_knowledge_path,
                persistent_shortcuts_path = persistent_shortcuts_path
            )
            return

        ### Executor: Action Decision ###
        print("\n### Executor ... ###\n")
        prompt_action = executor.get_prompt(info_pool)
        chat_action = executor.init_chat()
        chat_action = add_response("user", prompt_action, chat_action, image=screenshot_file)
        output_action = inference_chat(chat_action, REASONING_MODEL, OPENAI_API_URL, OPENAI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL)
        parsed_result_action = executor.parse_response(output_action)
        action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']

        info_pool.last_action_thought = action_thought
        ## execute the action ##
        action_object, num_atomic_actions_executed, shortcut_error_message = executor.execute(action_object_str, info_pool, 
                        screenshot_file=screenshot_file, 
                        ocr_detection=perceptor.ocr_detection,
                        ocr_recognition=perceptor.ocr_recognition,
                        thought = action_thought,
                        )
        if action_object is None:
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "abnormal",
                "final_info_pool": asdict(info_pool),
            })
            finish(
                info_pool, 
                persistent_knowledge_path = persistent_knowledge_path,
                persistent_shortcuts_path = persistent_shortcuts_path
            ) # 
            print("WARNING!!: Abnormal finishing:", action_object_str)
            return

        info_pool.last_action = action_object
        info_pool.last_summary = action_description
        
        
        ## log ##
        steps.append({
            "step": iter,
            "operation": "action",
            "prompt_action": prompt_action,
            "action_object": action_object,
            "action_object_str": action_object_str,
            "action_thought": action_thought,
            "action_description": action_description
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        pdb_hook()
        
        print("\n### Perceptor ... ###\n")
        ## perception on the next step ##
        last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = "./screenshot/last_screenshot.jpg"
        last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)
        
        perception_infos, width, height = perceptor.get_perception_infos(screenshot_file, temp_file=TEMP_DIR)
        shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)
        
        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break
        
        info_pool.perception_infos_post = perception_infos
        info_pool.keyboard_post = keyboard
        assert width == info_pool.width and height == info_pool.height # assert the screen size not changed

        ## log ##
        Image.open(screenshot_file).save(f"{log_dir}/screenshots/{iter+1}.jpg")
        steps.append({
            "step": iter+1,
            "operation": "perception",
            "screenshot": f"{log_dir}/screenshots/{iter+1}.jpg",
            "perception_infos": perception_infos,
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        pdb_hook()
        ##

        print("\n### Action Reflector ... ###\n")
        ### Action Reflection: Check whether the action works as expected ###
        prompt_action_reflect = action_reflector.get_prompt(info_pool)
        chat_action_reflect = action_reflector.init_chat()
        chat_action_reflect = add_response_two_image("user", prompt_action_reflect, chat_action_reflect, [last_screenshot_file, screenshot_file])
        output_action_reflect = inference_chat(chat_action_reflect, REASONING_MODEL, OPENAI_API_URL, OPENAI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL)
        parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
        outcome, error_description, progress_status = (
            parsed_result_action_reflect['outcome'], 
            parsed_result_action_reflect['error_description'], 
            parsed_result_action_reflect['progress_status']
        )
        info_pool.progress_status_history.append(progress_status)

        if "A" in outcome: # Successful. The result of the last action meets the expectation.
            action_outcome = "A"
        elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
            action_outcome = "B"
            # check how many backs to take
            action_name = action_object['name']
            if action_name in ATOMIC_ACTION_SIGNITURES:
                back(ADB_PATH) # back one step for atomic actions
            elif action_name in info_pool.shortcuts:
                # shortcut_object = info_pool.shortcuts[action_name]
                # num_of_atomic_actions = len(shortcut_object['atomic_action_sequence'])
                if shortcut_error_message is not None:
                    error_description += f"; Error occured while executing the shortcut: {shortcut_error_message}"
                for _ in range(num_atomic_actions_executed):
                    back(ADB_PATH)   
            else:
                raise ValueError("Invalid action name:", action_name)

        elif "C" in outcome: # Failed. The last action produces no changes.
            action_outcome = "C"
        else:
            raise ValueError("Invalid outcome:", outcome)
        
        # update action history
        info_pool.action_history.append(action_object)
        info_pool.summary_history.append(action_description)
        info_pool.action_outcomes.append(action_outcome)
        info_pool.error_descriptions.append(error_description)
        info_pool.progress_status = progress_status

        ## log ##
        steps.append({
            "step": iter+1,
            "operation": "action_reflection",
            "prompt_action_reflect": prompt_action_reflect,
            "outcome": outcome,
            "error_description": error_description,
            "progress_status": progress_status,
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        pdb_hook()
        ##
        
        ### NoteTaker: Record Important Content ###
        if action_outcome == "A":
            print("\n### NoteKeeper ... ###\n")
            # if previous action is successful, record the important content
            prompt_note = notetaker.get_prompt(info_pool)
            chat_note = notetaker.init_chat()
            chat_note = add_response("user", prompt_note, chat_note, image=screenshot_file) # new screenshot
            note_output = inference_chat(chat_note, REASONING_MODEL, OPENAI_API_URL, OPENAI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL)
            parsed_result_note = notetaker.parse_response(note_output)
            important_notes = parsed_result_note['important_notes']
            info_pool.important_notes = important_notes
            os.remove(last_screenshot_file)
        
            steps.append({
                "step": iter+1,
                "operation": "notetaking",
                "prompt_note": prompt_note,
                "important_notes": important_notes,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

        elif action_outcome == "B":
            # # backed to the previous state, update perception back to the previous state
            # perception_infos = copy.deepcopy(last_perception_infos)
            # keyboard = last_keyboard
            # os.remove(screenshot_file) # remove the post screenshot
            # os.rename(last_screenshot_file, screenshot_file) # rename the last screenshot to the current screenshot
            os.remove(last_screenshot_file)
            # redo the perception at next step

        elif action_outcome == "C":
            os.remove(last_screenshot_file)

        
        pdb_hook()
        from time import sleep
        print("sleeping for 5 before next iteration")
        sleep(5)
        #


###################################################################################################

### example tasks_json ###
EXAMPLE_TASKS_JSON = [
    {
        "instruction": "Find the current weather in my city.",
        "category": "information_research_and_summarization",
        "type": "single_app",
        "apps": [
            "Google",
        ]
    },
    {
        "instruction": "Search for the top 3 trending news topics on X and write a short summary in Notes.",
        "type": "multi_app",
        "category": "information_research_and_summarization",
        "apps": [
            "X",
            "Notes"
        ]
    },
]


### For debugging ###
ENABLE_PDB = False
def pdb_hook():
    if ENABLE_PDB:
        import pdb; pdb.set_trace()
    else:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_root", type=str, default="log/agent_E")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--tasks_json", type=str, default=None)
    parser.add_argument("--specified_knowledge_path", type=str, default=None)
    parser.add_argument("--specified_shortcuts_path", type=str, default=None)
    parser.add_argument("--setting", type=str, default="individual") # individual or curriculum
    parser.add_argument("--future_tasks_visible", action="store_true", default=False)
    parser.add_argument("--reset_phone_state", action="store_true", default=True)
    parser.add_argument("--max_itr", type=str, default=None)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.instruction is None and args.tasks_json is None:
        raise ValueError("You must provide either instruction or tasks_json.")
    if args.instruction is not None and args.tasks_json is not None:
        raise ValueError("You cannot provide both instruction and tasks_json.")
    
    if args.instruction is not None:
        # single task inference
        run_single_task(
            args.instruction,
            run_name=args.run_name,
            log_root=args.log_root,
            knowledge_path=args.specified_knowledge_path,
            shortcuts_path=args.specified_shortcuts_path,
            persistent_knowledge_path=None,
            persistent_shortcuts_path=None,
            reset_phone_state=True,
            perceptor=None,
            perception_args=DEFAULT_PERCEPTION_ARGS,
            max_itr=args.max_itr
        )
    else:
        # multi task inference
        tasks = json.load(open(args.tasks_json, "r"))
        perceptor = Perceptor(ADB_PATH, perception_args=DEFAULT_PERCEPTION_ARGS)
        if args.setting == "individual":
            ## invidual setting ##
            persistent_knowledge_path = None
            persistent_shortcuts_path = None
        elif args.setting == "curriculum":
            ## curriculum setting ##
            run_log_dir = f"{args.log_root}/{args.run_name}"
            os.makedirs(run_log_dir, exist_ok=True)
            persistent_knowledge_path = os.path.join(run_log_dir, "persistent_knowledge.txt")
            persistent_shortcuts_path = os.path.join(run_log_dir, "persistent_shortcuts.json")

            if args.specified_knowledge_path is not None:
                shutil.copy(args.specified_knowledge_path, persistent_knowledge_path)
            else:
                with open(persistent_knowledge_path, "w") as f:
                    f.write(INIT_TIPS)
            
            if args.specified_shortcuts_path is not None:
                shutil.copy(args.specified_shortcuts_path, persistent_shortcuts_path)
            else:
                with open(persistent_shortcuts_path, "w") as f:
                    json.dump(INIT_SHORTCUTS, f)
        else:
            raise ValueError("Invalid setting:", args.setting)
        
        print(f"INFO: Running tasks from {args.tasks_json} using {args.setting} setting ...")
        for i, task in enumerate(tasks):
            ## if future tasks are visible, specify them in the args ##
            if args.future_tasks_visible and i < len(tasks) - 1 and args.setting == "curriculum":
                future_tasks = [t['instruction'] for t in tasks[i+1:]]
            else:
                future_tasks = []

            print("\n\n### Running on task:", task["instruction"])
            print("\n\n")
            instruction = task["instruction"]
            if "task_id" in task:
                task_id = task["task_id"]
            else:
                task_id = args.tasks_json.split("/")[-1].split(".")[0] + f"_{args.setting}" + f"_{i}"
            run_single_task(
                instruction,
                future_tasks=future_tasks,
                log_root=args.log_root,
                run_name=args.run_name,
                task_id=task_id,
                knowledge_path=args.specified_knowledge_path,
                shortcuts_path=args.specified_shortcuts_path,
                persistent_knowledge_path=persistent_knowledge_path,
                persistent_shortcuts_path=persistent_shortcuts_path,
                reset_phone_state=True,
                perceptor=perceptor,
                perception_args=DEFAULT_PERCEPTION_ARGS,
                max_itr=args.max_itr
            )
            import time
            print("DONE:", task["instruction"])
            print("Sleeping for 5 seconds before next task ...")
            time.sleep(5)

if __name__ == "__main__":
    main()

