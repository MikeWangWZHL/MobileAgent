import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw

from MobileAgentE.api import inference_chat
from MobileAgentE.text_localization import ocr
from MobileAgentE.icon_localization import det
from MobileAgentE.controller import get_screenshot, back, clear_background_and_back_to_home
from MobileAgentE.agents import InfoPool, Manager, Executor, Notetaker, ActionReflector, KnowledgeReflector, INIT_SHORTCUTS
from MobileAgentE.agents import add_response, add_response_two_image, finish
from MobileAgentE.agents import ATOMIC_ACTION_SIGNITURES
# from MobileAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
# from MobileAgentE.prompt_v2 import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
# from MobileAgentE.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image

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
# Your ADB path
adb_path = "/Users/wangz3/Desktop/vlm_agent_project/platform-tools/adb"

# Your instruction
# instruction = "Find the best reviewed Korean restaurant in my area that is within 10min drive and opens late until 10pm. Provide a list of top 3 restaurants with their addresses and phone numbers."
# instruction = "Find the best reviewed Korean restaurant in my area that is within 10min drive and opens late until 10pm."
instruction = "Find a good bargin for a used iphone 16 on ebay. The phone should be in good condition and unlocked. Put at least three sharable links to Notes for me to review."
run_name = "test"

# Your GPT-4o API URL
API_url = "https://api.openai.com/v1/chat/completions"

# Your GPT-4o API Token
# token = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school", "r").read()
# token = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school_ecole", "r").read()
token = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_taobao", "r").read()

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
caption_call_method = "api"

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
caption_model = "qwen-vl-plus"

# If you choose the api caption call method, input your Qwen api here
qwen_api = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/qwen_key_mine", "r").read()
# qwen_api = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/qwen_key_from_xi", "r").read()

# # You can add operational knowledge to help Agent operate more accurately.
# user_provided_knowledge = """1.	To open an app, use the action \"Open_App\".
# 2.	To exit an app, use the action \"Home\".
# 3.	To enter new text in an input box with existing text, clear it first by tapping the \"X\" button.
# """
user_provided_knowledge = """If you want to type new text in a search box that already contains text you previously entered, make sure to clear it first by tapping the 'X' button."""

# existing shortcuts
shortcuts_path = None # shortcuts.json file create by the agent from previous runs

# existing knowledge
knowledge_path = None # knowledge.txt file create by the agent from previous runs

# Reasoning GPT model
# reasoning_model_name = "gpt-4o"
reasoning_model_name = "gpt-4o-2024-11-20"

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


# def merge_text_blocks(text_list, coordinates_list):


#     merged_text_blocks = []
#     merged_coordinates = []

#     sorted_indices = sorted(range(len(coordinates_list)), key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
#     sorted_text_list = [text_list[i] for i in sorted_indices]
#     sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

#     num_blocks = len(sorted_text_list)
#     merge = [False] * num_blocks

#     for i in range(num_blocks):
#         if merge[i]:
#             continue
        
#         anchor = i
        
#         group_text = [sorted_text_list[anchor]]
#         group_coordinates = [sorted_coordinates_list[anchor]]

#         for j in range(i+1, num_blocks):
#             if merge[j]:
#                 continue

#             if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
#             sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
#             abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
#                 group_text.append(sorted_text_list[j])
#                 group_coordinates.append(sorted_coordinates_list[j])
#                 merge[anchor] = True
#                 anchor = j
#                 merge[anchor] = True

#         merged_text = "\n".join(group_text)
#         min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
#         min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
#         max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
#         max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

#         merged_text_blocks.append(merged_text)
#         merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

#     return merged_text_blocks, merged_coordinates

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


def get_perception_infos(adb_path, screenshot_file):
    get_screenshot(adb_path)
    
    width, height = Image.open(screenshot_file).size
    
    text, coordinates = ocr(screenshot_file, ocr_detection, ocr_recognition)
    text, coordinates = merge_text_blocks(text, coordinates)
    
    center_list = [[(coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2] for coordinate in coordinates]
    draw_coordinates_on_image(screenshot_file, center_list)
    
    perception_infos = []
    for i in range(len(coordinates)):
        perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
        perception_infos.append(perception_info)
        
    coordinates = det(screenshot_file, "icon", groundingdino_model)
    
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
        crop(screenshot_file, image_box[i], image_id[i])

    images = get_all_files_in_folder(temp_file)
    if len(images) > 0:
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
        icon_map = {}
        prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
        if caption_call_method == "local":
            for i in range(len(images)):
                image_path = os.path.join(temp_file, images[i])
                icon_width, icon_height = Image.open(image_path).size
                if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                    des = "None"
                else:
                    des = generate_local(tokenizer, model, image_path, prompt)
                icon_map[i+1] = des
        else:
            for i in range(len(images)):
                images[i] = os.path.join(temp_file, images[i])
            icon_map = generate_api(images, prompt)
        for i, j in zip(image_id, range(1, len(image_id)+1)):
            if icon_map.get(j):
                perception_infos[i]['text'] = "icon: " + icon_map[j]

    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
        
    return perception_infos, width, height

###################################################################################################

### Load caption model ###
device = "cuda"
torch.manual_seed(1234)
if caption_call_method == "local":
    if caption_model == "qwen-vl-chat":
        model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    elif caption_model == "qwen-vl-chat-int4":
        qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
        model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True,use_safetensors=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True, do_sample=False)
    else:
        print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
        exit(0)
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
elif caption_call_method == "api":
    pass
else:
    print("You must choose the caption model call function from \"local\" and \"api\"")
    exit(0)


### Load ocr and icon detection model ###
groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
# ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_detection = pipeline(Tasks.ocr_detection, model='iic/cv_resnet18_ocr-detection-db-line-level_damo') # dbnet (no tensorflow)
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='iic/cv_convnextTiny_ocr-recognition-document_damo')

print("DEBUG: Perception models loaded successfully.")



### reset the phone status ###
print("DEBUG: Reseting the phone status...\n")
# ask user if they want to reset the phone status
if input("Do you want to reset the phone status? (y/n): ") == "y":
    clear_background_and_back_to_home(adb_path)


def pdb_hook():
    # import pdb; pdb.set_trace()
    pass


### log dir ###
unique_id = time.strftime("%Y%m%d-%H%M%S")
log_dir = f"log/agent_E/{run_name}/{unique_id}"
os.makedirs(f"{log_dir}/screenshots", exist_ok=True)
log_json_path = f"{log_dir}/steps.json"
knowledge_save_path = f"{log_dir}/knowledge.txt"
shortcuts_save_path = f"{log_dir}/shortcuts.json"

### Init Information Pool ###
if shortcuts_path:
    initial_shortcuts  = json.load(open(shortcuts_path, "r")) # load agent collected shortcuts
else:
    initial_shortcuts = INIT_SHORTCUTS

if knowledge_path:
    additional_knowledge = open(knowledge_path, "r").read() # load agent updated knowledge
else:
    additional_knowledge = user_provided_knowledge # user provided initial knowledge

info_pool = InfoPool(
    instruction=instruction,
    shortcuts = initial_shortcuts,
    additional_knowledge = additional_knowledge, # initial knowledge from user
)

### temp dir ###
temp_file = "temp"
screenshot = "screenshot"
if not os.path.exists(temp_file):
    os.mkdir(temp_file)
else:
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)
if not os.path.exists(screenshot):
    os.mkdir(screenshot)

### Init Agents ###
manager = Manager()
executor = Executor(adb_path=adb_path)
notetaker = Notetaker()
action_reflector = ActionReflector()
knowledge_reflector = KnowledgeReflector()

# save initial knowledge and shortcuts
with open(knowledge_save_path, "w") as f:
    f.write(additional_knowledge)
with open(shortcuts_save_path, "w") as f:
    json.dump(initial_shortcuts, f)

### Start the agent ###
steps = []
steps.append({
    "step": 0,
    "step_type": "input",
    "instruction": instruction,
    "shortcuts": initial_shortcuts,
    "knowledge": additional_knowledge,
    "run_name": run_name,
})



iter = 0
while True:
    iter += 1
    if iter == 1:
        ## first perception ##
        screenshot_file = "./screenshot/screenshot.jpg"
        perception_infos, width, height = get_perception_infos(adb_path, screenshot_file)
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
    
    ### get perception infos ###
    info_pool.perception_infos_pre = perception_infos
    info_pool.keyboard_pre = keyboard

    ### Manager: High-level Planning ###

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
    output_planning = inference_chat(chat_planning, reasoning_model_name, API_url, token)
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

    ### Stopping by planner ###
    if "Finished" in info_pool.current_subgoal.strip():
        info_pool.finish_thought = parsed_result_planning['thought']
        finish(info_pool)


    ### Knowledge Reflection: Periodicly Update Knowledge / Creating Shortcuts for Self-Evolving ###
    # if a subgoal is completed and every 5 iterations, update the knowledge
    knowledge_accum_k = 5
    if len(info_pool.action_outcomes) > 0:
        if info_pool.action_outcomes[-1] == "A" and iter % knowledge_accum_k == 0:
            print("Updating knowledge...")
            prompt_knowledge = knowledge_reflector.get_prompt(info_pool)
            chat_knowledge = knowledge_reflector.init_chat()
            chat_knowledge = add_response("user", prompt_knowledge, chat_knowledge, image=None)
            output_knowledge = inference_chat(chat_knowledge, reasoning_model_name, API_url, token)
            parsed_result_knowledge = knowledge_reflector.parse_response(output_knowledge)
            new_shortcut_str, updated_tips = parsed_result_knowledge['new_shortcut'], parsed_result_knowledge['updated_tips']
            if new_shortcut_str != "None" and new_shortcut_str is not None:
                knowledge_reflector.add_new_shortcut(new_shortcut_str, info_pool)
            info_pool.additional_knowledge = updated_tips

            ## save the updated knowledge and shortcuts ##
            with open(knowledge_save_path, "w") as f:
                f.write(info_pool.additional_knowledge)
            with open(shortcuts_save_path, "w") as f:
                json.dump(info_pool.shortcuts, f)
            pdb_hook()

    ### Executor: Action Decision ###
    prompt_action = executor.get_prompt(info_pool)
    chat_action = executor.init_chat()
    chat_action = add_response("user", prompt_action, chat_action, image=screenshot_file)
    output_action = inference_chat(chat_action, reasoning_model_name, API_url, token)
    parsed_result_action = executor.parse_response(output_action)
    action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']

    info_pool.last_action_thought = action_thought
    ## execute the action ##
    action_object = executor.execute(action_object_str, info_pool, 
                     screenshot_file=screenshot_file, 
                     ocr_detection=ocr_detection,
                     ocr_recognition=ocr_recognition,
                     thought = action_thought,
                     )
    if action_object is None:
        raise ValueError("Failed to parse and execute the action:", action_object_str)

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
    
    ## perception on the next step ##
    last_perception_infos = copy.deepcopy(perception_infos)
    last_screenshot_file = "./screenshot/last_screenshot.jpg"
    last_keyboard = keyboard
    if os.path.exists(last_screenshot_file):
        os.remove(last_screenshot_file)
    os.rename(screenshot_file, last_screenshot_file)
    
    perception_infos, width, height = get_perception_infos(adb_path, screenshot_file)
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)
    
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

    ### Action Reflection: Check whether the action works as expected ###
    prompt_action_reflect = action_reflector.get_prompt(info_pool)
    chat_action_reflect = action_reflector.init_chat()
    chat_action_reflect = add_response_two_image("user", prompt_action_reflect, chat_action_reflect, [last_screenshot_file, screenshot_file])
    output_action_reflect = inference_chat(chat_action_reflect, reasoning_model_name, API_url, token)
    parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
    outcome, error_description, progress_status = (
        parsed_result_action_reflect['outcome'], 
        parsed_result_action_reflect['error_description'], 
        parsed_result_action_reflect['progress_status']
    )

    if "A" in outcome: # Successful. The result of the last action meets the expectation.
        action_outcome = "A"
    elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
        action_outcome = "B"
        # check how many backs to take
        action_name = action_object['name']
        if action_name in ATOMIC_ACTION_SIGNITURES:
            back(adb_path) # back one step for atomic actions
        elif action_name in info_pool.shortcuts:
            shortcut_object = info_pool.shortcuts[action_name]
            num_of_atomic_actions = len(shortcut_object['atomic_action_sequence'])
            for _ in range(num_of_atomic_actions):
                back(adb_path)
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
        # if previous action is successful, record the important content
        prompt_note = notetaker.get_prompt(info_pool)
        chat_note = notetaker.init_chat()
        chat_note = add_response("user", prompt_note, chat_note, image=screenshot_file) # new screenshot
        note_output = inference_chat(chat_note, reasoning_model_name, API_url, token)
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
        # backed to the previous state, update perception back to the previous state
        perception_infos = copy.deepcopy(last_perception_infos)
        keyboard = last_keyboard
        os.remove(screenshot_file) # remove the post screenshot
        os.rename(last_screenshot_file, screenshot_file) # rename the last screenshot to the current screenshot

    elif action_outcome == "C":
        pass # do nothing

    
    pdb_hook()
    from time import sleep
    print("sleeping for 5 before next iteration")
    sleep(5)
    #