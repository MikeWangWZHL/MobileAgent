from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from MobileAgentE.api import encode_image
from MobileAgentE.controller import tap, swipe, type, back, home, switch_app, enter
from MobileAgentE.text_localization import ocr
import copy
import re
import json
import time

### Helper Functions ###

def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
            "type": "text", 
            "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history


def add_response_two_image(role, prompt, chat_history, image):
    new_chat_history = copy.deepcopy(chat_history)

    base64_image1 = encode_image(image[0])
    base64_image2 = encode_image(image[1])
    content = [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}"
            }
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}"
            }
        },
    ]

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1) + "\n")
    print("*"*100)


def extract_json_object(text, json_type="dict"):
    """
    Extracts a JSON object from a text string.

    Parameters:
    - text (str): The text containing the JSON data.
    - json_type (str): The type of JSON structure to look for ("dict" or "list").

    Returns:
    - dict or list: The extracted JSON object, or None if parsing fails.
    """
    try:
        # Try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # Not a valid JSON, proceed to extract from text

    # Define patterns for extracting JSON objects or arrays
    json_pattern = r"({.*?})" if json_type == "dict" else r"(\[.*?\])"

    # Search for JSON enclosed in code blocks first
    code_block_pattern = r"```json\s*(.*?)\s*```"
    code_block_match = re.search(code_block_pattern, text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Failed to parse JSON inside code block

    # Fallback to searching the entire text
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue  # Try the next match

    # If all attempts fail, return None
    return None

########################


@dataclass
class InfoPool:
    """Keeping track of all information across the agents."""

    # User input / accumulated knowledge
    instruction: str = ""
    additional_knowledge: str = ""
    shortcuts: dict = field(default_factory=dict)

    # Perception
    width: int = 1080
    height: int = 2340
    clickable_infos_pre: list = field(default_factory=list) # List of clickable elements pre action
    keyboard_pre: bool = False # keyboard status pre action
    clickable_infos_post: list = field(default_factory=list) # List of clickable elements post action
    keyboard_post: bool = False # keyboard status post action

    # Working memory
    summary_history: list = field(default_factory=list)  # List of action descriptions
    action_history: list = field(default_factory=list)  # List of actions
    last_summary: str = ""  # Last action description
    last_action: str = ""  # Last action
    important_notes: str = ""
    error_flag_plan: bool = False
    error_flags_action: list = field(default_factory=list)
    error_descriptions: list = field(default_factory=list)

    # Planning
    plan: str = ""
    progress_status: str = ""
    current_subgoal: str = ""


class BaseAgent(ABC):
    @abstractmethod
    def init_chat(self) -> list:
        pass
    @abstractmethod
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    @abstractmethod
    def parse_response(self, response: str) -> dict:
        pass



class Manager(BaseAgent):

    def init_chat(self):
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to track progress and devise high-level plans to achieve the user's requests. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        if info_pool.plan == "":
            # first time planning
            prompt += "---\n"
            prompt += "Make a plan to achieve the user's instruction. If the request is complex, break it down into subgoals. If the request involves exploration, include concrete subgoals to quantify the necessary research.\n\n"
            prompt += "Provide your output in the following format which contains three parts:\n"
            prompt += "### Thought ###\n"
            prompt += "A detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "1. first subgoal\n"
            prompt += "2. second subgoal\n"
            prompt += "...\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The first subgoal you should work on.\n\n"
        else:
            # continue planning
            prompt += "### Current Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"
            prompt += "### Previous Subgoal ###\n"
            prompt += f"{info_pool.current_subgoal}\n\n"
            prompt += f"### Progress Status ###\n"
            prompt += f"{info_pool.progress_status}\n\n"
            prompt += "### Important Notes ###\n"
            if info_pool.important_notes != "":
                prompt += f"{info_pool.important_notes}\n\n"
            else:
                prompt += "No important notes recorded.\n\n"
            if info_pool.error_flag_plan:
                prompt += "### Error Description ###\n"
                prompt += f"{info_pool.error_description}\n\n"
            prompt += "---\n"
            prompt += "The sections above provide an overview of the plan you are following, the current subgoal you are working on, the overall progress made, and any important notes you have recorded.\n"
            prompt += "Carefully assess the current status to determine if the task has been fully completed. If the user's request involves exploration, ensure you have conducted sufficient investigation. If you are confident that no further actions are required, mark the task as \"Finished\" in your output. If the task is not finished, outline the next steps. If an \"Error Description\" is provided, think step by step about how to revise the plan to address the error.\n\n"

            prompt += "Provide your output in the following format, which contains three parts:\n"
            prompt += "### Thought ###\n"
            prompt += "Provide a detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "Include your current plan if no updates are needed. If an update is required, provide the updated plan here.\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "List the next subgoal to work on. If the previous subgoal is not yet complete, copy it here. If all subgoals are completed, write \"Finished.\"\n"
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "plan": plan, "current_subgoal": current_subgoal}


# name: {arguments: [argument_keys], description: description}
ATOMIC_ACTION_SIGNITURES = {
    "Open_App": {
        "arguments": ["app_name"],
        "description": lambda info: "If the current screen is Home or App screen, you can use this action to open the app named \"app_name\" on the visible on the current screen."
    },
    "Tap": {
        "arguments": ["x", "y"],
        "description": lambda info: "Tap the position (x, y) in current screen."
    },
    "Swipe": {
        "arguments": ["x1", "y1", "x2", "y2"],
        "description": lambda info: f"Swipe from position (x1, y1) to position (x2, y2). To swipe up or down to review more content, you can adjust the y-coordinate offset based on the desired scroll distance. For example, setting x1 = x2 = {int(0.5 * info.width)}, y1 = {int(0.5 * info.height)}, and y2 = {int(0.1 * info.height)} will swipe upwards to review additional content below. To swipe left or right in the App switcher screen to choose between open apps, set the x-coordinate offset to at least {int(0.5 * info.width)}."
    },
    "Type": {
        "arguments": ["text"],
        "description": lambda info: "Type the \"text\" in an input box."
    },
    "Enter": {
        "arguments": [],
        "description": lambda info: "Press the Enter key."
    },
    "Switch_App": {
        "arguments": [],
        "description": lambda info: "Show the App switcher for switching between opened apps."
    },
    "Back": {
        "arguments": [],
        "description": lambda info: "Return to the previous state."
    },
    "Home": {
        "arguments": [],
        "description": lambda info: "Return to home page."
    }
}

INIT_SHORTCUTS = {
    "Tap_Type_and_Enter": {
        "name": "Tap_Type_and_Enter",
        "arguments": ["x", "y", "text"],
        "description": "Tap an input box at position (x, y), Type the \"text\", and then perform the Enter operation (useful for searching or sending messages).",
        "precondition": "There is a text input box on the screen.",
        "atomic_action_sequence":[
            {"name": "Tap", "arguments_map": {"x":"x", "y":"y"}},
            {"name": "Type", "arguments_map": {"text":"text"}},
            {"name": "Enter", "arguments_map": {}}
        ]
    }
}

class Executor(BaseAgent):
    def __init__(self, adb_path):
        self.adb = adb_path

    def init_chat(self):
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Latest Action History ###\n"
        if info_pool.action_history != []:
            prompt += "The most recent actions you took and whether they were successful:\n"
            num_actions = min(5, len(info_pool.action_history))
            latest_actions = info_pool.action_history[-num_actions:]
            latest_summary = info_pool.summary_history[-num_actions:]
            error_flags = info_pool.error_flags_action[-num_actions:]
            error_descriptions = info_pool.error_descriptions[-num_actions:]
            for act, summ, err, err_des in zip(latest_actions, latest_summary, error_flags, error_descriptions):
                if not err:
                    prompt += f"Action: {act} | Description: {summ} | Outcome: Successful\n"
                else:
                    prompt += f"Action: {act} | Description: {summ} | Outcome: Failed | Error Description: {err_des}\n"
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"


        prompt += "### Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better perceive the content in this screenshot, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom)."
        )
        prompt += "The extracted information is as follows:\n"

        for clickable_info in info_pool.clickable_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += (
            "Note that this information might not be entirely accurate. "
            "You should combine it with the screenshot to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Keyboard status ###\n"
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        if info_pool.additional_knowledge != "":
            prompt += "### Tips ###\n"
            prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
            prompt += f"{info_pool.additional_knowledge}\n\n"


        prompt += "---\n"
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice any errors in the previous actions, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions or the shortcuts. The shortcuts are predefined sequences of actions that can be used to speed up the process. Each shortcut has a precondition specifying when it is suitable to use. If you plan to use a shortcut, ensure the current phone state satisfies its precondition first.\n\n"
        prompt += "#### Atomic Actions ####\n"
        prompt += "The atomic action functions are listed in the format of `name(arguments): description` as follows:\n"

        if info_pool.keyboard_pre:
            for action, value in ATOMIC_ACTION_SIGNITURES.items():
                prompt += f"{action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        else:
            for action, value in ATOMIC_ACTION_SIGNITURES.items():
                if "Type" not in action:
                    prompt += f"{action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
            prompt += "\nNOTE: Unable to Type. You cannot use the actions involving \"Type\" because the keyboard has not been activated. If you want to type, please first activate the keyboard by tapping on the input box on the screen.\n"
        
        prompt += "\n"
        prompt += "#### Shortcuts ####\n"
        if info_pool.shortcuts != {}:
            prompt += "The shortcut functions are listed in the format of `name(arguments): description | Precondition: precondition` as follows:\n"
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"{shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are available.\n"
        prompt += "\n"
        prompt += "---\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        prompt += "A detailed explanation of your rationale for the action you choose.\n\n"
        prompt += "### Action ###\n"
        prompt += "Choose only one action or shortcut from the options above. Be sure to fill in all required arguments, such as text and coordinates (e.g., x and y), in the provided fields.\n"
        prompt += "You must provide your decision using a valid JSON format specifying the name and arguments of the action. For example, if you choose to tap at position (100, 200), you should write {\"name\":\"Tap\", \"arguments\":{\"x\":100, \"y\":100}}. If an action does not require arguments, such as Home, fill in null to the \"arguments\" field. Ensure that the argument keys match the action function's signature exactly.\n\n"
        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action and the expected outcome."
        return prompt

    def execute_atomic_action(self, action: str, arguments: dict, **kwargs) -> None:
        adb_path = self.adb
        
        if "Open_App".lower() == action.lower():
            screenshot_file = kwargs["screenshot_file"]
            ocr_detection = kwargs["ocr_detection"]
            ocr_recognition = kwargs["ocr_recognition"]
            app_name = arguments["app_name"].strip()
            text, coordinate = ocr(screenshot_file, ocr_detection, ocr_recognition)
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [int((coordinate[ti][0] + coordinate[ti][2])/2), int((coordinate[ti][1] + coordinate[ti][3])/2)]
                    tap(adb_path, name_coordinate[0], name_coordinate[1]- int(coordinate[ti][3] - coordinate[ti][1]))# 
                    break
            time.sleep(5)
        
        elif "Tap".lower() == action.lower():
            x, y = int(arguments["x"]), int(arguments["y"])
            tap(adb_path, x, y)
            time.sleep(5)
        
        elif "Swipe".lower() == action.lower():
            x1, y1, x2, y2 = int(arguments["x1"]), int(arguments["y1"]), int(arguments["x2"]), int(arguments["y2"])
            swipe(adb_path, x1, y1, x2, y2)
            time.sleep(5)
            
        elif "Type".lower() == action.lower():
            text = arguments["text"]
            type(adb_path, text)
            time.sleep(1)

        elif "Enter".lower() == action.lower():
            enter(adb_path)
            time.sleep(5)

        elif "Back".lower() == action.lower():
            back(adb_path)
            time.sleep(2)
        
        elif "Home".lower() == action.lower():
            home(adb_path)
            time.sleep(2)
        
        elif "Switch_App".lower() == action.lower():
            switch_app(adb_path)
            time.sleep(2)
        
    def execute(self, action_str: str, info_pool: InfoPool, **kwargs) -> None:
        action_object = extract_json_object(action_str)
        if action_object is None:
            print("Error! Invalid JSON for executing action: ", action_str)
            return
        action, arguments = action_object["name"], action_object["arguments"]
        action = action.strip()

        # execute atomic action
        if action in ATOMIC_ACTION_SIGNITURES:
            print("Executing atomic action: ", action, arguments)
            self.execute_atomic_action(action, arguments, **kwargs)
        # execute shortcut
        elif action in info_pool.shortcuts:
            print("Executing shortcut: ", action)
            shortcut = info_pool.shortcuts[action]
            for i, atomic_action in enumerate(shortcut["atomic_action_sequence"]):
                atomic_action_name = atomic_action["name"]
                if atomic_action["arguments_map"] is None or len(atomic_action["arguments_map"]) == 0:
                    atomic_action_args = None
                else:
                    atomic_action_args = {}
                    for atomic_arg_key, short_cut_arg_key in atomic_action["arguments_map"].items():
                        atomic_action_args[atomic_arg_key] = arguments[short_cut_arg_key]
                print(f"\t Executing sub-step {i}:", atomic_action_name, atomic_action_args, "...")
                self.execute_atomic_action(atomic_action_name, atomic_action_args, **kwargs)
        else:
            print("Error! Invalid action name: ", action)
            return

    def parse_response(self, response: str, info_pool: InfoPool) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ", " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}


class ActionReflector(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the last action produced the expected behavior and to keep track of the overall progress."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "---\n"
        prompt += f"The attached two images are two phone screenshots before and after your last action. " 
        prompt += f"The width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        prompt += (
            "To help you better perceive the content in these screenshots, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom).\n"
        )
        prompt += (
            "Note that these information might not be entirely accurate. "
            "You should combine them with the screenshots to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Screen Information Before the Action ###\n"
        for clickable_info in info_pool.clickable_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status before the action: "
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"


        prompt += "### Screen Information After the Action ###\n"
        for clickable_info in info_pool.clickable_infos_post:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status after the action: "
        if info_pool.keyboard_post:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        # assert info_pool.last_action != ""
        prompt += f"Action: {info_pool.last_action}\n"
        prompt += f"Expectation: {info_pool.last_summary}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action was successful, update the progress status accordingly. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\", \"B\" or \"C\":\n"
        prompt += "A: Successful. The result of the last action meets the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

        prompt += "### Progress Status ###\n"
        prompt += "If the action was successful, update the progress status. If the action failed, copy the previous progress status.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Progress Status ###")[0].replace("\n", " ").replace("  ", " ").strip()
        progress_status = response.split("### Progress Status ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"outcome": outcome, "error_description": error_description, "progress_status": progress_status}


class Notetaker(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to take notes of important content relevant to the user's request while navigating different pages."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"

        prompt += "### Existing Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "### Current Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better perceive the content in this screenshot, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom)."
        )
        prompt += "The extracted information is as follows:\n"

        for clickable_info in info_pool.clickable_infos_post:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += (
            "Note that this information might not be entirely accurate. "
            "You should combine it with the screenshot to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the current screen and progress status to determine if there are any important notes that need to be recorded. If you identify any critical information relevant to the user's request that is not yet recorded, make a note of it and update the existing important notes. \n\n"

        prompt += "Provide your output in the following format:\n"
        prompt += "### Important Notes ###\n"
        prompt += "The updated important notes, combining the old and new ones.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        important_notes = response.split("### Important Notes ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"important_notes": important_notes}


SHORTCUT_EXMPALE = """
{
    "name": "Tap_Type_and_Enter",
    "arguments": ["x", "y", "text"],
    "description": "Tap an input box at position (x, y), Type the \"text\", and then perform the Enter operation (useful for searching or sending messages).",
    "precondition": "There is a text input box on the screen.",
    "atomic_action_sequence":[
        {"name": "Tap", "arguments_map": {"x":"x", "y":"y"}},
        {"name": "Type", "arguments_map": {"text":"text"}},
        {"name": "Enter", "arguments_map": {}}
    ]
}
"""

class KnowledgeReflector(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant specializing in mobile phone operations. Your goal is to reflect on past experiences and provide insights to improve future interactions."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### Existing Tips from Past Experience ###\n"
        if info_pool.additional_knowledge != "":
            prompt += f"{info_pool.additional_knowledge}\n\n"
        else:
            prompt += "No tips recorded.\n\n"

        prompt += "### Existing Shortcuts from Past Experience ###\n"
        if info_pool.shortcuts != {}:
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"{shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are provided.\n"

        prompt += "### Atomic Actions ###\n"
        for action, value in ATOMIC_ACTION_SIGNITURES.items():
            prompt += f"{action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"

        prompt += "---\n"
        prompt = "### Current Task ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"

        prompt += "### Full Action History ###\n"
        if info_pool.action_history != []:
            latest_actions = info_pool.action_history
            latest_summary = info_pool.summary_history
            error_flags = info_pool.error_flags_action
            error_descriptions = info_pool.error_descriptions
            for act, summ, err, err_des in zip(latest_actions, latest_summary, error_flags, error_descriptions):
                if not err:
                    prompt += f"Action: {act} | Description: {summ} | Outcome: Successful\n"
                else:
                    prompt += f"Action: {act} | Description: {summ} | Outcome: Failed | Error Description: {err_des}\n"
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "---\n"
        prompt += "Carefully reflect on the interaction history of the current task. Consider the following: (1) Are there any sequences of actions that could be consolidated into new \"shortcuts\" to improve efficiency? These shortcuts are subroutines consisting of a series of atomic actions that can be executed under specific preconditions. (2) Are there any general tips that might be useful for handling future tasks, such as advice on preventing certain common errors?\n\n"

        prompt += "Provide your output in the following format containing two parts:\n"
        prompt += "### New Shortcut ###\n"
        prompt += "If you decide to create a new shortcut (not already in the existing shortcuts), provide your shortcut object in a valid JSON format which is detailed below. If not, put \"None\" here.\n"
        prompt += "A shortcut object contains the following fields: name, arguments, description, precondition, and atomic_action_sequence. The keys in the arguements need to be unique. The atomic_action_sequence is a list of dictionaries, each containing the name of an atomic action and a mapping of its atomic argument names to the shortcut's argument name. If an atomic action in the atomic_action_sequence does not take any arugments, set the `arguments_map` to an empty dict. \n"
        prompt += f"Here is an example of a shortcut object: {SHORTCUT_EXMPALE}\n\n"

        prompt += "### New Tips ###\n"
        prompt += "If you have any important new tips to share (not already in the existing tips), provide them here. If not, write \"None\" here.\n"

    def add_new_shortcut(self, short_cut_str: str, info_pool: InfoPool) -> str:
        short_cut_object = extract_json_object(short_cut_str)
        if short_cut_object is None:
            print("Error! Invalid JSON for adding new shortcut: ", short_cut_str)
            return
        short_cut_name = short_cut_object["name"]
        if short_cut_name in info_pool.shortcuts:
            print("Error! The shortcut already exists: ", short_cut_name)
            return
        info_pool.shortcuts[short_cut_name] = short_cut_object
        print("Updated short_cuts:", info_pool.shortcuts)

    def parse_response(self, response: str) -> dict:
        new_shortcut = extract_json_object(response.split("### New Shortcut ###")[-1].split("### New Tips ###")[0].replace("\n", " ").replace("  ", " ").strip())
        new_tips = response.split("### New Tips ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"new_shortcut": new_shortcut, "new_tips": new_tips}
