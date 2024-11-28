from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from MobileAgentE.api import encode_image
from MobileAgentE.controller import tap, swipe, type, back, home, switch_app
from MobileAgentE.text_localization import ocr
import copy
import re
import json

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
    # TODO: debug to make sure this function works as expected
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
    def __init__(self, adb_path):
        self.adb = adb_path
        
    def init_chat(self):
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to track progress and devise high-level plans to achieve the user's requests. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])

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
            prompt += f"{info_pool.important_notes}\n\n"
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


class Executor(BaseAgent):
    def init_chat(self):
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better perceive the content in this screenshot, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom). The content represents text or an icon description, respectively. "
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
            prompt += "### Hint ###\n"
            prompt += "From previous experience interacting with the device, you have collected the following hints that might be useful for deciding what to do next:\n"
            prompt += f"{info_pool.additional_knowledge}\n\n"

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

        prompt += "---\n"
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice any errors in the previous actions, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions or the shortcuts. The shortcuts are predefined sequences of actions that can be used to speed up the process. Each shortcut has a precondition specifying when it is suitable to use. If you plan to use a shortcut, ensure the current phone state satisfies its precondition first.\n\n"
        prompt += "#### Atomic Actions ####\n"
        prompt += "Open_App(app_name): If the current screen is Home or App screen, you can use this action to open the app named \"app_name\" on the visible on the current screen.\n"
        prompt += "Tap(x, y): Tap the position (x, y) in current screen.\n"
        prompt += f"Swipe(x1, y1, x2, y2): Swipe from position (x1, y1) to position (x2, y2). To swipe up or down to review more content, you can use an approximate offset value for the y-coordinates. For example, setting x1 = x2 = {0.5 * info_pool.width}, y1 = {0.5 * info_pool.height}, and y2 = y1 - {0.4 * info_pool.height} will swipe upwards to review additional content below.\n"
        if info_pool.keyboard_pre:
            prompt += "Type(text): Type the \"text\" in an input box.\n"
            # prompt += "Type_and_Enter (text): Type the \"text\" followed by an Enter operation (useful for searching).\n"
        else:
            prompt += "Unable to Type. You cannot use the actions involving \"Type\" because the keyboard has not been activated. If you want to type, please first activate the keyboard by tapping on the input box on the screen.\n"
        prompt += "Home(): Return to home page.\n\n"

        prompt += "#### Shortcuts ####\n"
        if info_pool.shortcuts != {}:
            prompt += "You have the following shortcuts available:\n"
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"{shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are available.\n"
        prompt += "\n"
        prompt += "---\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        prompt += "A detailed explanation of your rationale for the action you choose.\n\n"
        prompt += "### Action ###\n"
        prompt += "Choose only one action or one shortcut from above. Be sure to fill in all required arguments such as text and coordinates in the ()."
        prompt += "Please provide your decision using a valid JSON format specifying the name and arguments of the action. For example, if you choose to tap at position (100, 200), you should write {\"name\":\"Tap\", \"arguments\":{\"x\":100, \"y\":100}}. If an action does not require arguments, such as Home, fill in None to the \"arguments\" field.\n\n"
        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action and the expected outcome."
        return prompt

    def execute_atomic_action(self, action: str, **kwargs) -> None:
        #TODO: Implement this with json formatting
        adb_path = self.adb
        if "Open_App" in action:
            screenshot_file = kwargs["screenshot_file"]
            ocr_detection = kwargs["ocr_detection"]
            ocr_recognition = kwargs["ocr_recognition"]
            app_name = action.split("(")[-1].split(")")[0]
            text, coordinate = ocr(screenshot_file, ocr_detection, ocr_recognition)
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [int((coordinate[ti][0] + coordinate[ti][2])/2), int((coordinate[ti][1] + coordinate[ti][3])/2)]
                    tap(adb_path, name_coordinate[0], name_coordinate[1]- int(coordinate[ti][3] - coordinate[ti][1]))# 
                    break
        
        elif "Tap" in action:
            coordinate = action.split("(")[-1].split(")")[0].split(", ")
            x, y = int(coordinate[0]), int(coordinate[1])
            tap(adb_path, x, y)
        
        elif "Swipe" in action:
            # coordinate1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
            # coordinate2 = action.split("), (")[-1].split(")")[0].split(", ")
            coordinates = action.split("(")[-1].split(")")[0].split(", ")
            x1, y1 = int(coordinate1[0]), int(coordinate1[1])
            x2, y2 = int(coordinate2[0]), int(coordinate2[1])
            swipe(adb_path, x1, y1, x2, y2)
            
        elif "Type" in action and "Enter" not in action:
            if "(text)" not in action:
                text = action.split("(")[-1].split(")")[0]
            else:
                text = action.split(" \"")[-1].split("\"")[0]
            type(adb_path, text)
        
        elif "Type" in action and "Enter" in action:
            if "(text)" not in action:
                text = action.split("(")[-1].split(")")[0]
            else:
                text = action.split(" \"")[-1].split("\"")[0]
            type_and_enter(adb_path, text)
        
        elif "Back" in action:
            back(adb_path)
        
        elif "Home" in action:
            home(adb_path)
        
        time.sleep(5)

    def execute_action(self, action_str: str, info_pool: InfoPool) -> InfoPool:

    def parse_response(self, response: str, info_pool: InfoPool) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ", " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}
        

class Notetaker(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    def parse_response(self, response: str) -> dict:
        pass

class ActionReflector(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    def parse_response(self, response: str) -> dict:
        pass

class KnowledgeReflector(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    def parse_response(self, response: str) -> dict:
        pass

