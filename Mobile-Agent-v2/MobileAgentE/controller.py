import os
import time
import subprocess
from PIL import Image
from time import sleep

def get_screenshot(adb_path):
    command = adb_path + " shell rm /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + " shell screencap -p /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + " pull /sdcard/screenshot.png ./screenshot"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    image_path = "./screenshot/screenshot.png"
    save_path = "./screenshot/screenshot.jpg"
    image = Image.open(image_path)
    image.convert("RGB").save(save_path, "JPEG")
    os.remove(image_path)

def save_screenshot_to_file(adb_path, file_path="screenshot.png"):
    """
    Captures a screenshot from an Android device using ADB, saves it locally, and removes the screenshot from the device.

    Args:
        adb_path (str): The path to the adb executable.

    Returns:
        str: The path to the saved screenshot, or raises an exception on failure.
    """
    # Define the local filename for the screenshot
    local_file = file_path
    
    if os.path.dirname(local_file) != "":
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

    # Define the temporary file path on the Android device
    device_file = "/sdcard/screenshot.png"
    
    try:
        print("\tRemoving existing screenshot from the Android device...")
        command = adb_path + " shell rm /sdcard/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)

        # Capture the screenshot on the device
        print("\tCapturing screenshot on the Android device...")
        result = subprocess.run(f"{adb_path} shell screencap -p {device_file}", capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(f"Error: Failed to capture screenshot on the device. {result.stderr}")
        
        # Pull the screenshot to the local computer
        print("\tTransferring screenshot to local computer...")
        result = subprocess.run(f"{adb_path} pull {device_file} {local_file}", capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(f"Error: Failed to transfer screenshot to local computer. {result.stderr}")
        
        # Remove the screenshot from the device
        print("\tRemoving screenshot from the Android device...")
        result = subprocess.run(f"{adb_path} shell rm {device_file}", capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error: Failed to remove screenshot from the device. {result.stderr}")
        
        print(f"\tScreenshot saved to {local_file}")
        return local_file
    
    except Exception as e:
        print(str(e))
        return None


def tap(adb_path, x, y):
    command = adb_path + f" shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            command = adb_path + f" shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == '_':
            command = adb_path + f" shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in '-.,!?@\'°/:;()':
            command = adb_path + f" shell input text \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)

def enter(adb_path):
    command = adb_path + f" shell input keyevent KEYCODE_ENTER"
    subprocess.run(command, capture_output=True, text=True, shell=True)

# def type_and_enter(adb_path, text):
#     type(adb_path, text)
#     sleep(0.5)
#     enter(adb_path)

def swipe(adb_path, x1, y1, x2, y2):
    command = adb_path + f" shell input swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_path):
    command = adb_path + f" shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    
    
def home(adb_path):
    # command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    command = adb_path + f" shell input keyevent KEYCODE_HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)

def switch_app(adb_path):
    command = adb_path + f" shell input keyevent KEYCODE_APP_SWITCH"
    subprocess.run(command, capture_output=True, text=True, shell=True)

### for debugging only ###
# 540 1835
def clear_background_and_back_to_home(adb_path):
    # pull up background apps
    command = adb_path + f" shell input keyevent KEYCODE_APP_SWITCH"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    # tap closs all
    sleep(2)
    tap(adb_path, 540, 1835)
    sleep(2)
    home(adb_path)


def check_pixel_color(
        image_path, x, y, 
        target_color_1_name = "all_off",
        target_color_2_name = "all_on",
        target_color_1 = [1, 1, 1], # black 
        target_color_2 = [242, 101, 73] # red
    ):
    """
    Loads an image and checks the color of a pixel at (x, y).
    Determines whether the pixel color is closer to red or black.

    Args:
        image_path (str): Path to the image file.
        x (int): X-coordinate of the pixel.
        y (int): Y-coordinate of the pixel.

    Returns:
        str: 'red' if the pixel color is closer to red, 'black' if closer to black.
    """
    # Load the image
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        return "Image not found. Please check the path."

    # Ensure the coordinates are within the image bounds
    width, height = image.size
    if not (0 <= x < width and 0 <= y < height):
        return "Pixel coordinates are out of bounds."

    # Get the RGB value of the pixel
    rgb = image.convert('RGB').getpixel((x, y))

    print(rgb)
    target_color_1_distance = ((rgb[0] - target_color_1[0])**2 + (rgb[1] - target_color_1[1])**2 + (rgb[2] - target_color_1[2])**2)**0.5
    target_color_2_distance = ((rgb[0] - target_color_2[0])**2 + (rgb[1] - target_color_2[1])**2 + (rgb[2] - target_color_2[2])**2)**0.5

    # Determine closer color
    if target_color_1_distance < target_color_2_distance:
        return target_color_1_name
    else:
        return target_color_2_name

def clear_chrome_tabs(adb_path):
    home(adb_path)
    sleep(1.5)

    # open chrome
    command = adb_path + f" shell am start -n com.android.chrome/com.google.android.apps.chrome.Main"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    sleep(8)

    # tap tabs
    tap(adb_path, 878, 162)
    sleep(2)

    # tap three dots
    tap(adb_path, 1013, 162)
    sleep(2)

    # tap close all tabs
    tap(adb_path, 630, 432)
    sleep(2)

    # confirm
    tap(adb_path, 727, 1330)
    sleep(2)

    home(adb_path)

def clear_notes(adb_path):
    # # notes package name: com.samsung.android.app.notes
    home(adb_path)
    sleep(1.5)

    # open notes
    command = adb_path + f" shell am start -n com.samsung.android.app.notes/.memolist.MemoListActivity"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    sleep(4)

    # tap three dots
    tap(adb_path, 987, 820)
    sleep(2)
    
    # tap edit
    tap(adb_path, 642, 827)
    sleep(2)

    # check if the all toggle is on
    image_path = save_screenshot_to_file(adb_path, "./screenshot_for_checking.png")
    if image_path is not None:
        toggle_status = check_pixel_color(
            image_path, 97, 783, 
            target_color_1_name="all_off", target_color_2_name="all_on",
            target_color_1=[1, 1, 1], target_color_2=[242, 101, 73]
        )
        print(toggle_status)
        if toggle_status == "all_off":
            # tap all
            tap(adb_path, 92, 799)
            sleep(2)

    # tap delete
    tap(adb_path, 751, 2102)
    sleep(2)

    # tap move to trash
    tap(adb_path, 723, 2056)
    sleep(2)

    home(adb_path)
    # command = adb_path + f" shell pm clear com.samsung.android.app.notes"
    # subprocess.run(command, capture_output=True, text=True, shell=True)


APP_PACKAGE_LIST=[
    "com.walmart.android",
    "com.samsung.android.app.notes",
    "com.bd.nproject",
    "com.zhiliaoapp.musically",
    "com.amazon.mShop.android.shopping",
    "com.samsung.android.app.reminder",
    "com.google.android.calendar",
    "com.fandango",
    "com.tripadvisor.tripadvisor",
    "com.whatsapp",
    "com.twitter.android",
    "com.mcdonalds.app",
    "com.sec.android.app.kidshome",
    "com.sec.android.app.clockpackage",
    "com.booking",
    "com.bestbuy.android",
    "com.google.android.youtube",
    "com.google.android.apps.maps",
    "com.google.android.calendar",
    "com.android.chrome",
    "com.google.android.gm",
    "com.google.android.googlequicksearchbox"
]

def clear_processes(adb_path, device=None):
    ## 华为
    # command = adb_path + (f" -s {device}" if device is not None else '') + \
    #           f" shell am force-stop $( {adb_path} shell dumpsys activity activities | grep mResumedActivity | awk '{{print $4}}' | cut -d '/' -f 1)"

    # ## 小米
    # command = adb_path + (f" -s {device}" if device is not None else '') + \
    #           f" shell am force-stop $( {adb_path} shell dumpsys activity activities | grep topResumedActivity | awk '{{print $3}}' | cut -d '/' -f 1)"
    
    ## Samsung
    # command = adb_path + (f" -s {device}" if device is not None else '') + \
    #           f" shell am force-stop $( {adb_path} shell dumpsys activity activities | grep topResumedActivity | awk '{{print $3}}' | cut -d '/' -f 1)"
    # subprocess.run(command, capture_output=True, text=True, shell=True)
    
    for app in APP_PACKAGE_LIST:
        command = adb_path + f" shell am force-stop {app}"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        pass


def reset_everything(adb_path):
    clear_processes(adb_path=adb_path)
    clear_background_and_back_to_home(adb_path=adb_path)
    clear_chrome_tabs(adb_path=adb_path)
    clear_notes(adb_path=adb_path)
    clear_background_and_back_to_home(adb_path=adb_path)