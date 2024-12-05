from MobileAgentE.agents import *

if __name__ == "__main__":
    info_pool = InfoPool(
        instruction="hello world",
        plan="hello big world",
        current_subgoal="hello world subgoal",
        shortcuts=INIT_SHORTCUTS
    )
    # manager = Manager()
    # prompt = manager.get_prompt(info_pool)
    # print(prompt)

    # executor = Executor(adb_path="/Users/wangz3/Desktop/vlm_agent_project/platform-tools/adb")
    # prompt = executor.get_prompt(info_pool)
    # print(prompt)

    # action_object = {
    #     "name": "Tap",
    #     "arguments": {"x": 150, "y": 270}
    # }
    # action_object = {
    #     "name": "Switch_App",
    #     "arguments": None
    # }
    # action_object = {
    #     "name": "Swipe",
    #     "arguments": {"x1": 10, "y1": 270, "x2": 450, "y2": 270}
    # }
    # action_object = {
    #     "name": "Tap_Type_and_Enter",
    #     "arguments": {"x": 542, "y": 593, "text": "hello world"}
    # }
    # action_str = json.dumps(action_object)
    # print(action_str)
    # executor.execute(action_str=action_str, info_pool=info_pool)

    # notetaker = Notetaker()
    # prompt = notetaker.get_prompt(info_pool)
    # print(prompt)
    
    # notetaker = ActionReflector()
    # prompt = notetaker.get_prompt(info_pool)
    # print(prompt)
    
    notetaker = KnowledgeReflector()
    prompt = notetaker.get_prompt(info_pool)
    print(prompt)



    # tmp = """```json{
    #     "hello": "world",
    #     "hello2": [100, 200]
    #     }```
    # """

    # tmp = """```json{
    #     "name": "Tap",
    #     "arguments": null
    #     }```
    # """
    # print(tmp)
    # ret = extract_json_object(tmp)
    # print(ret)