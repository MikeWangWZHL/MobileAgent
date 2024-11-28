from MobileAgentE.agents import *

if __name__ == "__main__":
    info_pool = InfoPool(
        instruction="hello world",
        plan="hello big world",
        current_subgoal="hello world subgoal",
    )
    # manager = Manager()
    # prompt = manager.get_prompt(info_pool)
    # print(prompt)

    # executor = Executor()
    # prompt = executor.get_prompt(info_pool)
    # print(prompt)


    # tmp = "{{\"name\":\"TAP\",\"args\":[\"{{'x': 100, 'y': 200}}\"]}}"