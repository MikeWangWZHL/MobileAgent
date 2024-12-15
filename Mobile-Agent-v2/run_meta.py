
from inference_agent_E import run_single_task as run_single_task_agent_E
from inference_agent_E import Perceptor, DEFAULT_PERCEPTION_ARGS, ADB_PATH, INIT_TIPS, INIT_SHORTCUTS
from inference_mobile_agent_v2 import run_single_task as run_single_task_mobile_agent_v2 # TODO
import torch
import os
import json
import shutil
import time

###################################################################################################

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
    parser.add_argument("--agent_type", type=str, default="agent_E", choices=['agent_E', 'mobile_agent_v2']) # agent system to use
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--tasks_json", type=str, default=None)
    parser.add_argument("--specified_knowledge_path", type=str, default=None)
    parser.add_argument("--specified_shortcuts_path", type=str, default=None)
    parser.add_argument("--setting", type=str, default="individual", choices=["individual", "curriculum"]) # individual or curriculum
    parser.add_argument("--future_tasks_visible", action="store_true", default=True)
    parser.add_argument("--reset_phone_state", action="store_true", default=True)
    parser.add_argument("--max_itr", type=int, default=40)
    parser.add_argument("--max_consecutive_failures", type=int, default=3)
    parser.add_argument("--max_repetitive_actions", type=int, default=3)
    parser.add_argument("--overwrite_task_log_dir", action="store_true", default=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.log_root is None:
        args.log_root = f"logs/{args.agent_type}"

    if args.instruction is None and args.tasks_json is None:
        raise ValueError("You must provide either instruction or tasks_json.")
    if args.instruction is not None and args.tasks_json is not None:
        raise ValueError("You cannot provide both instruction and tasks_json.")
    
    # choose the agent system
    if args.agent_type == "agent_E":
        print("#########\n### Using agent_E ###\n#########")
        run_single_task = run_single_task_agent_E
    elif args.agent_type == "mobile_agent_v2":
        print("#########\n### Using mobile_agent_v2 ###\n#########")
        if args.setting != "individual":
            print("WARNING: Only individual setting is supported for mobile_agent_v2, overwriting args.setting ...")
        args.setting = "individual" # only individual setting is supported for mobile_agent_v2
        run_single_task = run_single_task_mobile_agent_v2
    else:
        raise ValueError("Invalid agent_type:", args.agent_type)
    
    # run inference
    if args.instruction is not None:
        # single task inference
        try:
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
                max_itr=args.max_itr,
                max_consecutive_failures=args.max_consecutive_failures,
                max_repetitive_actions=args.max_repetitive_actions,
                overwrite_log_dir=args.overwrite_task_log_dir
            )
        except Exception as e:
            print(f"Failed when doing task: {args.instruction}")
            print("ERROR:", e)
    else:
        # multi task inference
        task_json = json.load(open(args.tasks_json, "r"))
        if "tasks" in task_json:
            tasks = task_json["tasks"]
        else:
            tasks = task_json

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
                    json.dump(INIT_SHORTCUTS, f, indent=4)
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
            try:
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
                    max_itr=args.max_itr,
                    max_consecutive_failures=args.max_consecutive_failures,
                    max_repetitive_actions=args.max_repetitive_actions,
                    overwrite_log_dir=args.overwrite_task_log_dir
                )
                print("DONE:", task["instruction"])
                print("Sleeping for 5 seconds before next task ...")
                time.sleep(5)
            except Exception as e:
                print(f"Failed when doing task: {instruction}")
                print("ERROR:", e)

if __name__ == "__main__":
    main()

