APP_LIST = [
    "Settings",
    "Contacts",
    "Notes",
    "Calendar"
    "Chrome",
    "Google",
    "YouTube",
    "TikTok",
    "Gmail",
    "Maps",
    "TripAdvisor",
    "Booking",
    "Amazon Shopping",
    "Walmart",
    "Best Buy",
    "Fandango",
    "Lemon8",
    "REDnote",
    "X",
    "Instagram"
]

single_app_seed_tasks = {
    "information_research_and_summarization": [
        {
            "task_description": "Find the best reviewed Korean restaurant in my area that is within 10min drive and opens late until 10pm.",
            "type": "single_app",
            "apps": ["Maps"]
        },
        {
            "task_description": "Find recent arxiv papers and news on \"X\" about GUI agents and put a summarized report in Notes.",
            "type": "multi_app",
            "apps": ["Chrome", "X", "Notes"]
        }
    ],
    "routine_management": [
        {
            "task_description": "Move all advertisement emails in my Gmail into trash, and clear trash.",
            "apps": ["Gmail"]
        },
        {
            "task_description": "Check recent emails in my Gmail, find meetings involving me and add them in the Calendar.",
            "apps": ["Gmail", "Calendar"]
        }
    ],
    "online_shopping_and_deal_hunting": [
        {
            "task_description": "Find me a laptop on Amazon that under $1000 with Nvidia GPU and more than 16GB RAM.",
            "apps": ["Amazon Shopping"]
        },
        {
            "task_description": "Compare the price of a brand new iPhone 15 pro in Walmart, Best Buy and Amazon Shopping. Find me the best deal.",
            "apps": ["Amazon Shopping", "Walmart", "Best Buy"]
        }
    ],
    "entertainment_and_recreation": [
        {
            "task_description": "Read blogs about LA on REDnote, find interesting things to do and make a 2-day travel plan for me.",
            "apps": ["REDnote"]
        },
        {
            "task_description": "Check movies currently in theater, check if there are any horror movies. Check some reviews on Lemon8 about the movies, and suggest which one worth trying.",
            "apps": ["Fandango", "Lemon8"]
        }
    ]
}


prompt_seed_tasks_template = """Imagine a smart robot that can operate your smartphone on your behalf. Help me brainstorm additional challenging mobile tasks that could be highly beneficial for human users. Below, I have provided some seed examples for inspiration. Please suggest 10 more tasks for each of the following four categories: information_research_and_summarization, routine_management, online_shopping_and_deal_hunting, and entertainment_and_recreation. Include examples of both single-app tasks and multi-app tasks. 
Note that the seed tasks are only examples and you can suggest any task that you think would be interesting or useful. Be creative!

### APP LIST ###
{app_list}

### Seed Tasks ###
{seed_tasks}

### Your New tasks ###
Use the same JSON format as the seed tasks."""

seq_task_prompt_template = """Imagine a smart robot that can operate your smartphone on your behalf. Now you want to test the robot's ability to learn from experience. Design a sequence of tasks (a total number of 5 tasks) from simple to hard for each category. Ensure that some skills or subroutines in the previous tasks can be helpful for later tasks.
Below are a list of avaliable APPs and a set of seed tasks for inspiration. Please design a sequence of tasks for each category: information_research_and_summarization, routine_management, online_shopping_and_deal_hunting, and entertainment_and_recreation. Include examples of both single-app tasks and multi-app tasks. Note that the seed tasks are only examples and you can suggest any task that you think would be interesting or useful. Be creative!

### APP LIST ###
{app_list}

### Seed Tasks ###
{seed_tasks}

### Your Task Sequences ###
Use the same JSON format as the seed tasks."""




from pprint import pformat
import json

prompt_seed_tasks = prompt_seed_tasks_template.format(app_list=json.dumps(APP_LIST), seed_tasks=json.dumps(single_app_seed_tasks, indent=4))

print(prompt_seed_tasks)

print("==============================\n\n")


prompt_seq_tasks = seq_task_prompt_template.format(app_list=json.dumps(APP_LIST), seed_tasks=json.dumps(single_app_seed_tasks, indent=4))

print(prompt_seq_tasks)


