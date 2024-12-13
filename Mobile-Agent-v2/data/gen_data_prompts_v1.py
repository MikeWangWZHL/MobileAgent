APP_LIST = [
    "Settings",
    "Contacts",
    # "Clock",
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
    # "REDnote",
    "X",
    "McDonald's",
]

single_app_seed_tasks = {
    "information_research_and_summarization": [
        {
            "instruction": "Find the best reviewed Korean restaurant in my area that is within 10min drive and opens late until 10pm.",
            "category": "information_research_and_summarization",
            "type": "single_app",
            "apps": ["Maps"]
        },
        {
            "instruction": "Find recent arxiv papers and news on \"X\" about GUI agents and put a summarized report in Notes.",
            "category": "information_research_and_summarization",
            "type": "multi_app",
            "apps": ["Chrome", "X", "Notes"]
        }
    ],
    "online_shopping_and_deal_hunting": [
        {
            "instruction": "Find me a laptop on Amazon that under $1000 with Nvidia GPU and more than 16GB RAM.",
            "category": "online_shopping_and_deal_hunting",
            "type": "single_app",
            "apps": ["Amazon Shopping"]
        },
        {
            "instruction": "Compare the price of a brand new iPhone 15 pro in Walmart, Best Buy and Amazon Shopping. Find me the best deal.",
            "category": "online_shopping_and_deal_hunting",
            "type": "multi_app",
            "apps": ["Amazon Shopping", "Walmart", "Best Buy"]
        }
    ],
    "entertainment_and_recreation": [
        {
            "instruction": "Read blogs about LA on REDnote, find interesting things to do and make a 2-day travel plan for me.",
            "category": "entertainment_and_recreation",
            "type": "single_app",
            "apps": ["REDnote"]
        },
        {
            "instruction": "Check movies currently in theater, check if there are any horror movies. Check some reviews on Lemon8 about the movies, and suggest which one worth trying.",
            "category": "entertainment_and_recreation",
            "type": "multi_app",
            "apps": ["Fandango", "Lemon8"]
        }
    ]
}


prompt_seed_tasks_template = """Imagine a smart assistant capable of operating a smartphone on behalf of human users. Brainstorm challenging mobile tasks that would be highly beneficial to users. Below, I have provided some seed examples to inspire your ideas.

Please propose more specific tasks for the category: {category}. Focus on tasks requiring complex reasoning and long-term planning. You may suggest both single-app and multi-app tasks. Be innovative!

### Available App List ###
{app_list}

### Seed Tasks ###
{seed_tasks}

### Your New Tasks ###
Provide your suggestions in the same JSON format as the seed tasks."""

seq_task_prompt_template = """
Imagine a smart assistant capable of operating a smartphone for human users. You want to test the assistant's ability to learn from experience. Brainstorm some challenging mobile task curriculums where later tasks are progressively more complex and can benefit from the knowledge and skills learned in earlier tasks. Below, I have provided seed examples for inspiration.

Please create a more concrete sequence of tasks for the category {category}. Focus on tasks that demand complex reasoning and long-horizon planning. The curriculum can include both single-app tasks and multi-app tasks. Be creative!

### Avaliable APP LIST ###
{app_list}

### Seed Tasks ###
{seed_tasks}

### Your Task Curriculums ###
Provide 3 curriculums, with a length of tasks [2,3,4] respectively.
Each curriculum should be a list of JSON objects, same as the seed tasks provided."""



if __name__ == "__main__":


    from pprint import pformat
    import json

    category = "information_research_and_summarization"

    prompt_seed_tasks = prompt_seed_tasks_template.format(
        app_list=json.dumps(APP_LIST), 
        seed_tasks=json.dumps(single_app_seed_tasks[category], indent=4), 
        category=category
    )

    print(prompt_seed_tasks)

    print("==============================\n\n")


    prompt_seq_tasks = seq_task_prompt_template.format(
        app_list=json.dumps(APP_LIST), 
        seed_tasks=json.dumps(single_app_seed_tasks[category], indent=4),
        category=category
    )

    print(prompt_seq_tasks)


