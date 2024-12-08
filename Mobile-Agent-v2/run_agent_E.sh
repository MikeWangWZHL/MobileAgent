# python run_agent_E.py --instruction "Open Notes and type hello world"

# python run_agent_E.py --instruction "Search for the top 3 trending news topics on \"X\" and write a short summary in the \"Notes\" APP."

# python run_agent_E.py \
#     --tasks_json data/seq_task/example.json \
#     --setting "individual"

# python run_agent_E.py \
#     --tasks_json data/seq_task/example.json \
#     --setting "curriculum"

# python run_agent_E.py \
#     --run_name "information_research_example_0" \
#     --tasks_json data/curriculum/information_research_example_0.json \
#     --setting "curriculum"


## future task visible ##
python run_agent_E.py \
    --run_name "information_research_example_0_future_task_visible__fifth_run" \
    --tasks_json data/curriculum/information_research_example_0.json \
    --setting "curriculum" \
    --future_tasks_visible