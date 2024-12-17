AGENT_TYPE="agent_E"
SETTING="individual"
# TASK_ROOT="/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/Mobile-Agent-v2/data/batch_v1"

# TASK_JSON_NAME="scenario_1_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --setting $SETTING \
    --instruction "On Maps, find out how long would it take to drive from here to LA?" \
    # --instruction "Create a new note and write a joke in it."
    # --instruction "Can you check the MacDonald's APP to see if there are any Rewards or Deals including Spicy McCrispy. If so, help me add that to Mobile Order (Do not pay yet, I will do it myself). And then check the pickup location and get directions on Google Maps. Stop at the screen showing the route."
    # --instruction "Find the most-cited paper that cites the paper 'Segment Anything' on Google Scholar. Stop at the screen showing the paper abstract."
    # --instruction "Check if any of the following items are on sale at Walmart: ribeye steak, fresh oranges, or toilet paper. If any are on sale, add a note in Notes with their prices."