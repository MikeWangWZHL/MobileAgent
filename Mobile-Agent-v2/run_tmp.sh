AGENT_TYPE="agent_E"
SETTING="individual"
# TASK_ROOT="/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/Mobile-Agent-v2/data/batch_v1"

# TASK_JSON_NAME="scenario_1_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --setting $SETTING \
    --instruction "Check if any of the following items are on sale at Walmart: ribeye steak, fresh oranges, or toilet paper. If any are on sale, add a note in Notes with their prices."