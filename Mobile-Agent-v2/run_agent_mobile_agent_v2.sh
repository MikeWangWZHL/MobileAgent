AGENT_TYPE="mobile_agent_v2"
SETTING="individual"
TASK_ROOT="/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/Mobile-Agent-v2/data/batch_v1"

# # job 1
# TASK_JSON_NAME="scenario_1_batch_v1.json"
# python run_meta.py \
#     --agent_type $AGENT_TYPE \
#     --log_root "logs/$AGENT_TYPE" \
#     --run_name "$TASK_JSON_NAME-$SETTING" \
#     --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
#     --setting $SETTING

# job 2
TASK_JSON_NAME="scenario_2_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --log_root "logs/$AGENT_TYPE" \
    --run_name "$TASK_JSON_NAME-$SETTING" \
    --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
    --setting $SETTING

# # job 3
# TASK_JSON_NAME="scenario_3_batch_v1.json"
# python run_meta.py \
#     --agent_type $AGENT_TYPE \
#     --log_root "logs/$AGENT_TYPE" \
#     --run_name "$TASK_JSON_NAME-$SETTING" \
#     --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
#     --setting $SETTING

# # job 4
# TASK_JSON_NAME="scenario_4_batch_v1.json"
# python run_meta.py \
#     --agent_type $AGENT_TYPE \
#     --log_root "logs/$AGENT_TYPE" \
#     --run_name "$TASK_JSON_NAME-$SETTING" \
#     --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
#     --setting $SETTING

# # job 5
# TASK_JSON_NAME="scenario_5_batch_v1.json"
# python run_meta.py \
#     --agent_type $AGENT_TYPE \
#     --log_root "logs/$AGENT_TYPE" \
#     --run_name "$TASK_JSON_NAME-$SETTING" \
#     --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
#     --setting $SETTING