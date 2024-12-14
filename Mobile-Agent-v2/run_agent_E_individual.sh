AGENT_TYPE="agent_E"
SETTING="individual"
TASK_ROOT="/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/Mobile-Agent-v2/data/batch_v1"

TASK_JSON_NAME="scenario_1_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --log_root "logs/$AGENT_TYPE" \
    --run_name "$TASK_JSON_NAME-$SETTING" \
    --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
    --setting $SETTING

TASK_JSON_NAME="scenario_2_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --log_root "logs/$AGENT_TYPE" \
    --run_name "$TASK_JSON_NAME-$SETTING" \
    --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
    --setting $SETTING

TASK_JSON_NAME="scenario_3_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --log_root "logs/$AGENT_TYPE" \
    --run_name "$TASK_JSON_NAME-$SETTING" \
    --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
    --setting $SETTING

TASK_JSON_NAME="scenario_4_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --log_root "logs/$AGENT_TYPE" \
    --run_name "$TASK_JSON_NAME-$SETTING" \
    --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
    --setting $SETTING

TASK_JSON_NAME="scenario_5_batch_v1.json"
python run_meta.py \
    --agent_type $AGENT_TYPE \
    --log_root "logs/$AGENT_TYPE" \
    --run_name "$TASK_JSON_NAME-$SETTING" \
    --tasks_json "$TASK_ROOT/$TASK_JSON_NAME" \
    --setting $SETTING
