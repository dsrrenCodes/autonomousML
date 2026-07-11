from app.state.agent_state import AgentState



def route_by_status(state: AgentState)->str:
    current_status=state['status']
    current_retries_count= state.get('retry_count',0)
    max_retries= state.get('max_retries')
    if current_status== 'retry' and current_retries_count<max_retries:
        return 'retry' # map to name of experiment_node (change according to lokav experiment fn)
    else:
        return 'end'
