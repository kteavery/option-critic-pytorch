import os

def get_intervention_dir(environment):
    os.makedirs(
        f"jsons/states/interventions/{environment}", exist_ok=True
    )
    return f"jsons/states/interventions/{environment}"


def get_start_state_path(environment):
    prefix = "trajectory_"
    os.makedirs(f"jsons/states/starts/{environment}", exist_ok=True)
    return f"jsons/states/{prefix}starts/{environment}/vanilla.json"
