from mujoco_infra.mujoco_utils.topology.state_2_topology import state2topology


def uncertainty_score_function_topology(predictions, topology_gt):
    uncertainty = 0
    if len(predictions.shape) != 2:
        raise ValueError('predictions shape is wrong, shape=', len(predictions.shape))
    for prediction_index in range(predictions.shape[0]):
        topology = state2topology(predictions[prediction_index].clone().detach())
        if topology == topology_gt:
            uncertainty +=1
    return uncertainty/predictions.shape[0]