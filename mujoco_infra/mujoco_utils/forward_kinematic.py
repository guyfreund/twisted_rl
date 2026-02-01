import torch
from mujoco_infra.mujoco_utils.mujoco import get_number_of_links, get_joints_indexes

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_rotation(quaternion):
    batch_size= quaternion.shape[0]
    rotation = torch.ones([4,4], device='cuda:0', dtype=torch.float64)
    rotation = rotation.repeat(batch_size,1,1)
    rotation[:,3,:3] *= 0
    rotation[:,:3,3] *= 0
    rotation[:,:3,:3] *= quaternion_to_matrix(quaternion)
    return rotation

def update_sum_rotation_matrix(matrix, rotation):
  mat = torch.matmul(matrix[:],rotation[:])
  return mat

def forward_kinematic_from_qpos_torch_batch(qpos):
  num_of_links = get_number_of_links(qpos=qpos)
  INDEX = get_joints_indexes(num_of_links=num_of_links)
  qpos = qpos.to("cuda:0")
  batch_size = qpos.shape[0]

  #angles_roll, angles_pitch, angles_yaw
  locations_joints = torch.ones([batch_size,num_of_links+1,4], device='cuda:0',\
    dtype=torch.float64)
  locations_joints[:,int(num_of_links/2)] *= torch.tensor([-0.02,0.0,0.0,1.0],\
    dtype=torch.float64, device='cuda:0',requires_grad=True)
  locations_joints[:,int(num_of_links/2)+1] *= torch.tensor([0.02,0.0,0.0,1.0],\
    dtype=torch.float64, device='cuda:0',requires_grad=True)

  sum_rotation_matrix = torch.tensor([
      [1.0,0.0,0.0,0.0],
      [0.0,1.0,0.0,0.0],
      [0.0,0.0,1.0,0.0],
      [0.0,0.0,0.0,1.0],
      ], requires_grad=True, dtype=torch.float64, device='cuda:0')
  sum_rotation_matrix = sum_rotation_matrix.repeat(batch_size,1,1)

  for joint_index in reversed(range(int(num_of_links/2))):
    rotation_matrix = torch.ones([batch_size,4,4], device='cuda:0',dtype=torch.float64)
    euler = torch.ones([batch_size,3], device='cuda:0',dtype=torch.float64)
    euler[:,0] *= 0
    joint_0 = "J0_"+str(joint_index)
    index_0 = INDEX[joint_0]
    euler[:,1] *= qpos[:,index_0]
    joint_1 = "J1_"+str(joint_index)
    index_1 = INDEX[joint_1]
    euler[:,2] *= qpos[:,index_1]
    rotation_matrix[:,:3,:3] *= euler_angles_to_matrix(euler, "XYZ")
    rotation_matrix[:,:3,3] *= 0
    rotation_matrix[:,3,:3] *= 0
    try_location = torch.tensor([[-0.04,0,0,1]], device='cuda:0',dtype=torch.float64,requires_grad=True)

    temp_location = torch.ones([batch_size, 4], device='cuda:0',dtype=torch.float64)
    temp_location[:] = run_forward_torch_batch(sum_rotation_matrix, try_location, rotation_matrix)
    locations_joints[:,joint_index] =\
      temp_location[:] + locations_joints[:,joint_index+1] - torch.tensor([0,0,0,1.0],\
       device='cuda:0',dtype=torch.float64,)
    sum_rotation_matrix = update_sum_rotation_matrix(sum_rotation_matrix, rotation_matrix)

  sum_rotation_matrix = torch.tensor([
      [1.0,0.0,0.0,0.0],
      [0.0,1.0,0.0,0.0],
      [0.0,0.0,1.0,0.0],
      [0.0,0.0,0.0,1.0],
      ], requires_grad=True, dtype=torch.float64, device='cuda:0')
  sum_rotation_matrix = sum_rotation_matrix.repeat(batch_size,1,1)

  for joint_index in range(int(num_of_links/2)+2,num_of_links+1):
    rotation_matrix = torch.ones([batch_size,4,4], device='cuda:0',dtype=torch.float64)
    euler = torch.ones([batch_size,3], device='cuda:0',dtype=torch.float64)
    euler[:,0] *= 0
    joint_0 = "J0_"+str(joint_index-1)
    index_0 = INDEX[joint_0]
    euler[:,1] *= qpos[:,index_0]
    joint_1 = "J1_"+str(joint_index-1)
    index_1 = INDEX[joint_1]
    euler[:,2] *= qpos[:,index_1]
    rotation_matrix[:,:3,:3] *= euler_angles_to_matrix(euler, "XYZ")
    rotation_matrix[:,:3,3] *= 0
    rotation_matrix[:,3,:3] *= 0

    try_location = torch.tensor([[0.04,0,0,1]], device='cuda:0',dtype=torch.float64, requires_grad=True)

    temp_location = torch.ones([batch_size, 4], device='cuda:0',dtype=torch.float64,)
    temp_location[:] = run_forward_torch_batch(sum_rotation_matrix, try_location, rotation_matrix)
    locations_joints[:,joint_index] =\
      temp_location[:] + locations_joints[:,joint_index-1] - torch.tensor([0,0,0,1.0],\
       device='cuda:0',dtype=torch.float64,)
    sum_rotation_matrix = update_sum_rotation_matrix(sum_rotation_matrix, rotation_matrix)
  rotation_matrix = quaternion_to_rotation(qpos[:,3:7])

  sum_rotation_matrix = torch.tensor([
      [1.0,0.0,0.0,0.0],
      [0.0,1.0,0.0,0.0],
      [0.0,0.0,1.0,0.0],
      [0.0,0.0,0.0,1.0],
      ], requires_grad=True, dtype=torch.float64, device='cuda:0')
  sum_rotation_matrix = sum_rotation_matrix.repeat(batch_size,1,1)

  final_locations_joints = torch.ones([batch_size,num_of_links+1,4], device='cuda:0', dtype=torch.float64)
  final_locations_joints[:,:] *= summary_run_forward_torch_batch(sum_rotation_matrix, locations_joints[:],\
     rotation_matrix, num_of_links)
  qpos = torch.unsqueeze(qpos,1)
  final_locations_joints[:,:,0:3] += qpos[:,:,0:3]
  return final_locations_joints[:,:,:3]

def run_forward_torch_batch(sum_rotation_matrix, location, rot, trans=None):
  batch_size = rot.shape[0]
  if rot is not None:
    output = torch.ones([batch_size, 4,1], device='cuda:0',dtype=torch.float64)
    mat = torch.ones([batch_size, 4, 4], device='cuda:0',dtype=torch.float64)
    if trans is not None:
      mat *= torch.matmul(trans[:],torch.matmul(sum_rotation_matrix[:],rot[:]))
    else:
      mat *= torch.matmul(sum_rotation_matrix[:],rot[:])
    if len(location.shape) == 2:
      location = torch.unsqueeze(location,-1)
    output *= torch.matmul(mat[:,:],location[:])
    return torch.squeeze(output)
  else:
    location = torch.unsqueeze(location,-1)
    if trans is not None:
     output = torch.matmul(trans[:,:],location[:])
    else:
      output = location
    return torch.squeeze(output)

def summary_run_forward_torch_batch(sum_rotation_matrix, location, rot, num_of_links, trans=None):
  batch_size = rot.shape[0]
  output = torch.ones([batch_size,num_of_links+1, 4,1], device='cuda:0',dtype=torch.float64)
  mat = torch.ones([batch_size,  4, 4], device='cuda:0',dtype=torch.float64)
  if trans is not None:
    mat *= torch.matmul(trans[:],torch.matmul(rot[:],sum_rotation_matrix[:]))
  else:
    mat *= torch.matmul(rot[:],sum_rotation_matrix[:])
  if len(location.shape) == 3:
    location = torch.unsqueeze(location,-1)
  mat = torch.unsqueeze(mat,1)
  output *= torch.matmul(mat[:,:],location[:,:])
  return torch.squeeze(output)
