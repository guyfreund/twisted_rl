RopeAssetsPaths = {
   7: "mujoco_infra/assets/rope_v3_7_links.xml",
   15: "mujoco_infra/assets/rope_v3_15_links.xml",
   21: "mujoco_infra/assets/rope_v3_21_links.xml",
}


def convert_name_to_index(name:str, num_of_links:int=None):
    """
    Convert G to index
    """
    if num_of_links == 21:
        names = {
            "G0": 21,"G1": 20,"G2": 19,"G3": 18,"G4": 17,
            "G5": 16,"G6": 15,"G7": 14,"G8": 13,"G9": 12,
            "G10": 1,"G11": 2,"G12": 3,"G13": 4,"G14": 5,
            "G15": 6,"G16": 7,"G17": 8,"G18": 9,"G19": 10,"G20": 11
        }
    elif num_of_links == 15:
        names = {
            "G0": 15,"G1": 14,"G2": 13,"G3": 12,"G4": 11,
            "G5": 10,"G6": 9,"G7": 1,"G8": 2,"G9": 3,
            "G10": 4, "G11": 5, "G12": 6, "G13": 7,
            "G14": 8
        }
    elif num_of_links == 11:
        names = {
            "G0": 11,"G1": 10,"G2": 9,"G3": 8,"G4": 7,
            "G5": 1,"G6": 2,"G7": 3,"G8": 4,"G9": 5,
            "G10": 6
        }
    elif num_of_links == 7:
        names = {
            "G0": 7,"G1": 6,"G2": 5,"G3": 1,"G4": 2,
            "G5": 3,"G6": 4
        }
    else:
        raise
    return names[name]
