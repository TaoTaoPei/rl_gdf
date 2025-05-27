import xml.etree.ElementTree as ET

def calculate_total_mass(urdf_file_path):
    """
    Calculate the total mass of a robot described in a URDF file.

    Args:
        urdf_file_path (str): Path to the URDF file.

    Returns:
        float: Total mass of the robot.
    """
    try:
        # Parse the URDF file
        tree = ET.parse(urdf_file_path)
        root = tree.getroot()

        total_mass = 0.0

        # Iterate through all <link> elements
        for link in root.findall("link"):
            # Find the <inertial> element inside the <link>
            inertial = link.find("inertial")
            if inertial is not None:
                # Find the <mass> element inside the <inertial>
                mass = inertial.find("mass")
                if mass is not None:
                    try:
                        # Add the mass value to the total mass
                        total_mass += float(mass.attrib.get("value", 0.0))
                    except ValueError:
                        print(f"Warning: Invalid mass value in link {link.attrib.get('name', 'unknown')}")

        return total_mass

    except FileNotFoundError:
        print(f"Error: File not found at {urdf_file_path}")
        return 0.0
    except ET.ParseError:
        print(f"Error: Failed to parse URDF file at {urdf_file_path}")
        return 0.0

# Example usage
if __name__ == "__main__":
    urdf_path = "/home/lwx/data/rl-gym/work/hw5_base_v5_c9_all/logs/humanoid_ppo_hw/hw5_base_new_T5_A1_20250112164136/hw5_base_new_T5_A1/v2_0114/urdf/xyuan_description_right_link.urdf"  # Replace with the path to your URDF file
    total_mass = calculate_total_mass(urdf_path)
    print(f"Total Mass of the Robot: {total_mass} kg")
