"""
Efficient Robot Grasping Simulation for M1 Pro MacBook
Uses Genesis robotics simulator with optimized settings for Apple Silicon
"""

import genesis as gs
import numpy as np


def main():
    # Initialize Genesis with performance settings for M1 Pro
    gs.init(
        backend=gs.metal,  # Use Metal backend for Apple Silicon
        precision='32',    # Use 32-bit precision for good balance
        logging_level='warning',
    )
    
    # Create scene with efficient options
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,           # 10ms timestep - good balance
            substeps=10,       # Physics substeps
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,        # Limit FPS for efficiency
            camera_pos=(2.5, -2.5, 1.8),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.5,
            show_link_frame=False,
            show_cameras=False,
        ),
        renderer=gs.renderers.Rasterizer(),  # Use Rasterizer (faster than RayTracer)
    )
    
    # Add ground plane
    ground = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=0.8,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4, 1.0),
        ),
    )
    
    # Add robot with gripper (using Franka Panda as example - common research robot)
    # If you don't have the URDF, Genesis often includes common robot models
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file='franka_panda/panda.urdf',  # Adjust path to your robot URDF
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 0),
            fixed=True,  # Fix base to ground
        ),
        material=gs.materials.Rigid(),
    )
    
    # Add target object (cube) to grasp
    target_object = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.05, 0.05, 0.05),  # 5cm cube
            pos=(0.5, 0.0, 0.05),
            euler=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            rho=500.0,  # Density
            friction=0.8,
        ),
        surface=gs.surfaces.Smooth(
            color=(0.8, 0.2, 0.2, 1.0),  # Red color
        ),
    )
    
    # Build the scene
    scene.build()
    
    # Get robot DOF information
    n_dofs = robot.n_dofs
    print(f"Robot has {n_dofs} DOFs")
    
    # Define initial joint positions (example for Franka Panda)
    # Adjust these based on your specific robot
    initial_qpos = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785, 0.04, 0.04])
    
    # Target joint positions for grasping (reaching toward object)
    grasp_approach_qpos = np.array([0.0, 0.2, 0.0, -1.5, 0.0, 1.7, 0.785, 0.04, 0.04])
    grasp_close_qpos = np.array([0.0, 0.2, 0.0, -1.5, 0.0, 1.7, 0.785, 0.0, 0.0])
    
    # Simulation parameters
    steps_per_phase = 200
    
    # Simulation loop
    print("Starting simulation...")
    print("Phase 1: Moving to initial position")
    
    for i in range(1000):
        # Phase 1: Move to initial position (0-200 steps)
        if i < steps_per_phase:
            alpha = i / steps_per_phase
            target_qpos = initial_qpos
            robot.set_qpos(target_qpos[:n_dofs])
        
        # Phase 2: Move toward object (200-400 steps)
        elif i < 2 * steps_per_phase:
            alpha = (i - steps_per_phase) / steps_per_phase
            target_qpos = initial_qpos + alpha * (grasp_approach_qpos - initial_qpos)
            robot.set_qpos(target_qpos[:n_dofs])
            if i == steps_per_phase:
                print("Phase 2: Approaching object")
        
        # Phase 3: Close gripper (400-600 steps)
        elif i < 3 * steps_per_phase:
            alpha = (i - 2 * steps_per_phase) / steps_per_phase
            target_qpos = grasp_approach_qpos + alpha * (grasp_close_qpos - grasp_approach_qpos)
            robot.set_qpos(target_qpos[:n_dofs])
            if i == 2 * steps_per_phase:
                print("Phase 3: Closing gripper")
        
        # Phase 4: Lift object (600-800 steps)
        elif i < 4 * steps_per_phase:
            alpha = (i - 3 * steps_per_phase) / steps_per_phase
            lift_qpos = grasp_close_qpos.copy()
            lift_qpos[1] -= 0.3 * alpha  # Lift by moving joint
            robot.set_qpos(lift_qpos[:n_dofs])
            if i == 3 * steps_per_phase:
                print("Phase 4: Lifting object")
        
        # Phase 5: Hold (800-1000 steps)
        else:
            if i == 4 * steps_per_phase:
                print("Phase 5: Holding object")
            pass
        
        # Step simulation
        scene.step()
    
    print("Simulation complete!")


if __name__ == "__main__":
    main()

