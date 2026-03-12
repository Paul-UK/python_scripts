#!/usr/bin/env python3
"""
Robot Grasping Demonstration with Franka Emika Panda
Uses Genesis simulator with built-in Franka model
Every steps has to be coorelated with the previous phase
"""

import genesis as gs
import numpy as np


def main():
    # Initialize Genesis with Metal backend for M1 Pro
    gs.init(
        backend=gs.metal,
        precision='32',
        logging_level= 'warning',   
    )
    
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
        ),
        show_viewer=True,  # Enable visual viewer window
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(1.5, -1.5, 1.2),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=45,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.3,
            show_link_frame=True,
            link_frame_size=0.1,
            show_cameras=False,
        ),
        renderer=gs.renderers.Rasterizer(),
    )
    
    # Add ground
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.8),
        surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5, 1.0)),
    )
    
    # Add Franka Emika Panda robot
    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )
    
    # Target object to grasp - positioned at low table height for easier reach
    target = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.1), 
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            friction=0.8,
        ),
        surface=gs.surfaces.Smooth(color=(0.9, 0.3, 0.2, 1.0)),
        vis_mode='visual',  # Ensure cube frame is visible
    )
    
    # Add camera for recording
    cam = scene.add_camera(
        res=(1280, 720),      # HD resolution
        pos=(1.5, -1.5, 1.2), # Same as viewer camera
        lookat=(0.0, 0.0, 0.2),
        fov=45,
    )
    
    # Build scene
    scene.build()
    
    # Get robot DOF information
    n_dofs = franka.n_dofs
    print(f"Franka Panda robot loaded with {n_dofs} DOFs")
    print("Running grasping demonstration with Franka Emika Panda robot...")
    print("Recording video...")
    
    # Define joint positions for different phases
    # Initial home position - ready position
    home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    
    # Pre-grasp position - above object with open gripper
    above_object_qpos = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785, 0.04, 0.04])
    
    # Grasp position - reaching 2cm lower to object level with open gripper
    at_object_qpos = np.array([0.0, 0.75, 0.0, -1.8, 0.0, 2.6, 0.785, 0.04, 0.04])
    
    # Grasp with closed gripper - fingers press against cube (tightened for contact)
    grasp_closed_qpos = np.array([0.0, 0.75, 0.0, -1.8, 0.0, 2.6, 0.785, 0.015, 0.015])
    
    # Lift position - maintain firm grasp on cube
    lift_qpos = np.array([0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.785, 0.015, 0.015])
    
    # Start recording
    cam.start_recording()
    
    # Run simulation with different phases - 6 phases for smooth grasping
    num_steps = 1000
    steps_per_phase = num_steps // 6
    
    for i in range(num_steps):
        # Phase 1: Move to home position (0-166 steps)
        if i < steps_per_phase:
            alpha = i / steps_per_phase
            target_qpos = home_qpos
            franka.control_dofs_position(target_qpos[:n_dofs], np.arange(n_dofs))
            if i == 0:
                print("Phase 1: Moving to home position...")
            
        # Phase 2: Move above object (166-333 steps)
        elif i < 2 * steps_per_phase:
            alpha = (i - steps_per_phase) / steps_per_phase
            target_qpos = home_qpos + alpha * (above_object_qpos - home_qpos)
            franka.control_dofs_position(target_qpos[:n_dofs], np.arange(n_dofs))
            if i == steps_per_phase:
                print("Phase 2: Moving above object...")
            
        # Phase 3: Lower down to object (333-500 steps)
        elif i < 3 * steps_per_phase:
            alpha = (i - 2 * steps_per_phase) / steps_per_phase
            target_qpos = above_object_qpos + alpha * (at_object_qpos - above_object_qpos)
            franka.control_dofs_position(target_qpos[:n_dofs], np.arange(n_dofs))
            if i == 2 * steps_per_phase:
                print("Phase 3: Lowering to grasp position...")
            
        # Phase 4: Close gripper (500-666 steps)
        elif i < 4 * steps_per_phase:
            alpha = (i - 3 * steps_per_phase) / steps_per_phase
            target_qpos = at_object_qpos + alpha * (grasp_closed_qpos - at_object_qpos)
            franka.control_dofs_position(target_qpos[:n_dofs], np.arange(n_dofs))
            if i == 3 * steps_per_phase:
                print("Phase 4: Closing gripper...")
            
        # Phase 5: Lift object (666-833 steps)
        elif i < 5 * steps_per_phase:
            alpha = (i - 4 * steps_per_phase) / steps_per_phase
            target_qpos = grasp_closed_qpos + alpha * (lift_qpos - grasp_closed_qpos)
            franka.control_dofs_position(target_qpos[:n_dofs], np.arange(n_dofs))
            if i == 4 * steps_per_phase:
                print("Phase 5: Lifting object...")
                
        # Phase 6: Hold (833-1000 steps)
        else:
            target_qpos = lift_qpos
            franka.control_dofs_position(target_qpos[:n_dofs], np.arange(n_dofs))
            if i == 5 * steps_per_phase:
                print("Phase 6: Holding object...")
        
        scene.step()
        
        # Render frame for recording
        cam.render()
        
        if i % 200 == 0:  # Print progress every 200 steps
            print(f"Progress: {i}/{num_steps} steps")
    
    # Stop recording and save video
    print("Saving video...")
    cam.stop_recording(save_to_filename='simulation_video.mp4', fps=60)
    
    print("Simulation complete!")
    print("Video saved as: simulation_video.mp4")


if __name__ == "__main__":
    main()


