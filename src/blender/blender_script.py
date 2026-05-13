import blenderproc as bproc
import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
from skspatial.objects import Line, Sphere
import argparse, bpy, math, pickle, cv2, os, sys, shutil
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from blenderproc.python.material.MaterialLoaderUtility import convert_to_materials
sys.path.insert(0, os.path.abspath(__file__+'/../../..'))
from src.blender.movement import animation
from src.hdr2ldr import hdr2ldr, tone_mapping
from src.utils import safe_sample
import yaml
import json
import random
from pathlib import Path
from blenderproc.python.utility.Utility import Utility
from scipy.spatial.transform import Rotation as R
import glob
"""
BlinkSim Blender Rendering Script

This script renders scenes using Blender for event camera simulation.

OUTPUT FILES STRUCTURE:
----------------------
output/{mode}/{seq_id}/
├── rgb_reference/      # Motion-blurred RGB at rgb_image_fps (e.g., 11 fps)
│   └── ####.png        # Used for reference video and optical flow computation
├── rgb_event_input/    # Clean HDR->LDR frames at event_image_fps (e.g., 301 fps)
│   └── ####.png        # High framerate input for event camera simulation
├── hdf5/
│   ├── rgb_and_flow/   # HDF5 containing motion-blurred RGB + optical flow
│   │   └── #.hdf5      # blur, forward_flow, backward_flow at rgb_image_fps
│   └── event_input/    # HDF5 files containing clean HDR images
│       └── #.hdf5      # HDR data at event_image_fps for DVS simulation
├── dvs_events/         # Event camera data (created by main.py from event_input)
├── optical_flow/       # Forward optical flow visualization (created by main.py)
├── hdr.mp4             # Video compiled from HDR frames (created by main.py)
└── debug_scene.blend   # Optional: Blender scene file (if save_blend_file: true)

RENDERING PIPELINE:
------------------
1. First pass: Render motion-blurred frames at rgb_image_fps with optical flow
2. Second pass: Render clean frames at event_image_fps for event simulation
3. Post-processing: Convert HDR to events using DVS simulation
"""


config = None

def sample_pose(obj: bproc.types.MeshObject):
    global config
    world_length = config['world_length']
    world_width = config['world_width']

    # Sample the spheres location above the surface
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))
    obj.set_location([0,0,0])
    bbox = obj.get_bound_box()
    min_x, max_x = min(bbox[:,0]), max(bbox[:,0])
    min_y, max_y = min(bbox[:,1]), max(bbox[:,1])
    min_z, max_z = min(bbox[:,2]), max(bbox[:,2])
    min_z = min_z if min_z < 0 else 0
    min_x, max_x = -world_length - min_x, world_length - max_x
    min_y, max_y = -world_width - min_y, world_width - max_y
    min_height, max_height = 1+abs(min_z), 4+abs(min_z)
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    z = random.uniform(min_height, max_height)
    obj.set_location([x,y,z])




def euler_from_height_dependent_pitch(cam_position, target, z_reference=8.0, base_pitch_deg=90.0, 
                                       pitch_scale=2.0, up=None):
    """
    Compute camera euler angles with height-dependent pitch towards target.
    
    Camera aims towards target in XY plane, with pitch depending on Z height.
    - At Z = z_reference: pitch = base_pitch_deg (typically 90° for straight down)
    - Above z_reference: pitch increases (looking more down)
    - Below z_reference: pitch decreases (looking more horizontally)
    
    Args:
        cam_position: Camera [x, y, z]
        target: Target point [x, y, z]
        z_reference: Z value where pitch = base_pitch_deg (default 8.0)
        base_pitch_deg: Pitch angle at z_reference in degrees (default 90.0 = straight down)
        pitch_scale: How much pitch changes per unit of Z, in degrees/unit (default 2.0)
        up: Up vector for camera (default [0, 0, 1] for Blender)
    
    Returns:
        Euler angles [X, Y, Z] in radians for Blender
    """
    if up is None:
        up = np.array([0, 0, 1])
    else:
        up = np.array(up)
    
    cam_pos = np.array(cam_position)
    target_pos = np.array(target)
    
    # Height-dependent pitch: compute pitch angle based on camera Z position
    cam_z = cam_pos[2]
    pitch_deg = base_pitch_deg + pitch_scale * (cam_z - z_reference)

    # Convert legacy pitch into a stable downward-tilt angle:
    # - <= 90 deg: level (no downward tilt)
    # - > 90 deg: progressively tilt down, clamped for stability
    down_tilt_deg = np.clip(max(0.0, pitch_deg - 90.0), 0.0, 89.0)
    down_tilt_rad = np.radians(down_tilt_deg)
    
    # Compute horizontal direction from camera towards target in XY plane
    horiz_dir = target_pos[:2] - cam_pos[:2]
    horiz_dist = np.linalg.norm(horiz_dir)
    
    if horiz_dist > 0.001:
        horiz_dir = horiz_dir / horiz_dist
    else:
        # If camera is directly above target in XY, default to looking in +Y direction
        horiz_dir = np.array([0, 1])
    
    # Create an effective look-at point by combining:
    # - Horizontal aiming (towards target's XY projection)
    # - Height-dependent pitch
    # This look-at point is roughly where the camera should be aiming
    look_ahead_dist = 50.0  # How far ahead to compute the look-at point
    forward_horiz = horiz_dir * look_ahead_dist
    # Vertical component from bounded downward tilt.
    # At/under z_reference this is 0 (more level); above z_reference it tilts down smoothly.
    forward_vert = -look_ahead_dist * np.tan(down_tilt_rad)
    
    # Effective look-at point
    effective_lookat = cam_pos + np.array([forward_horiz[0], forward_horiz[1], forward_vert])
    
    # Now use the standard euler_from_look_at to compute the rotation
    return euler_from_look_at(cam_pos, effective_lookat, up)

def euler_from_look_at(position, target, up):
    forward = np.subtract(target, position)
    forward = np.divide( forward, np.linalg.norm(forward) )

    right = np.cross( forward, up )
    
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array( [0.001, 0, 0] )
        right = np.cross( forward, up + epsilon )
        
    right = np.divide( right, np.linalg.norm(right) )
    
    up = np.cross( right, forward )
    up = np.divide( up, np.linalg.norm(up) )

    T = np.array([[right[0], up[0], -forward[0], position[0]], 
                    [right[1], up[1], -forward[1], position[1]], 
                    [right[2], up[2], -forward[2], position[2]],
                    [0, 0, 0, 1]]) 
    euler = R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False)

    return euler


def model_center_world(mesh_objs, frame=None):
    """Estimate model center in world coordinates using evaluated mesh bounds."""
    if mesh_objs is None or len(mesh_objs) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    scene = bpy.context.scene
    original_frame = scene.frame_current
    if frame is not None:
        scene.frame_set(int(frame))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    pts = []

    for obj in mesh_objs:
        bpy_obj = obj.blender_obj if hasattr(obj, 'blender_obj') else obj
        try:
            eval_obj = bpy_obj.evaluated_get(depsgraph)
            mw = eval_obj.matrix_world
            for corner in eval_obj.bound_box:
                p = mw @ Vector(corner)
                pts.append([p.x, p.y, p.z])
        except Exception:
            continue

    if frame is not None:
        scene.frame_set(original_frame)

    if len(pts) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    return np.mean(np.asarray(pts, dtype=float), axis=0)


def _find_primary_armature(mesh_objs):
    """Find the armature driving the imported human meshes."""
    for obj in mesh_objs:
        bpy_obj = obj.blender_obj if hasattr(obj, 'blender_obj') else obj
        parent = getattr(bpy_obj, 'parent', None)
        if parent is not None and parent.type == 'ARMATURE':
            return parent
    for arm in bpy.data.objects:
        if arm.type == 'ARMATURE':
            return arm
    return None


def bone_world_position(mesh_objs, bone_name='Hips', frame=None):
    """Return a pose-bone world position (head), fallback to model center if unavailable."""
    armature = _find_primary_armature(mesh_objs)
    if armature is None:
        return model_center_world(mesh_objs, frame=frame)

    scene = bpy.context.scene
    original_frame = scene.frame_current
    if frame is not None:
        scene.frame_set(int(frame))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = armature.evaluated_get(depsgraph)
    pose_bones = arm_eval.pose.bones

    # Prefer exact name, then common hips aliases, then fuzzy fallback.
    candidates = [
        bone_name,
        'Hips',
        'hips',
        'mixamorig:Hips',
        'pelvis',
        'Pelvis',
    ]
    pb = None
    for name in candidates:
        if name in pose_bones:
            pb = pose_bones[name]
            break
    if pb is None:
        for name in pose_bones.keys():
            lname = name.lower()
            if 'hip' in lname or 'pelvis' in lname:
                pb = pose_bones[name]
                break

    if pb is None:
        if frame is not None:
            scene.frame_set(original_frame)
        return model_center_world(mesh_objs, frame=frame)

    world_head = arm_eval.matrix_world @ pb.head

    if frame is not None:
        scene.frame_set(original_frame)

    return np.array([world_head.x, world_head.y, world_head.z], dtype=float)


def point_view_coords(cam_obj, point_world, cam_location, cam_euler):
    """Project a world point into normalized camera view coordinates for a hypothetical camera pose."""
    scene = bpy.context.scene
    old_loc = cam_obj.location.copy()
    old_rot = cam_obj.rotation_euler.copy()
    try:
        cam_obj.location = Vector(cam_location)
        cam_obj.rotation_euler = cam_euler
        p = Vector(point_world)
        co = world_to_camera_view(scene, cam_obj, p)
        return float(co.x), float(co.y), float(co.z)
    finally:
        cam_obj.location = old_loc
        cam_obj.rotation_euler = old_rot


def target_near_edge(cam_obj, point_world, cam_location, cam_euler, edge_margin=0.15):
    """True if target is near/outside view edge or behind camera."""
    x, y, z = point_view_coords(cam_obj, point_world, cam_location, cam_euler)
    if z <= 0.0:
        return True
    return (x < edge_margin or x > 1.0 - edge_margin or y < edge_margin or y > 1.0 - edge_margin)

def make_action_cyclic(action):
    """Add cyclic modifiers so an action loops for the full render duration."""
    if action is None:
        return
    for fcu in action.fcurves:
        mod = fcu.modifiers.new(type='CYCLES')
        mod.mode_before = 'REPEAT'
        mod.mode_after = 'REPEAT'


def load_human_fbx(model_path, animation_path=None):
    """
    Load a human FBX model and optionally apply animation.
    
    Args:
        model_path: Path to the human model FBX file
        animation_path: Optional path to animation FBX file
    
    Returns:
        List of imported mesh objects wrapped in BlenderProc MeshObjects
    """
    # Import the human model FBX using Blender's FBX importer
    # Enable texture search to load texture files from the same directory
    model_dir = os.path.dirname(model_path)
    bpy.ops.import_scene.fbx(
        filepath=model_path,
        use_image_search=True,  # Search for textures in subdirectories
        directory=model_dir      # Start search from model directory
    )
    
    # Get the imported objects (they are now selected)
    imported_objects = list(bpy.context.selected_objects)
    
    # Find the armature (root of the rig) - transformations should be applied here
    armature = None
    for obj in imported_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    # Keep imported character exactly as authored in FBX.
    # Do not modify location/rotation/scale at import time.
    
    # Load and apply animation if provided
    animation_action = None
    locomotion = False
    if animation_path and os.path.exists(animation_path):
        locomotion_keywords = config.get('locomotion_keywords', ['run', 'walk', 'jog', 'sprint'])
        fname = os.path.basename(animation_path).lower()
        locomotion = any(k in fname for k in locomotion_keywords) and not ('strafe' in fname)
        # Import animation FBX (this contains the armature with animation data)
        bpy.ops.import_scene.fbx(filepath=animation_path)
        anim_objs = bpy.context.selected_objects
        
        # Find the armature in the animation file
        anim_armature = None
        for obj in anim_objs:
            if obj.type == 'ARMATURE':
                anim_armature = obj
                break
        
        if anim_armature and armature:
            # Copy animation data from animation armature to model armature
            if anim_armature.animation_data and anim_armature.animation_data.action:
                if not armature.animation_data:
                    armature.animation_data_create()
                
                # Copy the action (animation)
                animation_action = anim_armature.animation_data.action
                armature.animation_data.action = animation_action
                anim_start = float(animation_action.frame_range[0])
                anim_end = float(animation_action.frame_range[1])
                config['anim_frame_start'] = anim_start
                config['anim_frame_length'] = anim_end - anim_start

                print(f"Animation loaded with {len(animation_action.fcurves)} fcurves")
                
                # Get animation frame range
                anim_start = int(animation_action.frame_range[0])
                anim_end = int(animation_action.frame_range[1])
                anim_length = anim_end - anim_start

                # Store animation range for later retiming
                armature["anim_start"] = anim_start
                armature["anim_end"] = anim_end
                armature["anim_length"] = anim_length
        
        # Remove animation objects as we only needed their animation data
        for obj in anim_objs:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Record locomotion flag only; camera logic can use it if needed.
    if animation_path and locomotion:
        config['last_is_locomotion'] = True

    # Wrap mesh objects in BlenderProc MeshObjects
    mesh_objs = []
    for obj in imported_objects:
        if obj.type == 'MESH':
            bp_obj = bproc.types.MeshObject(obj)
            mesh_objs.append(bp_obj)
            
            # Convert FBX materials to Blender node-based materials
            # This ensures materials render correctly in Eevee
            for mat_slot in obj.material_slots:
                if mat_slot.material:
                    mat = mat_slot.material
                    # Ensure material uses nodes
                    if not mat.use_nodes:
                        mat.use_nodes = True
                    
                    # For Eevee, ensure blend mode is set correctly
                    mat.blend_method = 'OPAQUE'
                    mat.shadow_method = 'OPAQUE'
            
            # If no armature, keep mesh transform untouched as imported.
    
    return mesh_objs



def check_obj_hit(obj, origin, direction, max_distance):
    world2local = np.linalg.inv(obj.get_local2world_mat())
    origin_in_obj = world2local@origin.T
    dir_in_obj = world2local@direction.T
    origin_in_obj = origin_in_obj[:3] * obj.get_scale(0)
    dir_in_obj = dir_in_obj[:3] * obj.get_scale(0)
    hit, location, _, _ = obj.blender_obj.ray_cast(origin_in_obj, dir_in_obj, distance=max_distance)
    return hit, location

def filter_range(ori,objs_list, r):
    '''filter static objects within a certain range of the camera'''
    r_list = compute_radius(objs_list, change_origin = False)
    filter_list = []
    for index, obj in enumerate(objs_list):
        pos = obj.get_location()
        dist = np.linalg.norm(pos-ori)
        # print(dist)
        if dist-r_list[index] > r:
            filter_list.append(obj)
        else:
            obj.set_location([1e3,1e3,1e3])
    return filter_list

def test_obj_hist():
    obj = bproc.object.create_primitive(shape='CUBE')
    obj.set_location([0,0,100])
    obj.set_scale([2,3,4])
    obj.set_rotation_euler(np.array([-0.4636476, 0.111341, -0.4636476]))
    origin = np.array([0,5,100,1])
    direction = np.array([0,-1,0,0])
    for max_distance in range(0, 7):
        hit, location = check_obj_hit(obj, origin, direction, max_distance)
        print(max_distance)
        print(hit)
        print(location)
    exit(0)

def setup_placement(init_pose):
    global config
    world_length = config['world_length']
    world_width = config['world_width']
    canopy_distance = config['canopy_distance']
    canopy_height = config['canopy_height']

    # test_obj_hist()

    ground_plane = bproc.object.create_primitive(shape='PLANE')
    ground_plane.set_scale([world_length+canopy_distance,world_width+canopy_distance,1])
    ground_plane.set_location([0,0,0])
    # remove_shadow(ground_plane)
    objs_list = []
    num_static_obj = random.randint(*config['num_static_obj'])
    for i in range(num_static_obj):
        if random.random() < 0.5:
            obj = bproc.object.create_primitive(shape='CYLINDER')
        else:
            obj = bproc.object.create_primitive(shape='CUBE')
        if random.random() < 0.5:
            random_length = random.uniform(1.0, 4.0)
            random_height = random.uniform(1.0, 8.0)
        else:
            random_length = random.uniform(1.0, 8.0)
            random_height = random.uniform(1.0, 4.0)
        obj.set_scale([random_length, random_length, random_height])
        # remove_shadow(obj)
        objs_list.append(obj)

    # for obj in objs_list:
    #     sample_pose(obj)
    bproc.object.sample_poses_on_surface(
        objs_list,
        ground_plane,
        sample_pose,
        min_distance=0.1,
        max_distance=400,
        up_direction=[0,0,1]
    )

    for obj in objs_list:
        obj.enable_rigidbody(True)
    ground_plane.enable_rigidbody(False)

    # Run the physics simulation
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=2,
        max_simulation_time=8,
        check_object_interval=5
    )

    objs_list = filter_range(init_pose[0], objs_list, 20)

    # stitching ground plane
    bpy.data.objects.remove(ground_plane.blender_obj, do_unlink=True)
    plane_list = []
    # part_num = random.randint(1, 10)
    part_num = int(random.uniform(1, 3) ** 2)
    for i in range(part_num):
        for j in range(part_num):
            plane = bproc.object.create_primitive(shape='PLANE')
            length = math.ceil((world_length+canopy_distance)/part_num)
            width = math.ceil((world_width+canopy_distance)/part_num)
            plane.set_scale([length, width, 1])
            y = (2*i+1-part_num) * length
            x = (2*j+1-part_num) * width
            plane.set_location([y,x,0])
            plane_list.append(plane)

    canopy_list = []
    _scale_list = [
        [math.ceil((world_length+canopy_distance)/part_num),int(canopy_height/2),1],
        [math.ceil((world_length+canopy_distance)/part_num),int(canopy_height/2),1],
        [int(canopy_height/2),math.ceil((world_width+canopy_distance)/part_num),1],
        [int(canopy_height/2),math.ceil((world_width+canopy_distance)/part_num),1],
    ]
    _loc_list = [
        [0,-world_width-canopy_distance,int(canopy_height/2)],
        [0,world_width+canopy_distance,int(canopy_height/2)],
        [-world_length-canopy_distance,0,int(canopy_height/2)],
        [world_length+canopy_distance,0,int(canopy_height/2)],
    ]
    _rot_list = [
        [math.pi/2,0,0],
        [-math.pi/2,0,0],
        [0,-math.pi/2,0],
        [0,math.pi/2,0],
    ]
    for i in range(4):
        for j in range(part_num):
            length = math.ceil((world_length+canopy_distance)/part_num)
            y = (2*j+1-part_num) * length
            loc = _loc_list[i]
            loc = [y if math.fabs(item) < 1e-6 else item for item in loc]

            canopy = bproc.object.create_primitive(shape='PLANE')
            canopy.set_scale(_scale_list[i])
            # canopy.set_location(_loc_list[i])
            canopy.set_location(loc)
            canopy.set_rotation_euler(_rot_list[i])
            # remove_shadow(canopy)
            canopy_list.append(canopy)

    # return ground_plane, objs_list, canopy_list
    return plane_list, objs_list, canopy_list

def remove_shadow(obj):
    if obj.has_materials():
        for i in range(len(obj.blender_obj.data.materials)):
            obj.blender_obj.data.materials[i].shadow_method = 'NONE'
    else:
        obj.new_material("material_0")
        obj.blender_obj.data.materials[0].shadow_method = 'NONE'

def setup_material(ground, static_objs, canopys, mode):
    global config
    image_dir = Path(config['image_dir'])
    texture_split_file = config['texture_split_file']
    
    with open(texture_split_file, 'r') as fr:
        rel_path = json.load(fr)[mode]
        images = [image_dir.joinpath(rp) for rp in rel_path]
    skip_num = 1 + len(canopys)
    # objs = [ground] + canopys + static_objs
    objs = ground + canopys + static_objs
    for index, obj in enumerate(objs):
        if not obj.has_materials():
            obj.new_material("material_0")
        material_0 = obj.get_materials()[0]
        # add image texture as base color
        image = bpy.data.images.load(filepath=str(random.choice(images)))
        material_0.set_principled_shader_value("Base Color", image)
        # insert hsv node, to post-processing texture in hsv color space
        # do not process ground plane
        if index < skip_num: continue
        hsv_node = material_0.new_node(node_type="ShaderNodeHueSaturation")
        random_mode = random.random()
        min_hsv_range = config['min_hsv_range']
        max_hsv_range = config['max_hsv_range']
        hsv_dark_prob = config['hsv_dark_prob']
        if random_mode < hsv_dark_prob: # dark
            random_hsv_value = random.uniform(min_hsv_range[0], min_hsv_range[1])
        else: # bright
            random_hsv_value = random.uniform(max_hsv_range[0], max_hsv_range[1])
        hsv_node.inputs["Value"].default_value = random_hsv_value
        src_node = material_0.get_the_one_node_with_type("TexImage")
        dst_node = material_0.get_the_one_node_with_type("BsdfPrincipled")
        Utility.insert_node_instead_existing_link(
            material_0.links,
            src_node.outputs["Color"],
            hsv_node.inputs["Color"],
            hsv_node.outputs["Color"],
            dst_node.inputs["Base Color"]
        )

def setup_lighting(init_pose):
    # tri-light setup
    up = np.array([0, 0, 1])

    pos0 = np.array([42, -50, 32])
    target0 = np.array([10, -30, 0])
    light0 = bproc.types.Light()
    light0.set_type("AREA")
    light0.set_location(pos0)
    light0.set_rotation_euler(euler_from_look_at(pos0, target0, up))
    light0.set_energy(10e4)

    pos1 = np.array([-64, -13, 40])
    target1 = np.array([-30, 5, 0])
    light1 = bproc.types.Light()
    light1.set_type("AREA")
    light1.set_location(pos1)
    light1.set_rotation_euler(euler_from_look_at(pos1, target1, up))
    light1.set_energy(5e4)

    pos2 = np.array([-20, 58, 40])
    target2 = np.array([-10, 30, 0])
    light2 = bproc.types.Light()
    light2.set_type("AREA")
    light2.set_location(pos2)
    light2.set_rotation_euler(euler_from_look_at(pos2, target2, up))
    light2.set_energy(1e4)

    pos, target, euler = init_pose
    inner_rad = random.uniform(1, 3)
    outer_rad = random.uniform(3, 15)
    _alpha1 = random.uniform(0, 2*math.pi)
    target_offset = np.array([math.cos(_alpha1)*inner_rad, math.sin(_alpha1)*inner_rad, random.uniform(0,3)])
    _alpha2 = random.uniform(0, 2*math.pi)
    pos_offset = np.array([math.cos(_alpha2)*outer_rad, math.sin(_alpha2)*outer_rad, random.uniform(20,40)])
    light3 = bproc.types.Light()
    light3.set_type("AREA")
    light3.set_location(pos+pos_offset)
    light3.set_rotation_euler(euler_from_look_at(pos+pos_offset, pos+target_offset, up))
    light3.set_energy(3e4)


def setup_fill_light(cam_position, look_at):
    """Add an area light near the camera to lift shadows on the subject."""
    global config
    if not config.get('use_fill_light', True):
        return
    energy = config.get('fill_light_energy', 100)
    energy = random.uniform(energy[0], energy[1]) if isinstance(energy, list) else energy
    distance = config.get('fill_light_distance', 2.5)
    height = config.get('fill_light_height', 1.5)
    size = config.get('fill_light_size', 2.0)
    up = np.array([0, 0, 1])

    cam_position = np.array(cam_position, dtype=float)
    look_at = np.array(look_at, dtype=float)
    mode = config.get('fill_light_position_mode', 'camera_offset')

    if mode == 'cube_random':
        cube_extent = float(config.get('camera_cube_extent', 25.0))
        full_cube_z = bool(config.get('fill_light_cube_full_z', True))
        min_sep = float(config.get('fill_light_cube_min_separation', 3.0))
        max_attempts = int(config.get('fill_light_cube_max_attempts', 30))

        if full_cube_z:
            z_min, z_max = -cube_extent, cube_extent
        else:
            z_cfg = config.get('camera_z_range', [5.0, 50.0])
            z_min, z_max = float(z_cfg[0]), float(z_cfg[1])

        pos = None
        for _ in range(max_attempts):
            candidate = np.array([
                random.uniform(-cube_extent, cube_extent),
                random.uniform(-cube_extent, cube_extent),
                random.uniform(z_min, z_max),
            ], dtype=float)
            if np.linalg.norm(candidate - cam_position) >= min_sep:
                pos = candidate
                break

        if pos is None:
            # Fallback if constraints are too strict.
            pos = np.array([
                random.uniform(-cube_extent, cube_extent),
                random.uniform(-cube_extent, cube_extent),
                random.uniform(z_min, z_max),
            ], dtype=float)
    else:
        dir_flat = look_at - cam_position
        dir_flat[2] = 0.0
        norm = np.linalg.norm(dir_flat)
        if norm < 1e-6:
            dir_flat = np.array([0, 1, 0], dtype=float)
        else:
            dir_flat /= norm

        pos = cam_position + dir_flat * distance + np.array([0, 0, height])

    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location(pos)
    light.set_rotation_euler(euler_from_look_at(pos, look_at, up))
    try:
        light.set_size(size)
    except Exception:
        pass
    light.set_energy(energy)


def normalized(a):
    length = np.linalg.norm(a)
    return a / length

def spline_from_3pose(init_pose, up):
    pos, target, euler = init_pose
    cam_pose_list = [[pos, target, euler]]

    # last pose
    _alpha = random.uniform(0, 2*math.pi)
    _theta = random.uniform(0, math.pi/5*2)
    _z = math.cos(_theta)
    _xy_len = math.sin(_theta)
    _x, _y = math.cos(_alpha)*_xy_len, math.sin(_alpha)*_xy_len
    up_new = np.array([_x, _y, _z])
    pos_offset = np.random.uniform([2.0,2.0,2.0], [6.0,6.0,6.0])
    target_offset = np.random.uniform([1.0,1.0,1.0], [3.0,3.0,3.0])

    _pos = pos + pos_offset/2 + np.random.uniform([-1.0,-1.0,-1.0], [1.0,1.0,1.0])
    _target = target + target_offset/2 + np.random.uniform([-0.5,-0.5,-0.5], [0.5,0.5,0.5])
    _up = normalized((up + up_new) / 2)
    euler = euler_from_look_at(_pos, _target, _up)
    cam_pose_list.append([_pos, _target, euler])

    _pos = pos + pos_offset
    _target = target + target_offset
    _up = up_new
    euler = euler_from_look_at(_pos, _target, _up)
    cam_pose_list.append([_pos, _target, euler])

    return cam_pose_list

def linear_pose(init_pose, up, num_kf):
    pos, target, euler = init_pose
    cam_pose_list = [[pos, target, euler]]

    min_height = 12-3
    max_height = 12+5
    for frame_idx in range(num_kf-1):
        # pos_offset = np.random.uniform([-6.0,-6.0,-3.0], [6.0,6.0,3.0])
        # target_offset = np.random.uniform([-2.0,-2.0,0.0], [2.0,2.0,0.0])
        pos_offset = np.random.uniform([0.0,0.0,-3.0], [5.0,5.0,3.0])
        target_offset = np.random.uniform([-2.0,-2.0,0.0], [2.0,2.0,0.0])
        pos_offset[0] = pos_offset[0] if pos_offset[0] < 1 else pos_offset[0]**2
        pos_offset[1] = pos_offset[1] if pos_offset[1] < 1 else pos_offset[1]**2
        sign = 1 if random.random() < 0.5 else -1
        pos_offset[:2] = sign * pos_offset[:2]
        pos = pos + pos_offset
        target = target + target_offset
        pos[2] = np.clip(pos[2], a_min=min_height, a_max=max_height)
        euler = euler_from_look_at(pos, target, up)
        cam_pose_list.append([pos, target, euler])
    
    return cam_pose_list

def setup_camera_extrinsic():
    global config
    num_kf = config['num_keyframes']

    theta = random.uniform(0, math.pi*2)
    radius = 49
    height = 12
    look_at_radius = radius - 35

    # init pose
    pos = np.array([radius*math.sin(theta), radius*math.cos(theta), height])
    target = np.array([look_at_radius*math.sin(theta), look_at_radius*math.cos(theta), 0])
    up = np.array([0, 0, 1])
    euler = euler_from_look_at(pos, target, up)
    init_pose = [pos, target, euler]

    if num_kf == 3 and config['animation_mode'] == 'cubinc_spline':
        cam_pose_list = spline_from_3pose(init_pose, up)
    else:
        cam_pose_list = linear_pose(init_pose, up, num_kf)

    return cam_pose_list

def load_taxonomy_list(path):
    name = f'{path}/taxonomy.json'
    l = []
    with open(name, 'r') as fr:
        d = json.load(fr)
        for i in range(len(d)):
            if d[i]['numInstances'] < 500:
                l.append(d[i]['synsetId'])
    return l

def correct_materials(obj):
    """ If the used material contains an alpha texture, the alpha texture has to be flipped to be correct

    :param obj: object where the material maybe wrong
    """
    for material in obj.get_materials():
        if material is None:
            continue
        texture_nodes = material.get_nodes_with_type("ShaderNodeTexImage")
        if texture_nodes and len(texture_nodes) > 1:
            principled_bsdf = material.get_the_one_node_with_type("BsdfPrincipled")
            # find the image texture node which is connect to alpha
            node_connected_to_the_alpha = None
            for node_links in principled_bsdf.inputs["Alpha"].links:
                if "ShaderNodeTexImage" in node_links.from_node.bl_idname:
                    node_connected_to_the_alpha = node_links.from_node
            # if a node was found which is connected to the alpha node, add an invert between the two
            if node_connected_to_the_alpha is not None:
                invert_node = material.new_node("ShaderNodeInvert")
                invert_node.inputs["Fac"].default_value = 1.0
                material.insert_node_instead_existing_link(node_connected_to_the_alpha.outputs["Color"],
                                                            invert_node.inputs["Color"],
                                                            invert_node.outputs["Color"],
                                                            principled_bsdf.inputs["Alpha"])

def load_shapenet_obj(filepath, move_object_origin=True):
    loaded_objects = bproc.loader.load_obj(filepath)

    # In shapenet every .obj file only contains one object, make sure that is the case
    if len(loaded_objects) != 1:
        raise Exception("The ShapeNetLoader expects every .obj file to contain exactly one object")
    obj = loaded_objects[0]

    correct_materials(obj)

    # removes the x axis rotation found in all ShapeNet objects, this is caused by importing .obj files
    # the object has the same pose as before, just that the rotation_euler is now [0, 0, 0]
    obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)

    # check if the move_to_world_origin flag is set
    if move_object_origin:
        # move the origin of the object to the world origin and on top of the X-Y plane
        # makes it easier to place them later on, this does not change the `.location`
        obj.move_origin_to_bottom_mean_point()
    bpy.ops.object.select_all(action='DESELECT')

    return obj

def filter_obj_by_volume(obj_list, thres=0.80, keep_num=5):
    filter_list = []
    for obj in obj_list:
        bb = obj.get_bound_box()
        min_point, max_point = bb[0], None
        max_dist = -1
        for point in bb:
            dist = np.linalg.norm(point - min_point)
            if dist > max_dist:
                max_point = point
                max_dist = dist
        diag = max_point - min_point
        if diag.max() < thres:
            filter_list.append(obj)
    return filter_list[:keep_num]

def normaliz_vec(vec):
    vec = vec / np.linalg.norm(vec)
    return vec

def sphere_seg_inter(start, end, r):
    '''calculate the intersection of a sphere with a certain radius and a line segment'''
    line_vec = end - start
    sphere = Sphere([0, 0, 0], r)
    line = Line(start, line_vec)
    point_a = None
    point_b = None
    try:
        point_a, point_b = sphere.intersect_line(line)
    except:
        pass
    if point_a is not None:
        sa = (point_a - start)/line_vec
        sb = (point_b - start)/line_vec
        sa = sa[~np.isnan(sa)]
        sb = sb[~np.isnan(sb)]
        t = min(np.median(sa),np.median(sb))
    else:
        t = 2.0
    return t

def check_end_clip(objs_radius, obj_pose_list, obj_idx, pos, valid_object):
    '''check whether the motion track coincides with that of other objects'''
    min_t = 1.0
    is_overlap = False
    for check_idx in range(obj_idx):
        if len(obj_pose_list[check_idx]) != 2 or (not valid_object[check_idx]):
            continue
        a0 = obj_pose_list[check_idx][0][0]
        a1 = obj_pose_list[check_idx][1][0]
        b0 = obj_pose_list[obj_idx][0][0]
        b1 = pos
        t = sphere_seg_inter(a0-b0, a1-b1, objs_radius[obj_idx] + objs_radius[check_idx])
        dist = np.linalg.norm(b1-a1)
        if dist < objs_radius[obj_idx] + objs_radius[check_idx]:
            is_overlap = True
        if t>0.0 and t <1.0:
            min_t = min(min_t, t)
            is_overlap = True
    return min_t, is_overlap

def check_start_overlap(objs_radius, obj_pose_list, obj_idx, pos, valid_object):
    '''check whether the starting point coincides with other objects'''
    for check_idx in range(obj_idx):
        if (not valid_object[check_idx]):
            continue
        a0 = obj_pose_list[check_idx][0][0]
        b0 = pos
        dist = np.linalg.norm(b0-a0)
        if dist < objs_radius[obj_idx] + objs_radius[check_idx]:
            return True
    return False

def compute_radius(dynamic_objs, change_origin = True):
    '''calculate the object radius and set the object origin at the object center'''
    objs_radius = []
    for obj in dynamic_objs:
        bb = obj.get_bound_box()
        bb_array = np.array(bb)
        bb_min = np.min(bb_array, axis = 0)
        bb_max = np.max(bb_array, axis = 0)
        if(change_origin):
            obj.set_origin((bb_min+bb_max)/2)
            obj.set_location([0,0,0])
        bb = obj.get_bound_box()
        diag = bb_max - bb_min
        radius = np.linalg.norm(diag)/2
        objs_radius.append(radius)
    return objs_radius

def setup_dynamic_objs(cam_pose_list, mode):
    global config
    activte_range_for_dynamic_obj = config['activte_range_for_dynamic_obj']
    dynamic_obj_split_file = config['dynamic_obj_split_file']
    shape_dir = Path(config['shape_dir'])
    # shape_dir = config['shape_dir']
    
    with open(dynamic_obj_split_file, 'r') as fr:
        rel_path = json.load(fr)[mode]
        all_shapes_p = [shape_dir.joinpath(rp) for rp in rel_path]

    shape_num = random.randrange(config['shape_num'][0], config['shape_num'][1])
    shapes_p = safe_sample(all_shapes_p, shape_num * 3)
    dynamic_objs = [load_shapenet_obj(str(shape_p)) for shape_p in shapes_p]
    dynamic_objs = filter_obj_by_volume(dynamic_objs, thres=0.8, keep_num=shape_num * 3)

    obj_pose_list = [[]] * len(dynamic_objs)
    valid_object = [True]*len(dynamic_objs)
    for obj in dynamic_objs:
        obj.set_scale([random.uniform(5.0, 8.0)]*3)
        obj.enable_rigidbody(True)
        # remove_shadow(obj)
    objs_radius = compute_radius(dynamic_objs)
    for frame_idx, cam_pose in enumerate(cam_pose_list):
        cam_position, look_at = cam_pose[0], cam_pose[1]
        forward_vec = normaliz_vec(np.pad(look_at[:2]-cam_position[:2], (0,1), 'constant'))
        right_vec = np.array([-forward_vec[1], forward_vec[0], 0])
        up_vec = np.array([0, 0, 1])
        look_at = look_at -forward_vec*15 + up_vec*6

        Twb = np.eye(4)
        Twb[:3, 0] = right_vec
        Twb[:3, 1] = forward_vec
        Twb[:3, 2] = up_vec
        Twb[:3, 3] = look_at
        for obj_idx, obj in enumerate(dynamic_objs):
            if not valid_object[obj_idx]:
                continue
            max_t = -1.0
            max_pos = None
            # print('obj_idx:',obj_idx)
            for iter in range(30):
                pos = np.random.uniform([-7, -5, -5], activte_range_for_dynamic_obj)
                #pos = np.random.uniform(-activte_range_for_dynamic_obj, activte_range_for_dynamic_obj)
                pos = np.expand_dims(np.pad(pos, (0,1), 'constant', constant_values=1), axis=1)
                pos = (Twb @ pos).squeeze()[:3]
                if len(obj_pose_list[obj_idx]) == 0: # 随机起始点的时候
                    is_overlap = check_start_overlap(objs_radius, obj_pose_list, obj_idx, pos, valid_object)
                    max_pos = pos
                    if not is_overlap:
                        break # 只要和前面的没有重合，就是一个有效的起始点
                else: # 随机终点的时候
                    t, is_overlap = check_end_clip(objs_radius, obj_pose_list, obj_idx, pos, valid_object)
                    max_pos = pos
                    if not is_overlap:
                        break
            if len(obj_pose_list[obj_idx]) == 0 and is_overlap:
                valid_object[obj_idx] = False
            if len(obj_pose_list[obj_idx]) != 0 and is_overlap:
                valid_object[obj_idx] = False
            euler = np.random.uniform([0, 0, 0], [np.pi/5, np.pi/5, np.pi/5]) #np.array([0.0,0.0,0.0])
            if len(obj_pose_list[obj_idx]) == 0: obj_pose_list[obj_idx] = [[max_pos, euler]]
            else: obj_pose_list[obj_idx].append([max_pos, euler])
    valid_dynamic_objs = []
    valid_obj_pose_list = []
    for obj_idx, obj in enumerate(dynamic_objs):
        if valid_object[obj_idx]: 
            valid_dynamic_objs.append(obj)
            valid_obj_pose_list.append(obj_pose_list[obj_idx])
    return valid_dynamic_objs, valid_obj_pose_list

def setup_camera_intrinsic():
    global config
    width, height = config['image_width'], config['image_height']
    bproc.camera.set_resolution(width, height)

def setup_envmap():
    global config
    use_solid_background = config.get('use_solid_background', False)
    if use_solid_background:
        # Configure Blender world nodes for a flat monochrome background.
        randomize_solid_background = config.get('randomize_solid_background', False)
        if randomize_solid_background:
            gray_min, gray_max = config.get('solid_background_gray_range', [0.2, 0.8])
            gray = random.uniform(float(gray_min), float(gray_max))
            color = [gray, gray, gray]
        else:
            color = config.get('solid_background_color', [0.5, 0.5, 0.5])
        strength = float(config.get('solid_background_strength', 1.0))

        # Ensure world exists and uses nodes.
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        world = bpy.context.scene.world
        world.use_nodes = True

        nodes = world.node_tree.nodes
        links = world.node_tree.links

        bg_node = nodes.get('Background')
        output_node = nodes.get('World Output')
        if bg_node is None:
            bg_node = nodes.new(type='ShaderNodeBackground')
        if output_node is None:
            output_node = nodes.new(type='ShaderNodeOutputWorld')

        bg_node.inputs['Color'].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
        bg_node.inputs['Strength'].default_value = strength

        has_link = False
        for link in links:
            if link.from_node == bg_node and link.to_node == output_node:
                has_link = True
                break
        if not has_link:
            links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
        return

    hdr_dir = config['hdr_dir']
    hdr_folders = [f for f in os.listdir(hdr_dir) if os.path.isdir(os.path.join(hdr_dir, f))]
    hdr_folder = random.choice(hdr_folders)
    # hdr_folder = "autumn_forest_01"  # For testing
    
    # Find the actual .hdr file in the folder
    hdr_folder_path = os.path.join(hdr_dir, hdr_folder)
    hdr_files = [f for f in os.listdir(hdr_folder_path) if f.endswith('.hdr')]
    if hdr_files:
        hdr_file = hdr_files[0]  # Take the first .hdr file
        path = os.path.join(hdr_folder_path, hdr_file)
        bproc.world.set_world_background_hdr_img(path)
    else:
        print(f"Warning: No .hdr file found in {hdr_folder_path}")

def setup_human_env(mode):
    """
    Setup environment for human animation rendering.
    Fixed camera, no static objects, only human model with animation.
    """
    global config
    
    # Setup camera - fixed position looking at the human
    width, height = config['image_width'], config['image_height']
    bproc.camera.set_resolution(width, height)
    
    # Set camera clipping to avoid near-plane clipping
    bpy.context.scene.camera.data.clip_start = 0.1
    bpy.context.scene.camera.data.clip_end = 100.0
    
    up = [0, 0, 1]
    
    # Setup environment map for lighting
    setup_envmap()
    
    # Load human model and animation
    human_model_dir = config.get('human_model_dir', 'data/human_models')
    human_anim_dir = config.get('human_anim_dir', 'data/human_animations')
    
    # Get model and animation files
    model_files = sorted(glob.glob(f"{human_model_dir}/*.fbx"))
    anim_files = sorted(glob.glob(f"{human_anim_dir}/*.fbx"))
    
    if not model_files:
        raise ValueError(f"No FBX models found in {human_model_dir}")
    
    forced_model = config.get('forced_model_path')
    forced_anim = config.get('forced_animation_path')

    # Enforce deterministic selection from per-job config; no cycling/random fallback
    if forced_model:
        model_path = forced_model
    else:
        model_path = model_files[0]

    if forced_anim:
        anim_path = forced_anim
    else:
        anim_path = anim_files[0] if anim_files else None
    
    print(f"Loading human model: {model_path}")
    if anim_path:
        print(f"Loading animation: {anim_path}")
    
    # Load the human with animation
    human_objs = load_human_fbx(model_path, anim_path)

    # Optionally consider animation length; clamp duration to min(config duration, animation length)
    if 'anim_frame_length' in config:
        anim_len_frames = config['anim_frame_length']
        fps = config.get('rgb_image_fps', 10)
        anim_sec = float(anim_len_frames) / float(fps)
        
        # clamp_duration_to_animation: when True, render min(config_duration, animation_length)
        # when False, render full animation regardless of config_duration
        clamp_to_anim = config.get('clamp_duration_to_animation', True)
        cfg_dur = config.get('duration', anim_sec)
        
        if clamp_to_anim:
            # Clamp to whichever is shorter
            new_dur = min(cfg_dur, anim_sec)
            if abs(new_dur - cfg_dur) > 1e-6:
                print(f"Clamping duration to animation length: anim {anim_sec:.3f}s, cfg {cfg_dur:.3f}s -> {new_dur:.3f}s")
        else:
            # Use full animation length
            new_dur = anim_sec
            if abs(new_dur - cfg_dur) > 1e-6:
                print(f"Using full animation length: anim {anim_sec:.3f}s (ignoring config {cfg_dur:.3f}s)")
        
        config['duration'] = new_dur
    else:
        print("Animation frame length not found; using config duration")

    
    # Camera motion controls all randomization; model stays untouched.
    target_mode = config.get('camera_target_mode', 'hips_bone')
    if target_mode == 'hips_bone':
        target_frame = config.get('camera_target_frame', config.get('anim_frame_start', 0))
        target_bone = config.get('camera_target_bone', 'Hips')
        look_at = bone_world_position(human_objs, bone_name=target_bone, frame=target_frame).tolist()
        print(f"Camera target from bone '{target_bone}' at frame {target_frame}: {look_at}")
    elif target_mode == 'model_initial':
        target_frame = config.get('camera_target_frame', config.get('anim_frame_start', 0))
        look_at = model_center_world(human_objs, frame=target_frame).tolist()
        print(f"Camera target from model center at frame {target_frame}: {look_at}")
    else:
        look_at = config.get('camera_target', [0.0, 0.0, 0.0])
    camera_cube_extent = config.get('camera_cube_extent', 25.0)  # Half-side length of cube
    camera_z_range = config.get('camera_z_range', [5.0, 50.0])   # Z range for upper half

    # Sample random position on upper half of a cube around origin.
    # The cube extends ±camera_cube_extent in X and Y.
    # Z is sampled from camera_z_range (typically positive for "upper half").
    
    cam_x = random.uniform(-camera_cube_extent, camera_cube_extent)
    cam_y = random.uniform(-camera_cube_extent, camera_cube_extent)
    cam_z = random.uniform(float(camera_z_range[0]), float(camera_z_range[1]))

    cam_position = [cam_x, cam_y, cam_z]
    
    num_keyframes = config.get('num_keyframes', 2)
    follow_enabled = config.get('camera_follow_enabled', False)

    if follow_enabled and num_keyframes >= 2:
        target_bone = config.get('camera_target_bone', 'Hips')
        rotation_only = config.get('camera_follow_rotation_only', True)
        follow_distance = float(config.get('camera_follow_distance', 8.0))
        follow_height_offset = float(config.get('camera_follow_height_offset', 2.0))
        inertia = float(config.get('camera_follow_inertia', 0.85))
        inertia = min(max(inertia, 0.0), 0.99)

        anim_start = float(config.get('anim_frame_start', 0.0))
        anim_len = float(config.get('anim_frame_length', max(1, num_keyframes)))
        sample_frames = np.linspace(anim_start, anim_start + max(1.0, anim_len - 1.0), num_keyframes)

        hip_points = [bone_world_position(human_objs, bone_name=target_bone, frame=f) for f in sample_frames]

        cam_pose = []
        if rotation_only:
            # Keep camera fixed in place and only rotate to track hips with inertia.
            cam_fixed = np.array(cam_position, dtype=float)
            edge_track = bool(config.get('camera_follow_edge_tracking', True))
            edge_margin = float(config.get('camera_follow_edge_margin', 0.15))
            min_target_shift = float(config.get('camera_follow_min_target_shift', 0.10))
            cam_obj = bpy.context.scene.camera
            target_prev = np.array(hip_points[0], dtype=float)
            hip_prev_raw = np.array(hip_points[0], dtype=float)
            first_target = target_prev + np.array([0.0, 0.0, follow_height_offset], dtype=float)
            current_euler = euler_from_look_at(cam_fixed.tolist(), first_target.tolist(), up)
            for i in range(num_keyframes):
                hip_raw = np.array(hip_points[i], dtype=float)
                hip = hip_raw + np.array([0.0, 0.0, follow_height_offset], dtype=float)
                should_track = True
                if edge_track and cam_obj is not None:
                    near_edge = target_near_edge(
                        cam_obj,
                        hip_raw.tolist(),
                        cam_fixed.tolist(),
                        current_euler,
                        edge_margin=edge_margin,
                    )
                    moved_enough = np.linalg.norm(hip_raw - hip_prev_raw) >= min_target_shift
                    should_track = near_edge and moved_enough

                if should_track:
                    target_smooth = inertia * target_prev + (1.0 - inertia) * hip
                    cam_euler_curr = euler_from_look_at(cam_fixed.tolist(), target_smooth.tolist(), up)
                else:
                    # Keep pan/tilt fixed while subject remains comfortably inside frame.
                    target_smooth = target_prev
                    cam_euler_curr = current_euler

                cam_pose.append([cam_fixed.tolist(), cam_euler_curr])
                target_prev = target_smooth
                hip_prev_raw = hip_raw
                current_euler = cam_euler_curr
        else:
            # Optional full follow mode: move behind motion direction with inertia.
            cam_prev = np.array(cam_position, dtype=float)
            last_dir = None
            for i in range(num_keyframes):
                hip = np.array(hip_points[i], dtype=float)
                if i < num_keyframes - 1:
                    d = np.array(hip_points[i + 1], dtype=float) - hip
                else:
                    d = hip - np.array(hip_points[i - 1], dtype=float)

                d_xy = np.array([d[0], d[1], 0.0], dtype=float)
                d_norm = np.linalg.norm(d_xy)
                if d_norm > 1e-6:
                    move_dir = d_xy / d_norm
                    last_dir = move_dir
                else:
                    move_dir = last_dir if last_dir is not None else np.array([0.0, 1.0, 0.0], dtype=float)

                desired = hip - move_dir * follow_distance + np.array([0.0, 0.0, follow_height_offset], dtype=float)
                cam_curr = inertia * cam_prev + (1.0 - inertia) * desired
                cam_euler_curr = euler_from_look_at(cam_curr.tolist(), hip.tolist(), up)
                cam_pose.append([cam_curr.tolist(), cam_euler_curr])
                cam_prev = cam_curr

        # Set camera to first pose for debug .blend visibility
        if bpy.context.scene.camera:
            bpy.context.scene.camera.location = cam_pose[0][0]
            bpy.context.scene.camera.rotation_euler = cam_pose[0][1]

        # Add camera-based fill using first camera/target pair
        setup_fill_light(cam_pose[0][0], hip_points[0].tolist())
    else:
        # Compute camera rotation with height-dependent pitch
        # At z_reference, pitch = base_pitch_deg; changes by pitch_scale per unit Z
        z_reference = config.get('camera_pitch_z_reference', 8.0)
        base_pitch_deg = config.get('camera_pitch_base_deg', 90.0)
        pitch_scale = config.get('camera_pitch_scale_per_unit_z', 2.0)
        max_z_lookat_origin = config.get('camera_pitch_max_z_lookat_origin', 25.0)
        if cam_z >= max_z_lookat_origin:
            # Prevent over-tilt at high altitude by using direct look-at to the target.
            cam_euler = euler_from_look_at(cam_position, look_at, up)
        else:
            cam_euler = euler_from_height_dependent_pitch(
                cam_position,
                look_at,
                z_reference=z_reference,
                base_pitch_deg=base_pitch_deg,
                pitch_scale=pitch_scale,
                up=up,
            )

        # Set the actual camera object's location and rotation in the Blender scene
        # so it appears at the correct position when the .blend file is opened/saved
        if bpy.context.scene.camera:
            bpy.context.scene.camera.location = cam_position
            bpy.context.scene.camera.rotation_euler = cam_euler

        # Fixed camera needs multiple identical poses to satisfy interpolation requirements
        cam_pose = [[cam_position, cam_euler]] * num_keyframes

        # Add camera-based fill to brighten foreground near the camera
        setup_fill_light(cam_position, look_at)

    # Human objects stay in place (skeletal animation handles movement)
    # But we need to provide poses for each keyframe for the animation system
    # Each human object gets the same pose repeated for each keyframe
    human_pose = [[0, 0, 0], [0, 0, 0]]  # Position and rotation (already applied to the object)
    human_poses = [[human_pose] * num_keyframes for _ in human_objs]
    
    setup_info = {
        'cam_pose': cam_pose,
        'dynamic_objs': human_objs,
        'dyna_objs_pose': human_poses,
    }
    
    return setup_info

def setup_env(mode):
    cam_pose_list = setup_camera_extrinsic()
    init_pose = cam_pose_list[0]
    ground, static_objs, canopys = setup_placement(init_pose)
    setup_material(ground, static_objs, canopys, mode)
    # setup_lighting(cam_pose_list[0])
    setup_envmap()
    # Camera-based fill to brighten foreground near the camera
    pos, lookat = init_pose
    setup_fill_light(pos, lookat)
    setup_camera_intrinsic()
    pos_lookat_list = [[pos, lookat] for (pos, lookat, euler) in cam_pose_list]
    pos_euler_list = [[pos, euler] for (pos, lookat, euler) in cam_pose_list]
    dynamic_objs, obj_pose_list = setup_dynamic_objs(pos_lookat_list, mode)

    setup_info = {
        'cam_pose': pos_euler_list,
        'dynamic_objs': dynamic_objs,
        'dyna_objs_pose': obj_pose_list,
    }

    return setup_info

def main():
    global config 
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file')
    parser.add_argument('-output_dir')
    parser.add_argument('-mode')
    args = parser.parse_args()

    output_dir = args.output_dir
    config_file = args.config_file
    mode = args.mode

    os.system(f'mkdir -p {output_dir}/hdf5/slow')
    os.system(f'mkdir -p {output_dir}/hdf5/fast')
    os.system(f'mkdir -p {output_dir}/tmp')

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config['activte_range_for_dynamic_obj'] = np.array(config['activte_range_for_dynamic_obj'] )
        except yaml.YAMLError as exc:
            print(exc)

    # Keep output_dir in config so downstream logic can map sequences to model/animation pairs
    config['output_dir'] = output_dir

    bproc.init()

    # Check if we're in human mode
    use_humans = config.get('use_humans', False)
    
    if use_humans:
        setup_info = setup_human_env(mode)
    else:
        setup_info = setup_env(mode)

    # Optionally save .blend file for debugging (before rendering)
    if config.get('save_blend_file', False):
        blend_path = os.path.abspath(f'{output_dir}/debug_scene.blend')
        # Use copy=True to save without changing current file state (prevents material/texture issues)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)
        print(f"Saved .blend file to {blend_path}")

    ########### first pass, render motion blur ########### 
    rgb_fps = config['rgb_image_fps']
    rgb_frames = int(round(rgb_fps * config['duration']))  # exact count
    bpy.context.scene.render.fps = rgb_fps
    animation(output_dir, setup_info, {
        'animation_mode': config['animation_mode'],
        'num_frame': rgb_frames,
        'num_keyframes': config['num_keyframes'],
    })
    bpy.context.scene.render.frame_map_old = 1
    bpy.context.scene.render.frame_map_new = 1
    bpy.context.scene.frame_start = 0
    # Exclusive end so we render exactly rgb_frames frames if renderer iterates range(frame_start, frame_end)
    bpy.context.scene.frame_end = rgb_frames

    # Enable GPU rendering if available and configured
    use_gpu = config.get('use_gpu', True)
    if use_gpu:
        # Enable GPU for Cycles (in case it's used elsewhere)
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
            print(f"Enabled GPU device: {device.name}")
        # Note: Eevee automatically uses GPU when available, no additional setup needed
    else:
        bpy.context.scene.cycles.device = 'CPU'
    
    # exr format which allows linear colorspace
    bproc.renderer.set_output_format("OPEN_EXR", 16)
    
    if config.get('use_cycles', False):
        bpy.context.scene.render.engine = 'CYCLES'
        # Extremely fast Cycles settings
        bpy.context.scene.cycles.samples = 4       
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.cycles.max_bounces = 0
        bpy.context.scene.cycles.diffuse_bounces = 0
        bpy.context.scene.cycles.glossy_bounces = 0
        bpy.context.scene.cycles.transparent_max_bounces = 0
        bpy.context.scene.cycles.transmission_bounces = 0
        bpy.context.scene.cycles.volume_bounces = 0

        # Turn off all expensive features
        bpy.context.scene.cycles.use_caustics = False
        bpy.context.scene.cycles.use_fast_gi = False
    else:
        # Use Eevee renderer (faster than Cycles)
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = 64
        bpy.context.scene.eevee.use_gtao = True  # Ambient occlusion
        bpy.context.scene.eevee.use_ssr = True   # Screen space reflections

    
    # Set Eevee to use GPU
    if use_gpu:
        bpy.context.scene.render.use_simplify = False
        print("GPU rendering enabled for Eevee")
    else:
        bproc.renderer.set_cpu_threads(int(config.get('num_cpu_threads', 8)))

    # TODO: Currently we only use slow fps RGB image for ref video, so close motion blur simulation
    bpy.context.scene.render.use_motion_blur = False
    # bproc.renderer.enable_motion_blur(motion_blur_length=0.0)
    blur_data = bproc.renderer.render(f'{output_dir}/tmp')
    # TODO: tmp, diable hdr exposure time
    blur_img = hdr2ldr(blur_data['colors'], 1)
    data = dict()
    data['blur'] = blur_img
    
    # Save RGB frames as PNG files (motion-blurred, low fps, for reference video)
    if config.get('save_rgb_reference', True):
        os.makedirs(f'{output_dir}/rgb_reference', exist_ok=True)
        for i, frame in enumerate(blur_img):
            cv2.imwrite(f'{output_dir}/rgb_reference/{i:04d}.png', 
                        cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    bproc.renderer.set_output_format("PNG")
    data.update(bproc.renderer.render_optical_flow(f'{output_dir}/tmp', f'{output_dir}/tmp', 
        get_backward_flow=True, get_forward_flow=True, blender_image_coordinate_style=False))

    bproc.writer.write_hdf5(f'{output_dir}/hdf5/rgb_and_flow', data)
    shutil.rmtree(f'{output_dir}/tmp')

    ########### second pass, render clean image for event simulation ###########
    # rgb_reference: Low FPS (rgb_image_fps) motion-blurred frames for reference video & optical flow
    # rgb_event_input: High FPS (event_image_fps) clean HDR frames for event camera simulation 
    event_fps = config['event_image_fps']
    event_frames = int(round(event_fps * config['duration']))  # exact count
    bpy.context.scene.render.fps = event_fps

    animation(output_dir, setup_info, {
        'animation_mode': config['animation_mode'],
        'num_frame': event_frames,
        'num_keyframes': config['num_keyframes'],
    })
    
    # Scale skeletal animation keyframes to match the new frame range
    # This ensures the FBX animation plays through the same motion over event_frames as it did over rgb_frames
    scale_factor = float(event_frames) / float(rgb_frames)
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            for fcu in action.fcurves:
                for kf in fcu.keyframe_points:
                    kf.co.x *= scale_factor
                    kf.handle_left.x *= scale_factor
                    kf.handle_right.x *= scale_factor
                fcu.update()
    
    bpy.context.scene.render.frame_map_old = 1
    bpy.context.scene.render.frame_map_new = 1
    bpy.context.scene.frame_start = 0
    # Use end as exclusive upper bound to produce exactly `target` frames when iterating range(frame_start, frame_end)
    bpy.context.scene.frame_end = event_frames

    bproc.renderer.set_output_format("OPEN_EXR", 16)
    if config.get('use_cycles', False):
        bpy.context.scene.render.engine = 'CYCLES'
        # Extremely fast Cycles settings
        bpy.context.scene.cycles.samples = 4       
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.cycles.max_bounces = 0
        bpy.context.scene.cycles.diffuse_bounces = 0
        bpy.context.scene.cycles.glossy_bounces = 0
        bpy.context.scene.cycles.transparent_max_bounces = 0
        bpy.context.scene.cycles.transmission_bounces = 0
        bpy.context.scene.cycles.volume_bounces = 0

        # Turn off all expensive features
        bpy.context.scene.cycles.use_caustics = False
        bpy.context.scene.cycles.use_fast_gi = False
    else:
        # Use Eevee renderer (faster than Cycles)
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = 64
        bpy.context.scene.eevee.use_gtao = True  # Ambient occlusion
        bpy.context.scene.eevee.use_ssr = True   # Screen space reflections
    
    # GPU already enabled in first pass, just skip CPU thread setting if using GPU
    if not use_gpu:
        bproc.renderer.set_cpu_threads(int(config.get('num_cpu_threads', 8)))

    # bproc.renderer.enable_motion_blur(motion_blur_length=0.0)
    bpy.context.scene.render.use_motion_blur = False
    clean_data = bproc.renderer.render(f'{output_dir}/tmp')
    hdr_img = clean_data['colors']
    data = dict()
    data['hdr'] = hdr_img
    
    # Save HDR frames converted to LDR as PNG files (clean, high fps, for event simulation)
    ldr_img = hdr2ldr(hdr_img, 1)
    if config.get('save_rgb_event_input', True):
        os.makedirs(f'{output_dir}/rgb_event_input', exist_ok=True)
        for i, frame in enumerate(ldr_img):
            cv2.imwrite(f'{output_dir}/rgb_event_input/{i:04d}.png', 
                        cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    bproc.writer.write_hdf5(f'{output_dir}/hdf5/event_input', data)
    shutil.rmtree(f'{output_dir}/tmp')



if __name__ == '__main__':
    main()





