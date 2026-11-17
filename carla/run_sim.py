#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA closed-loop client for "Real-Time Diffusion Relighting as a Neural Sensor".

An ego vehicle follows the Thunderhill raceway (PID throttle; MPC, MPC_DLC, Stanley,
Pure Pursuit or NMPC steering) while a YOLO detector on the front camera reports traffic cones. With --enable-relighting
the front camera is relit by FLUX.2-klein + Delight LoRA (the `relight` package at the
repository root) before detection, either continuously (--async-relighting) or only when
the raw detector's confidence drops (--event-driven-relighting).

Requires a running CARLA 0.9.16 server (see carla/README.md).
"""
from __future__ import print_function
from __future__ import division

# Kill stale copies of this client left over from earlier runs (they keep the CARLA server in
# synchronous mode). Only python processes whose script argument is this file are matched, so
# shells or wrappers that merely mention run_sim.py on their command line are left alone.
import os
import sys


def _kill_stale_clients():
    script = os.path.basename(os.path.realpath(__file__))
    try:
        pids = [p for p in os.listdir('/proc') if p.isdigit() and int(p) != os.getpid()]
    except OSError:
        return
    for pid in pids:
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                argv = [a.decode(errors='ignore') for a in f.read().split(b'\0') if a]
        except OSError:
            continue
        if (len(argv) >= 2 and os.path.basename(argv[0]).startswith('python')
                and any(os.path.basename(a) == script for a in argv[1:3])):
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass


if not {'-h', '--help'} & set(sys.argv[1:]):
    _kill_stale_clients()

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import random
import subprocess
from ultralytics import YOLO
import numpy as np
try:
    import torch
except ImportError:
    torch = None
import csv
import matplotlib.pyplot as plt
# import controller2d
import controller2d_fixed as controller2d

import configparser
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Script level imports. CARLA_DIR holds the track, waypoints and options.cfg; REPO_ROOT holds
# the shared `relight` package and the weights/ folder.
CARLA_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(CARLA_DIR)
sys.path.append(REPO_ROOT)
import live_plotter as lv   # Custom live plotting library
import carla
from datetime import datetime

# CLAHE (Contrast Limited Adaptive Histogram Equalization) imports
try:
    import cv2
    CLAHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenCV (cv2) not available for CLAHE: {e}")
    CLAHE_AVAILABLE = False
    cv2 = None

# Relighting imports (shared `relight` package at the repository root)
try:
    from relight import AsyncRelighter, ConfidenceTrigger, Relighter, load_pipeline
    from relight import BASE_MODEL as DEFAULT_BASE_MODEL, LORA_DIR as DEFAULT_LORA
    RELIGHTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Relighting modules not available: {e}")
    RELIGHTING_AVAILABLE = False
    AsyncRelighter = None
    ConfidenceTrigger = None
    Relighter = None
    load_pipeline = None
    DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-4b"
    DEFAULT_LORA = os.path.join(REPO_ROOT, "weights", "delight_lora")

# For debug window (separate from pygame)
try:
    import tkinter as tk
    from PIL import Image, ImageTk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    Image = None
    print("Warning: tkinter not available. Debug window will not be created.")

# YOLO26 detector fine-tuned on CARLA traffic cones (the paper's detector).
# Loaded once in main() from --yolo-weights (default: <repo>/weights/yolo26_carla.pt).
DEFAULT_YOLO_WEIGHTS = os.path.join(REPO_ROOT, 'weights', 'yolo26_carla.pt')
model = None


######

def _get_waypoints(trajectory):
    waypoints = []
    for index in range(len(trajectory)):
        waypoint = trajectory[index][0]
        waypoints.append([waypoint.location.x, waypoint.location.y, 0])
    return waypoints

def packWaypoints(waypoint_txt,world_input):
    world = world_input
    output_waypoint_list = []
    print(len(waypoint_txt))
    for index in range(len(waypoint_txt)):
        x = waypoint_txt[index][0] 
        y = waypoint_txt[index][1] 
        z = waypoint_txt[index][2] 
        location = carla.Vector3D(x, y, z)
        output_wp= world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
        # output_wp = carla.Waypoint(carla.Transform(carla.Location(x, y, z)))
        # print(x,y,z)
        # print(output_wp)
        # output_wp = sample_wp
        output_waypoint_list.append(output_wp)
    return output_waypoint_list

# In your main loop, process camera frames:
def detect_obstacles(image):
    """
    Detect obstacles in a CARLA image or numpy array using YOLO v8.
    
    Optimization D: Minimized memory copies - only copy when necessary.
    Note: YOLO's plot() method requires contiguous arrays, so we ensure contiguity here.
    
    Args:
        image: CARLA Image object OR numpy array (RGB format, shape HxWx3)
    
    Returns: (annotated_image, detections)
    - annotated_image: numpy array with bounding boxes drawn
    - detections: YOLO results object with boxes, classes, confidences
    """
    if image is None:
        return None, None
    
    # Check if image is already a numpy array (from relighting)
    if isinstance(image, np.ndarray):
        # Optimization D: Use views where possible, but ensure contiguity for YOLO
        if image.shape[2] == 4:  # If BGRA, convert to RGB
            array = image[:, :, [2, 1, 0]]  # View operation
        else:
            array = image  # Already RGB
        # Ensure contiguous for YOLO (required by plot() method)
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
    else:
        # Convert CARLA image to numpy array
        # Optimization D: Use view operations where possible
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # CARLA uses BGRA format (B=0, G=1, R=2, A=3), convert to RGB
        array = array[:, :, [2, 1, 0]]  # View operation
        # Ensure contiguous for YOLO (required by plot() method)
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
    
    # Run YOLO detection
    results = model(array, verbose=False)  # verbose=False to reduce output
    
    # Get annotated image with bounding boxes drawn
    # Note: plot() requires contiguous array, which we ensured above
    annotated_image = results[0].plot()  # This draws bboxes on the image
    
    # Blend the annotated image with the original array to make bounding boxes 50% transparent
    # Where there are no boxes, 0.5*original + 0.5*original = original.
    # Where there are boxes, it becomes 50% box color + 50% original image color.
    annotated_image = cv2.addWeighted(annotated_image, 0.5, array, 0.5, 0)
    
    # Extract detections (bounding boxes, classes, confidence)
    detections = results[0].boxes
    
    return annotated_image, detections

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        image: CARLA Image object OR numpy array (RGB format, shape HxWx3)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
    
    Returns:
        numpy array (RGB format, shape HxWx3) with CLAHE applied
    """
    if not CLAHE_AVAILABLE or cv2 is None:
        print("Warning: OpenCV not available, cannot apply CLAHE")
        return image
    
    # Convert image to numpy array if it's a CARLA Image
    if isinstance(image, np.ndarray):
        # Already a numpy array
        if image.shape[2] == 4:  # If BGRA, convert to RGB
            array = image[:, :, [2, 1, 0]]
        else:
            array = image.copy()  # Make a copy to avoid modifying original
    else:
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # CARLA uses BGRA format, convert to RGB
        array = array[:, :, [2, 1, 0]]
    
    # Ensure contiguous
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    
    # Convert RGB to LAB color space (CLAHE works better on L channel)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    
    # Convert back to RGB
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return rgb_clahe

def build_projection_matrix(w, h, fov):
    """
    Build camera projection matrix for 2D projection (same as collect_carla_dataset.py).
    Based on CARLA official examples: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
    
    Args:
        w: Image width
        h: Image height
        fov: Field of view in degrees
    
    Returns:
        3x3 projection matrix K
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def project_bbox_to_world(bbox, camera, image_width, image_height, estimated_depth=10.0):
    """
    Project a 2D bounding box center-bottom point to 3D world coordinates.
    Uses proper camera projection matrix approach (calibrated with collect_carla_dataset.py method).
    Projects 2D point along camera ray and intersects with ground plane.
    
    Args:
        bbox: YOLO bounding box (x1, y1, x2, y2) in pixel coordinates
        camera: CARLA camera sensor object
        image_width: Image width in pixels
        image_height: Image height in pixels
        estimated_depth: Estimated depth in meters (default: 10m for obstacles)
    
    Returns:
        carla.Location in world coordinates, or None if projection fails
    """
    try:
        # Get bbox center-bottom point (assume obstacle is on ground)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        bottom_y = max(y1, y2)  # Use bottom of bbox (closer to ground)
        
        # Get camera transform
        camera_transform = camera.get_transform()
        camera_location = camera_transform.location
        
        # Get camera FOV (default 90 degrees if not available)
        try:
            fov_str = camera.attributes.get('fov', '90')
            fov = float(fov_str)
        except (AttributeError, ValueError, TypeError):
            fov = 90.0
        
        # Build projection matrix (same as collect_carla_dataset.py)
        K = build_projection_matrix(image_width, image_height, fov)
        
        # Get camera-to-world transformation matrix (inverse of world-to-camera)
        # collect_carla_dataset.py uses: world_2_camera = camera_transform.get_inverse_matrix()
        # For reverse projection, we need camera-to-world = inverse of world-to-camera
        camera_2_world = np.array(camera_transform.get_matrix())
        
        # Reverse the projection: 2D image point -> 3D camera space -> 3D world space
        # Following the same coordinate system as collect_carla_dataset.py:
        # Forward projection: world -> w2c -> UE4 cam [x_c, y_c, z_c] -> convert -> std cam [y_c, -z_c, x_c] -> K -> image [u, v]
        # Reverse projection: image [u, v] -> K^-1 -> std cam [y_c, -z_c, x_c] -> convert back -> UE4 cam [x_c, y_c, z_c] -> c2w -> world
        
        # Step 1: Unproject 2D image point to standard camera coordinates using K^-1
        # Image point in homogeneous coordinates
        point_2d = np.array([center_x, bottom_y, 1.0])
        K_inv = np.linalg.inv(K)
        point_camera_standard = np.dot(K_inv, point_2d)
        
        # Step 2: Convert from standard camera coordinates back to UE4 camera coordinates
        # collect_carla_dataset.py converts: UE4 [x, y, z] -> std [y, -z, x] for forward projection
        # So after K^-1, we have std cam coords: [y_ue4, -z_ue4, x_ue4]
        # Reverse conversion: std [y, -z, x] -> UE4 [x, y, z] means:
        #   x_ue4 = z_std (the third component becomes first)
        #   y_ue4 = x_std (the first component becomes second)
        #   z_ue4 = -y_std (the second component negated becomes third)
        y_std = point_camera_standard[0]
        z_std_neg = point_camera_standard[1]  # This is -z_ue4
        x_std = point_camera_standard[2]
        
        # Convert back to UE4 camera coordinates
        x_ue4 = x_std
        y_ue4 = y_std
        z_ue4 = -z_std_neg  # Negate to get z_ue4 back
        
        # Step 3: Create ray direction in UE4 camera space (normalize first)
        ray_camera_ue4 = np.array([x_ue4, y_ue4, z_ue4])
        ray_length = np.linalg.norm(ray_camera_ue4)
        if ray_length < 1e-6:
            return None  # Invalid direction
        
        # Normalize direction
        ray_camera_ue4 = ray_camera_ue4 / ray_length
        
        # Step 4: Scale by estimated depth to get a point in camera space
        point_camera_ue4 = ray_camera_ue4 * estimated_depth
        
        # Step 5: Transform from camera space to world space
        # Convert to homogeneous coordinates
        point_camera_homogeneous = np.array([point_camera_ue4[0], point_camera_ue4[1], point_camera_ue4[2], 1.0])
        
        # Transform to world coordinates using camera-to-world matrix
        point_world_homogeneous = np.dot(camera_2_world, point_camera_homogeneous)
        
        # Extract world position
        world_location = carla.Location(
            x=float(point_world_homogeneous[0]),
            y=float(point_world_homogeneous[1]),
            z=float(point_world_homogeneous[2])
        )
        
        # Step 6: Project to ground plane (z = camera z - camera height, approximately)
        # For obstacles on road, assume they're at the road level
        # Use a simple heuristic: if camera is at z=1.4m, obstacles are at z=0.0m
        ground_z = camera_location.z - 1.4  # Approximate ground level
        
        # Calculate ray direction in world space for ground intersection
        ray_world = world_location - camera_location
        ray_world_length = math.sqrt(ray_world.x**2 + ray_world.y**2 + ray_world.z**2)
        
        if ray_world_length < 1e-6:
            return None  # Invalid ray
        
        # Normalize ray direction
        ray_world_normalized = carla.Vector3D(
            x=ray_world.x / ray_world_length,
            y=ray_world.y / ray_world_length,
            z=ray_world.z / ray_world_length
        )
        
        # Intersect ray with ground plane
        if abs(ray_world_normalized.z) > 0.001:  # Ray not parallel to ground
            # Calculate intersection parameter t: z = camera_z + t * ray_z = ground_z
            t = (ground_z - camera_location.z) / ray_world_normalized.z
            
            if t > 0:  # Intersection is in front of camera
                world_location = carla.Location(
                    x=camera_location.x + ray_world_normalized.x * t,
                    y=camera_location.y + ray_world_normalized.y * t,
                    z=ground_z
                )
        
        return world_location
    except Exception as e:
        print(f"Error projecting bbox to world: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_obstacle_positions(detections, camera, image_width, image_height):
    """
    Extract 3D world positions of obstacles from YOLO detections.
    
    Args:
        detections: YOLO results boxes object
        camera: CARLA camera sensor object
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        List of carla.Location objects representing obstacle positions
    """
    obstacle_positions = []
    
    if detections is None:
        return obstacle_positions
    
    try:
        # Check if detections has any boxes
        try:
            boxes = detections.xyxy.cpu().numpy()  # Get boxes in (x1, y1, x2, y2) format
            if boxes is None or len(boxes) == 0:
                return obstacle_positions
        except Exception as e:
            # No boxes detected or error accessing boxes
            return obstacle_positions
        
        # Get class IDs (YOLO classes: 0=Vehicle, 1=Bike, 2=Motobike, 3=TrafficSign, 4=TrafficCone)
        # Filter to only TrafficCone (class 4) for obstacle avoidance
        try:
            class_ids = detections.cls.cpu().numpy()
        except:
            class_ids = None
        
        for idx, box in enumerate(boxes):
            # Filter to only TrafficCone (class 4)
            if class_ids is not None and len(class_ids) > idx:
                if int(class_ids[idx]) != 4:  # Not a TrafficCone
                    continue
            
            # Project bbox to world coordinates
            # Use different depth estimates based on bbox size (larger = closer)
            bbox_width = box[2] - box[0]
            bbox_height = box[3] - box[1]
            bbox_area = bbox_width * bbox_height
            
            # Estimate depth: larger bboxes are closer
            # Rough heuristic: assume 0.5m obstacle at 5m distance fills ~10% of image
            # Scale depth inversely with bbox area
            max_area = image_width * image_height
            area_ratio = bbox_area / max_area
            if area_ratio > 0.1:
                estimated_depth = 5.0  # Close obstacle
            elif area_ratio > 0.05:
                estimated_depth = 10.0  # Medium distance
            else:
                estimated_depth = 20.0  # Far obstacle
            
            world_pos = project_bbox_to_world(box, camera, image_width, image_height, estimated_depth)
            if world_pos is not None:
                obstacle_positions.append(world_pos)
    except Exception as e:
        print(f"Error extracting obstacle positions: {e}")
        import traceback
        traceback.print_exc()
    
    return obstacle_positions

# distance 5 meters from the obstacle

def modify_waypoints_for_obstacles(waypoints, obstacle_positions, vehicle_location, avoidance_distance=8.0):
    """
    Modify waypoints to avoid obstacles by shifting them laterally.
    
    Args:
        waypoints: List of waypoints in format [[x, y, v], [x, y, v], ...]
        obstacle_positions: List of carla.Location objects representing obstacles
        vehicle_location: Current vehicle location (carla.Location)
        avoidance_distance: Minimum distance to maintain from obstacles (meters)
    
    Returns:
        Modified waypoints list (same format as input)
    """
    if len(obstacle_positions) == 0:
        return waypoints
    
    # Convert waypoints to list of lists if needed
    modified_waypoints = []
    for wp in waypoints:
        if isinstance(wp, (list, np.ndarray)):
            modified_waypoints.append([float(wp[0]), float(wp[1]), float(wp[2]) if len(wp) > 2 else 0.0])
        else:
            modified_waypoints.append([float(wp), 0.0, 0.0])
    
    # Filter obstacles that are ahead of the vehicle
    vehicle_x = vehicle_location.x
    vehicle_y = vehicle_location.y
    
    # Process each waypoint
    for i in range(len(modified_waypoints)):
        wp_x = modified_waypoints[i][0]
        wp_y = modified_waypoints[i][1]
        
        # Check distance to each obstacle
        for obs_pos in obstacle_positions:
            obs_x = obs_pos.x
            obs_y = obs_pos.y
            
            # Calculate distance from waypoint to obstacle
            dist_to_obstacle = math.sqrt((wp_x - obs_x)**2 + (wp_y - obs_y)**2)
            
            # If waypoint is too close to obstacle, shift it laterally
            if dist_to_obstacle < avoidance_distance:
                # Calculate direction from obstacle to waypoint
                dx = wp_x - obs_x
                dy = wp_y - obs_y
                
                if abs(dx) < 0.01 and abs(dy) < 0.01:
                    # Waypoint is exactly at obstacle, shift perpendicular to path
                    # Use a default shift direction (to the right)
                    dx = 1.0
                    dy = 0.0
                
                # Normalize direction vector
                norm = math.sqrt(dx**2 + dy**2)
                if norm > 0.01:
                    dx /= norm
                    dy /= norm
                
                # Shift waypoint away from obstacle
                # Shift by (avoidance_distance - current_distance) to maintain safe distance
                shift_distance = avoidance_distance - dist_to_obstacle + 1.0  # Add 1m buffer
                modified_waypoints[i][0] = wp_x + dx * shift_distance
                modified_waypoints[i][1] = wp_y + dy * shift_distance
    
    return modified_waypoints
    
######

"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 5.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 400.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
# Simulation timeout: if lap not completed within this time, mark as failed and exit
SIMULATION_TIMEOUT     = None   # Will be set from command-line args or use TOTAL_RUN_TIME
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 0      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

# WAYPOINTS_FILENAME = 'T01_waypoint_all.txt'  # constant speed 60kph

# WAYPOINTS_FILENAME = 'all_waypoints_3.txt'  # constant speed to load
# WAYPOINTS_FILENAME = 'T01_waypoint_all_2.txt'  # constant speed path
WAYPOINTS_FILENAME = os.path.join(CARLA_DIR, 'stanford_waypoints_full.txt')  # optimized middle line

# WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'  # I don't know what but a curley one

DIST_THRESHOLD_TO_LAST_WAYPOINT = 10.0  # some distance from last position before
                                       # simulation ends

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory (override with --output-dir)
CONTROLLER_OUTPUT_FOLDER = os.path.join(CARLA_DIR, 'results', 'controller_output') + '/'

# Global collision counter updated by a collision sensor callback, and
# a handle to the collision sensor so it is not garbage-collected.
collision_count = 0
collision_sensor = None

# Global tracking for unique traffic cone collisions
# Track which cone IDs have been hit (each cone counted only once)
hit_cone_ids = set()  # Set of unique cone actor IDs that have been contacted
total_cone_count = 0  # Total number of traffic cones spawned

# Global store for latest camera frames from the multi-camera rig.
# Keys are camera names (e.g. "front").
latest_camera_frames = {}

# Keep strong references to camera sensors so their callbacks stay active.
camera_sensors = []

# Global camera references for obstacle detection (front camera)
cam_front_ref = None

# Global storage for front camera YOLO detections
front_camera_detections = None
front_camera_image = None
# Track the image size actually used for YOLO (may be relit & downscaled)
front_yolo_img_width = None
front_yolo_img_height = None

# Global pygame surface for visualizing multi-camera views.
camera_display = None

# Global relighting service and flags.
# relighting_service is a relight.Relighter (synchronous / event-driven modes) or a
# relight.AsyncRelighter (--async-relighting); relight_trigger is the event-driven
# relight.ConfidenceTrigger (hysteresis on the raw detector's c_max).
relighting_service = None
relight_trigger = None
# Optimization C: Removed Tkinter - relit view now in Pygame window
ENABLE_RELIGHTING = False  # Manual toggle flag (set via command-line argument)
ENABLE_CLAHE = False  # Manual toggle flag for CLAHE enhancement (set via command-line argument)
ENABLE_EVENT_DRIVEN = False  # Event-driven relighting mode (confidence-triggered)
TAU_LOW = 0.3  # Lower threshold for triggering relighting
TAU_HIGH = 0.5  # Upper threshold for disabling relighting
is_relighting_active = False  # State variable for event-driven hysteresis (persists across frames)
first_relit_received = False  # Flag to track if we've received the first relit image
last_relit_image = None  # Store last relit PIL image for frame skipping
last_relit_np = None  # Store last relit numpy array for YOLO when skipping frames
last_relit_image = None  # Store last relit image for frame skipping
last_relit_np = None  # Store last relit numpy array for YOLO

# Performance profiling class
class PerformanceProfiler:
    """Tracks timing for different stages of the simulation loop."""
    def __init__(self, enabled=True, report_interval=100):
        self.enabled = enabled
        self.report_interval = report_interval
        self.frame_count = 0
        self.timings = {
            'total_frame': [],
            'carla_tick': [],
            'vehicle_state': [],
            'camera_retrieval': [],
            'relighting': [],
            'clahe': [],  # Track CLAHE processing time
            'yolo_detection': [],
            'image_conversion': [],
            'pygame_render': [],
            'tkinter_update': [],
            'controller_update': [],
            'live_plotting': [],
            'control_command': [],
            'async_relighting': [],  # Track async relighting processing time
            'frame_overhead': [],  # Time not accounted for in other stages
            'waypoint_finding': [],  # Time to find closest waypoint
            'other_operations': [],  # Other untimed operations (history updates, debug drawing, etc.)
        }
        self.current_frame_start = None
        self.stage_times = {}  # Track individual stage times per frame
    
    def start_frame(self):
        """Mark the start of a new frame."""
        if not self.enabled:
            return
        self.current_frame_start = time.time()
        self.frame_count += 1
    
    def end_frame(self):
        """Mark the end of the current frame."""
        if not self.enabled or self.current_frame_start is None:
            return
        total_time = time.time() - self.current_frame_start
        
        # Calculate overhead (time not accounted for in other stages)
        accounted_time = sum(self.stage_times.values()) if self.stage_times else 0.0
        overhead = max(0.0, total_time - accounted_time)
        self.timings['frame_overhead'].append(overhead)
        
        self.timings['total_frame'].append(total_time)
        self.current_frame_start = None
        self.stage_times = {}  # Reset for next frame
    
    def time_stage(self, stage_name, func, *args, **kwargs):
        """Time a specific stage and execute the function."""
        if not self.enabled:
            return func(*args, **kwargs)
        
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        if stage_name in self.timings:
            self.timings[stage_name].append(elapsed)
        
        return result
    
    def record_timing(self, stage_name, elapsed_time, is_async=False):
        """Record a timing manually.
        
        Args:
            stage_name: Name of the stage
            elapsed_time: Time elapsed in seconds
            is_async: If True, this time is from async operations and shouldn't count toward overhead calculation
        """
        if not self.enabled:
            return
        if stage_name in self.timings:
            self.timings[stage_name].append(elapsed_time)
            # Track for overhead calculation (only count blocking operations)
            if not is_async:
                self.stage_times[stage_name] = self.stage_times.get(stage_name, 0.0) + elapsed_time
    
    def print_stats(self):
        """Print timing statistics."""
        if not self.enabled or self.frame_count % self.report_interval != 0:
            return
        
        print("\n" + "=" * 80)
        print(f"PERFORMANCE PROFILING REPORT (Frame {self.frame_count})")
        print("=" * 80)
        
        # Calculate statistics for each stage
        stats = {}
        for stage_name, timings in self.timings.items():
            if len(timings) > 0:
                stats[stage_name] = {
                    'mean': np.mean(timings),
                    'median': np.median(timings),
                    'max': np.max(timings),
                    'min': np.min(timings),
                    'std': np.std(timings),
                    'count': len(timings)
                }
        
        # Print in order of importance
        stage_order = [
            'total_frame',
            'carla_tick',
            'relighting',
            'async_relighting',
            'yolo_detection',
            'camera_retrieval',
            'image_conversion',
            'pygame_render',
            'tkinter_update',
            'controller_update',
            'live_plotting',
            'vehicle_state',
            'control_command',
            'waypoint_finding',
            'other_operations',
            'frame_overhead',
        ]
        
        for stage in stage_order:
            if stage in stats:
                s = stats[stage]
                # Calculate percentage of total frame time
                pct = (s['mean'] / stats['total_frame']['mean'] * 100) if 'total_frame' in stats else 0
                print(f"{stage:20s}: "
                      f"mean={s['mean']*1000:6.2f}ms ({pct:5.1f}%), "
                      f"median={s['median']*1000:6.2f}ms, "
                      f"max={s['max']*1000:6.2f}ms, "
                      f"min={s['min']*1000:6.2f}ms, "
                      f"std={s['std']*1000:6.2f}ms "
                      f"({s['count']} samples)")
        
        # Calculate total overhead
        # Exclude async operations (they don't block the main thread)
        async_stages = {'async_relighting', 'frame_overhead'}
        if 'total_frame' in stats:
            total_mean = stats['total_frame']['mean']
            # Sum only blocking operations (exclude async and overhead itself)
            blocking_sum = sum(s['mean'] for k, s in stats.items() 
                             if k != 'total_frame' and k not in async_stages)
            overhead = total_mean - blocking_sum
            overhead_pct = (overhead / total_mean * 100) if total_mean > 0 else 0
            print(f"\n{'overhead':20s}: {overhead*1000:6.2f}ms ({overhead_pct:5.1f}%) (unaccounted time)")
            print(f"{'fps_estimate':20s}: {1.0/total_mean:.2f} FPS")
            
            # Show async relighting separately (informational, not blocking)
            if 'async_relighting' in stats:
                async_mean = stats['async_relighting']['mean']
                print(f"{'async_relighting_info':20s}: {async_mean*1000:6.2f}ms (background processing, non-blocking)")
            
            # Identify bottleneck
            print(f"\n{'BOTTLENECK ANALYSIS':20s}:")
            if 'relighting' in stats:
                relight_mean = stats['relighting']['mean']
                relight_pct = (relight_mean / total_mean * 100) if total_mean > 0 else 0
                if relight_pct > 50:
                    print(f"  ⚠️  RELIGHTING is the major bottleneck: {relight_pct:.1f}% of frame time")
                    print(f"      Consider: async processing, reducing image size, or skipping frames")
                elif relight_pct > 25:
                    print(f"  ⚠️  RELIGHTING is significant: {relight_pct:.1f}% of frame time")
            
            # Find highest variance stages
            high_variance = [(k, s['std']/s['mean'] if s['mean'] > 0 else 0) 
                            for k, s in stats.items() 
                            if k != 'total_frame' and s['mean'] > 0.001]
            if high_variance:
                high_variance.sort(key=lambda x: x[1], reverse=True)
                top_var = high_variance[0]
                if top_var[1] > 1.0:  # std > mean (very high variance)
                    print(f"  ⚠️  HIGH VARIANCE in '{top_var[0]}': CV={top_var[1]:.2f} (inconsistent performance)")
        
        print("=" * 80 + "\n")
        
        # Clear old timings to prevent memory growth (keep last N samples)
        max_samples = 1000
        for stage_name in self.timings:
            if len(self.timings[stage_name]) > max_samples:
                self.timings[stage_name] = self.timings[stage_name][-max_samples:]


def carla_image_to_surface(image):
    """
    Convert a CARLA raw BGRA image to a pygame.Surface for visualization.
    
    Optimization D: Minimized memory copies - use views where possible.
    """
    if image is None:
        return None

    # Optimization D: Use view operations instead of copies
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # Convert BGRA to RGB using view (no copy)
    rgb_array = array[:, :, [2, 1, 0]]  # View operation
    # pygame expects (width, height, 3) - swapaxes creates a view
    surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
    return surface

def carla_image_to_rgb_array(image):
    """
    Convert a CARLA BGRA image (or numpy array) into an RGB numpy array (HxWx3, uint8).
    This is used for saving screenshots without requiring a pygame window.
    """
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        # If already numpy, normalize to RGB HxWx3
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, [2, 1, 0]]
        return image
    # CARLA Image: raw BGRA
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_array = array[:, :, [2, 1, 0]]
    return rgb_array

def relight_carla_image(relighter, image, num_inference_steps=1):
    """
    Relight a CARLA BGRA image (or an RGB numpy array) with the shared `relight` package.
    Returns a PIL image at the model resolution (long side 1024 / --relight-size-factor).
    """
    if image is None:
        return None
    return relighter(carla_image_to_rgb_array(image), num_inference_steps=num_inference_steps)

def _normalize_progress_list(values):
    """
    Normalize a list of progress specs into floats in [0, 1].
    Accepts either fractions (0.1) or percents (10 -> 0.10).
    """
    normalized = []
    if not values:
        return normalized
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if x > 1.0:
            x = x / 100.0
        # Clamp
        x = max(0.0, min(1.0, x))
        normalized.append(x)
    # Unique + sorted for deterministic ordering
    normalized = sorted(set(normalized))
    return normalized

def _capture_simulator_window_png(output_path, window_name="CarlaUE4", cached_window_id=None):
    """
    Best-effort capture of the CARLA simulator (CarlaUE4) OS window into a PNG.

    This requires a running X11 session (DISPLAY set) and at least one of:
    - xdotool + imagemagick 'import'
    - xwd + imagemagick 'convert'
    - gnome-screenshot
    - scrot

    Returns: (success: bool, window_id: str|None)
    """
    try:
        display = os.environ.get("DISPLAY", "")
        if not display:
            return False, cached_window_id

        # Find window id if not cached
        window_id = cached_window_id
        if not window_id:
            # Prefer xdotool search by window name
            try:
                r = subprocess.run(
                    ["xdotool", "search", "--name", window_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if r.returncode == 0 and r.stdout.strip():
                    # Take the first match
                    window_id = r.stdout.strip().splitlines()[0].strip()
            except FileNotFoundError:
                window_id = None

        # Attempt capture with ImageMagick import (best quality)
        if window_id:
            try:
                r = subprocess.run(
                    ["import", "-window", window_id, output_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if r.returncode == 0 and os.path.exists(output_path):
                    return True, window_id
            except FileNotFoundError:
                pass

            # Fallback: xwd -> convert
            try:
                xwd = subprocess.Popen(
                    ["xwd", "-silent", "-id", window_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                )
                conv = subprocess.run(
                    ["convert", "xwd:-", output_path],
                    stdin=xwd.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if conv.returncode == 0 and os.path.exists(output_path):
                    return True, window_id
            except FileNotFoundError:
                pass
            except Exception:
                pass

        # Fallback: gnome-screenshot active window (may capture wrong window if focus is not CarlaUE4)
        try:
            r = subprocess.run(
                ["gnome-screenshot", "-w", "-f", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if r.returncode == 0 and os.path.exists(output_path):
                return True, window_id
        except FileNotFoundError:
            pass

        # Fallback: scrot focused window
        try:
            r = subprocess.run(
                ["scrot", "-u", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if r.returncode == 0 and os.path.exists(output_path):
                return True, window_id
        except FileNotFoundError:
            pass

        return False, window_id
    except Exception:
        return False, cached_window_id

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()

    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings

class Timer(object):
    """ Timer Class

    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements

    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene

    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)
    print(x,y,yaw)
    return (x, y, yaw)

def send_control_command(vehicle,throttle, steer, brake,
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.

    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """    
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)
    control_para = carla.VehicleControl(throttle, steer, brake)
    control_para.hand_brake = hand_brake
    control_para.reverse = reverse
    vehicle.apply_control(control_para)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i]))

def calculate_path_tracking_error(x_history, y_history, waypoints_ref):
    """
    Calculate path tracking error (lateral deviation) from reference trajectory.
    
    For each point in the actual trajectory, finds the closest point on the reference
    trajectory and calculates the perpendicular distance (lateral deviation).
    
    Args:
        x_history: List of actual x positions
        y_history: List of actual y positions
        waypoints_ref: Reference waypoints as numpy array (N x 2) with [x, y] columns
    
    Returns:
        dict with keys: 'mean', 'std', 'max', 'min', 'all_errors' (list of all errors)
    """
    if len(x_history) == 0 or waypoints_ref.shape[0] < 2:
        return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0, 'all_errors': []}
    
    # Convert trajectory to numpy array
    actual_trajectory = np.array([[x_history[i], y_history[i]] for i in range(len(x_history))])
    
    lateral_deviations = []
    
    # For each point in actual trajectory, find closest point on reference trajectory
    for i in range(len(actual_trajectory)):
        actual_point = actual_trajectory[i]
        
        # Find closest waypoint (simple point-to-point distance)
        distances_to_waypoints = np.linalg.norm(waypoints_ref - actual_point, axis=1)
        closest_wp_idx = np.argmin(distances_to_waypoints)
        
        # Calculate perpendicular distance to line segments
        # Check both adjacent segments for more accurate lateral deviation
        min_lateral_dist = float('inf')
        
        # Check segment before closest waypoint
        if closest_wp_idx > 0:
            wp1 = waypoints_ref[closest_wp_idx - 1]
            wp2 = waypoints_ref[closest_wp_idx]
            lateral_dist = point_to_line_segment_distance(actual_point, wp1, wp2)
            min_lateral_dist = min(min_lateral_dist, lateral_dist)
        
        # Check segment after closest waypoint
        if closest_wp_idx < waypoints_ref.shape[0] - 1:
            wp1 = waypoints_ref[closest_wp_idx]
            wp2 = waypoints_ref[closest_wp_idx + 1]
            lateral_dist = point_to_line_segment_distance(actual_point, wp1, wp2)
            min_lateral_dist = min(min_lateral_dist, lateral_dist)
        
        # If at start or end, use point-to-point distance
        if min_lateral_dist == float('inf'):
            min_lateral_dist = distances_to_waypoints[closest_wp_idx]
        
        lateral_deviations.append(min_lateral_dist)
    
    lateral_deviations = np.array(lateral_deviations)
    
    return {
        'mean': np.mean(lateral_deviations),
        'std': np.std(lateral_deviations),
        'max': np.max(lateral_deviations),
        'min': np.min(lateral_deviations),
        'all_errors': lateral_deviations.tolist()
    }

def point_to_line_segment_distance(point, line_start, line_end):
    """
    Calculate perpendicular distance from a point to a line segment.
    
    Args:
        point: Point [x, y]
        line_start: Start of line segment [x, y]
        line_end: End of line segment [x, y]
    
    Returns:
        Perpendicular distance (lateral deviation) in meters
    """
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    
    # If line segment has zero length, return distance to start point
    if line_len_sq < 1e-10:
        return np.linalg.norm(point - line_start)
    
    # Vector from line_start to point
    point_vec = point - line_start
    
    # Project point onto line segment
    t = np.dot(point_vec, line_vec) / line_len_sq
    
    # Clamp t to [0, 1] to stay on line segment
    t = max(0.0, min(1.0, t))
    
    # Closest point on line segment
    closest_point = line_start + t * line_vec
    
    # Perpendicular distance (lateral deviation)
    lateral_dist = np.linalg.norm(point - closest_point)
    
    return lateral_dist

def calculate_speed_tracking_error(speed_history, desired_speed_history):
    """
    Calculate speed tracking error (deviation from desired/reference speed).
    
    Args:
        speed_history: List of actual speeds (m/s)
        desired_speed_history: List of desired/reference speeds (m/s)
    
    Returns:
        dict with keys: 'mean', 'std', 'max', 'min', 'rmse', 'all_errors' (list of all errors)
    """
    if len(speed_history) == 0 or len(desired_speed_history) == 0:
        return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0, 'rmse': 0.0, 'all_errors': []}
    
    if len(speed_history) != len(desired_speed_history):
        # Trim to shorter length if mismatch
        min_len = min(len(speed_history), len(desired_speed_history))
        speed_history = speed_history[:min_len]
        desired_speed_history = desired_speed_history[:min_len]
    
    # Calculate speed errors (absolute difference)
    speed_errors = np.abs(np.array(speed_history) - np.array(desired_speed_history))
    
    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(speed_errors**2))
    
    return {
        'mean': np.mean(speed_errors),
        'std': np.std(speed_errors),
        'max': np.max(speed_errors),
        'min': np.min(speed_errors),
        'rmse': rmse,
        'all_errors': speed_errors.tolist()
    }

def get_weather_parameters(weather_preset='clear', precipitation=None, fog_density=None, 
                           fog_distance=None, cloudiness=None, sun_angle=None):
    """
    Get CARLA weather parameters based on preset and optional custom values.
    
    Args:
        weather_preset: Weather preset name ('clear', 'rainy', 'heavy_rain', 'foggy', 'foggy_rainy', 'cloudy')
        precipitation: Custom precipitation (0-100), overrides preset if set
        fog_density: Custom fog density (0-100), overrides preset if set
        fog_distance: Custom fog distance in meters, overrides preset if set
        cloudiness: Custom cloudiness (0-100), overrides preset if set
        sun_angle: Sun altitude angle in degrees
    
    Returns:
        carla.WeatherParameters object
    """
    # Weather presets
    weather_presets = {
        'clear': {
            'cloudiness': 0.0,
            'precipitation': 0.0,
            'precipitation_deposits': 0.0,
            'fog_density': 0.0,
            'fog_distance': 0.0,
            'wetness': 0.0,
            'wind_intensity': 0.0,
        },
        'rainy': {
            'cloudiness': 80.0,
            'precipitation': 50.0,
            'precipitation_deposits': 50.0,
            'fog_density': 0.0,
            'fog_distance': 0.0,
            'wetness': 50.0,
            'wind_intensity': 20.0,
        },
        'heavy_rain': {
            'cloudiness': 100.0,
            'precipitation': 90.0,
            'precipitation_deposits': 100.0,
            'fog_density': 0.0,
            'fog_distance': 0.0,
            'wetness': 100.0,
            'wind_intensity': 50.0,
        },
        'foggy': {
            'cloudiness': 60.0,
            'precipitation': 0.0,
            'precipitation_deposits': 0.0,
            'fog_density': 50.0,
            'fog_distance': 10.0,  # 10 meters visibility
            'wetness': 0.0,
            'wind_intensity': 0.0,
        },
        'foggy_rainy': {
            'cloudiness': 90.0,
            'precipitation': 60.0,
            'precipitation_deposits': 70.0,
            'fog_density': 40.0,
            'fog_distance': 15.0,  # 15 meters visibility
            'wetness': 70.0,
            'wind_intensity': 30.0,
        },
        'cloudy': {
            'cloudiness': 70.0,
            'precipitation': 0.0,
            'precipitation_deposits': 0.0,
            'fog_density': 0.0,
            'fog_distance': 0.0,
            'wetness': 0.0,
            'wind_intensity': 10.0,
        },
    }
    
    # Get preset values
    preset = weather_presets.get(weather_preset, weather_presets['clear'])
    
    # Override with custom values if provided
    final_cloudiness = cloudiness if cloudiness is not None else preset['cloudiness']
    final_precipitation = precipitation if precipitation is not None else preset['precipitation']
    final_fog_density = fog_density if fog_density is not None else preset['fog_density']
    final_fog_distance = fog_distance if fog_distance is not None else preset['fog_distance']
    
    # Clamp values to valid ranges
    final_cloudiness = max(0.0, min(100.0, final_cloudiness))
    final_precipitation = max(0.0, min(100.0, final_precipitation))
    final_fog_density = max(0.0, min(100.0, final_fog_density))
    final_fog_distance = max(0.0, final_fog_distance)
    
    # Create weather parameters
    # weather = carla.WeatherParameters(
    #     cloudiness=final_cloudiness,
    #     precipitation=final_precipitation,
    #     precipitation_deposits=preset['precipitation_deposits'] if precipitation is None else min(100.0, final_precipitation * 1.1),
    #     fog_density=final_fog_density,
    #     fog_distance=final_fog_distance,
    #     wetness=preset['wetness'] if precipitation is None else min(100.0, final_precipitation),
    #     wind_intensity=preset['wind_intensity'],
    #     sun_altitude_angle=sun_angle if sun_angle is not None else 90.0,
    #     scattering_intensity=10.0 if sun_angle <0 else 0.0,
    # )

    weather = carla.WeatherParameters(
    cloudiness=15.0,              # Slight clouds catch the ambient light better than a clear sky
    precipitation=final_precipitation,
    precipitation_deposits=preset['precipitation_deposits'] if precipitation is None else min(100.0, final_precipitation * 1.1),
    fog_density=min(2.0, final_fog_density),# DO NOT set to 0. Fog is required for "volumetric" night light
    fog_distance=min(0.75, final_fog_distance),            # Brings the "glow" closer to the car
    wetness=preset['wetness'] if precipitation is None else min(100.0, final_precipitation),
    wind_intensity=preset['wind_intensity'],
    
    # REALISM TWEAK: 
    # Use -25.0 for a "bright" moonlit night (Blue hour)
    # Use -40.0 or lower for deep night
    sun_altitude_angle=sun_angle if sun_angle is not None else 90.0,
    
    # YOUR HACK (Optimized): 
    # 10.0 is quite high; 5.0 often looks more "moonlit" and less "radioactive"
    scattering_intensity=5.0 if sun_angle < 0 else 0.0,
)
    return weather

def make_carla_client(host, port, use_sync_mode=True, fixed_delta=0.05, weather_args=None, time_of_day=None):
    global collision_count, collision_sensor, latest_camera_frames, camera_sensors

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    # Read the opendrive file to a string

    # xodr_path = "speedway_5lanes.xodr"
    xodr_path = os.path.join(CARLA_DIR, "Thunderhilll.xodr")
    od_file = open(xodr_path)
    data = od_file.read()

    # # Load the opendrive map
    vertex_distance = 2.0  # in meters
    max_road_length = 50.0 # in meters  10
    wall_height = 0.5     # in meters  1.0
    extra_width = 0.6      # in meters
    global waypoint_grid
    waypoint_grid = 5
    # global waypoint_search_grid
    # waypoint_search_grid = 10
    throttle_value = 0.9
    initialized_point = 0
    constant_speed = 10 # m/s
    global section
    section = 500
    world = client.generate_opendrive_world(
        data, carla.OpendriveGenerationParameters(
        vertex_distance=vertex_distance,
        max_road_length=max_road_length,
        wall_height=wall_height,
        additional_width=extra_width,
        smooth_junctions=True,
        enable_mesh_visibility=True))
    # world = client.load_world('Town02')

    # Retrieve a snapshot of the world at this point in time.
    world_snapshot = world.get_snapshot()

    # Wait for the next tick and retrieve the snapshot of the tick.
    world_snapshot = world.wait_for_tick()

    # Register a callback to get called every time we receive a new snapshot.
    world.on_tick(lambda world_snapshot: world_snapshot)

    # Automatic change sun angle
    # Get time of day from parameter or use default
    if time_of_day is not None:
        current_hours = time_of_day
    else:
        now = datetime.now()
        current_hours = int(now.strftime('%H'))
        current_hours = 10  # Default to day
    # current_hours = 18

    # Draw the plot below, you'll understand.
    # time = [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23.]
    # angle = [-90. -75. -60. -45. -30. -15.   0.  15.  30.  45.  60.  75.  90.  75. 60.  45.  30.  15.  0. -15. -30. -45. -60. -75.]
    # plt.plot(time,angle)
    sun_angle =  current_hours*(15)-90 if (current_hours <= 12 and current_hours >= 0) else current_hours*(-15)+270

    # Set weather based on command-line arguments or defaults
    if weather_args is not None:
        weather = get_weather_parameters(
            weather_preset=weather_args.get('weather_preset', 'clear'),
            precipitation=weather_args.get('precipitation'),
            fog_density=weather_args.get('fog_density'),
            fog_distance=weather_args.get('fog_distance'),
            cloudiness=weather_args.get('cloudiness'),
            sun_angle=sun_angle
        )
        print(f"Weather preset: {weather_args.get('weather_preset', 'clear')}")
        if weather_args.get('precipitation') is not None:
            print(f"  Custom precipitation: {weather_args.get('precipitation')}")
        if weather_args.get('fog_density') is not None:
            print(f"  Custom fog density: {weather_args.get('fog_density')}")
        if weather_args.get('fog_distance') is not None:
            print(f"  Custom fog distance: {weather_args.get('fog_distance')}m")
        if weather_args.get('cloudiness') is not None:
            print(f"  Custom cloudiness: {weather_args.get('cloudiness')}")
    else:
        # Default: clear weather
        weather = get_weather_parameters(weather_preset='clear', sun_angle=sun_angle)
    
    world.set_weather(weather)
    print(f"Weather set: cloudiness={weather.cloudiness:.1f}, precipitation={weather.precipitation:.1f}, "
          f"fog_density={weather.fog_density:.1f}, fog_distance={weather.fog_distance:.1f}m")

    # Optionally enable synchronous mode with a fixed time step so that physics,
    # sensors, and visualization stay in lockstep. This helps avoid
    # flickering / jitter due to async updates.
    settings = world.get_settings()
    if use_sync_mode:
        settings.synchronous_mode = True
        
        # Validate fixed_delta_seconds: CARLA requires fixed_delta_seconds <= max_substep_delta_time * max_substeps
        # Default CARLA values: max_substep_delta_time = 0.01, max_substeps = 10, so max = 0.1 seconds
        # Warn if value is too large (will cause simulation instability)
        if fixed_delta > 0.1:
            print(f"WARNING: fixed_delta_seconds={fixed_delta} is too large!")
            print(f"  CARLA requires: fixed_delta_seconds <= max_substep_delta_time * max_substeps (typically <= 0.1)")
            print(f"  This may cause simulation instability or early termination.")
            print(f"  Recommended values: 0.05 (20 FPS), 0.02 (50 FPS), or 0.01 (100 FPS)")
            print(f"  Proceeding anyway, but simulation may not work correctly...")
        
        settings.fixed_delta_seconds = fixed_delta  # e.g. 0.05 -> 20 FPS simulation
    else:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
    
    # CARLA GPU Optimizations (from https://carla.readthedocs.io/en/latest/adv_rendering_options/)
    # Note: Quality level and no-rendering mode should be set via command-line when starting CARLA server
    # These settings can be applied at runtime but may have limited effect
    # For best performance: start CARLA with -quality-level=Low or -RenderOffScreen
    # No-rendering mode can be enabled here but will disable all cameras
    # settings.no_rendering_mode = False  # Set via command-line argument if needed
    
    world.apply_settings(settings)
    spectator = world.get_spectator()
    map = world.get_map()
    print('make_carla_client----Carla client connected.')

    transform = carla.Transform(carla.Location(x=18.30674744, y=191.2394257, z=0), carla.Rotation(pitch=0,yaw=0,roll=0))
    start_waypoint = map.get_waypoint(transform.location)

    # #Set spawning location as initial waypoint
    waypoint = start_waypoint
    blueprint = world.get_blueprint_library().filter('vehicle.*tt*')[0]
    blueprint.set_attribute('color', '255,255,246')
    location = waypoint.transform.location + carla.Vector3D(0, 0, 1.5)
    rotation = carla.Rotation(pitch=0,yaw=90,roll=0)

    vehicle = world.spawn_actor(blueprint, carla.Transform(location, rotation))
    print("make_carla_client----SPAWNED!")
    vehicle.set_simulate_physics(True)
    # vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.All | carla.VehicleLightState.LowBeam | carla.VehicleLightState.HighBeam | carla.VehicleLightState.Interior))
    vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LowBeam))

    # Add spectator camera (third-person chase view) to get the view to move with the car
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # Match the camera FPS to the simulation fixed delta (or let it run free
    # in async mode by using 0.0)
    sensor_tick = settings.fixed_delta_seconds if use_sync_mode else 0.0
    camera_bp.set_attribute('sensor_tick', str(sensor_tick))
    # camera_transform = carla.Transform(carla.Location(x=-10,z=10), carla.Rotation(-20,0,0))
    camera_transform = carla.Transform(carla.Location(x=-10,z=7), carla.Rotation(-20,0,0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    spectator.set_transform(camera.get_transform())
    spectator.set_transform(camera.get_transform())

    #############################################
    # Vehicle multi-camera rig (Waymo-style)
    #############################################
    # First-person / front view plus front-left and front-right cameras.
    # These will later feed a computer vision module via latest_camera_frames.
    latest_camera_frames = {}
    camera_sensors = []

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('sensor_tick', str(sensor_tick))
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')

    def _make_cam_callback(name):
        """Create a non-blocking camera callback that stores the latest frame.
        
        Optimization A: Non-blocking callback that immediately stores the frame
        without any processing to prevent lag accumulation.
        """
        def _callback(image):
            # Non-blocking update: just store the frame immediately
            # This prevents blocking the camera sensor thread
            latest_camera_frames[name] = image
        return _callback

    # Front (ego first-person style) camera: slightly above and ahead of the CG
    front_transform = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=1.4),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )
    cam_front = world.spawn_actor(cam_bp, front_transform, attach_to=vehicle)
    cam_front.listen(_make_cam_callback("front"))
    camera_sensors.append(cam_front)
    
    # Store front camera reference globally for obstacle detection
    global cam_front_ref
    cam_front_ref = cam_front

    # Attach a collision sensor to the ego vehicle to count collisions.
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
        attach_to=vehicle,
    )

    def _on_collision(event):
        """
        Collision callback that tracks unique traffic cone collisions.
        Each cone is counted only once, even if contact is continuous.
        """
        global collision_count, hit_cone_ids
        
        # Check if the other actor is a traffic cone
        other_actor = event.other_actor
        if other_actor is None:
            return
        
        # Check if it's a traffic cone by type_id
        type_id = other_actor.type_id.lower()
        if 'trafficcone' in type_id or 'static.prop.trafficcone' in type_id:
            # Get the unique ID of the cone
            cone_id = other_actor.id
            
            # Only count this cone if we haven't hit it before
            if cone_id not in hit_cone_ids:
                hit_cone_ids.add(cone_id)
                collision_count += 1
                # print(f"Collision with traffic cone ID {cone_id} (total unique collisions: {collision_count})")

    collision_sensor.listen(_on_collision)

    print("make_carla_client----Camera Done!")
    return client, world, spectator, map, vehicle, camera

def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """
    # Declare global variables at the start of the loop
    global front_camera_detections, front_camera_image, cam_front_ref, is_relighting_active, relight_trigger

    # Initialize event-driven relighting state (starts inactive)
    if ENABLE_EVENT_DRIVEN:
        is_relighting_active = False
        if ConfidenceTrigger is not None:
            relight_trigger = ConfidenceTrigger(tau_low=TAU_LOW, tau_high=TAU_HIGH)
        print("=" * 60)
        print("Event-driven relighting initialized: state = INACTIVE")
        print(f"  Will activate when raw c_max < {TAU_LOW}")
        print(f"  Will deactivate when raw c_max > {TAU_HIGH}")
        print("=" * 60)
    
    # Initialize tracking data storage for record mode
    record_mode = getattr(args, 'mode', 'exp') == 'record'
    tracking_data = []  # List of dicts: {timestamp, x, y, yaw, speed, c_max, frame, ...}
    yolo_cmax_data = []  # List of dicts: {timestamp, c_max, frame, has_detections} - separate for timestamp alignment

    # Progress screenshot configuration
    screenshot_progress_raw = getattr(args, 'screenshot_progress', None)
    screenshot_progress_targets = _normalize_progress_list(screenshot_progress_raw or [])
    screenshot_kind = getattr(args, 'screenshot_kind', 'annotated')
    screenshot_progress_captured = set()
    screenshot_sim_window = getattr(args, 'screenshot_sim_window', False)
    sim_window_name = getattr(args, 'sim_window_name', 'CarlaUE4')
    sim_window_id_cache = None
    exit_after_screenshots = bool(screenshot_progress_targets) and not getattr(args, 'no_exit_after_screenshots', False)
    ended_early_for_screenshots = False
    if screenshot_progress_targets:
        print("=" * 60)
        print("Progress screenshots ENABLED")
        print(f"  Targets: {[f'{p*100:.1f}%' for p in screenshot_progress_targets]}")
        print(f"  Kind: {screenshot_kind}")
        print(f"  Exit after screenshots: {'YES' if exit_after_screenshots else 'no'}")
        if screenshot_sim_window:
            print("  Simulator window screenshots: ENABLED")
            print(f"  Simulator window name: {sim_window_name}")
        else:
            print("  Simulator window screenshots: disabled")
        print(f"  Output: {os.path.join(CONTROLLER_OUTPUT_FOLDER, 'screenshots')}")
        print("=" * 60)

    # Optional: make obstacle spawning + any random decisions reproducible for paired tests
    random_seed = getattr(args, 'random_seed', None)
    if random_seed is not None:
        try:
            random_seed = int(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
            print(f"Random seed set to: {random_seed}")
        except Exception as e:
            print(f"Warning: could not set random seed ({random_seed}): {e}")

    # Reset collision tracking for this simulation run
    global collision_count, hit_cone_ids, total_cone_count, collision_sensor, camera_sensors
    collision_count = 0
    hit_cone_ids = set()
    total_cone_count = 0
    
    # Get collision avoidance setting (default: enabled)
    enable_collision_avoidance = getattr(args, 'enable_collision_avoidance', True)
    if enable_collision_avoidance:
        print("Collision avoidance: ENABLED")
    else:
        print("Collision avoidance: DISABLED")
    
    if args.control_method == 'PurePursuit':
        print("Control method selected : Pure Pursuit\n")
    elif args.control_method == 'Stanley':
        print("Control method selected : Stanley\n")
    elif args.control_method == 'MPC':
        print("Control method selected : MPC\n")
    elif args.control_method == 'MPC_DLC':
        print("Control method selected : MPC_DLC (Double Lane Change MPC)\n")
    elif args.control_method == 'NMPC':
        print("Control method selected : NMPC\n")
    else:
        print("Control method selected : Unknown\n")

    # with make_carla_client(args.host, args.port) as client:
    # Prepare weather arguments
    weather_args = {
        'weather_preset': getattr(args, 'weather_preset', 'clear'),
        'precipitation': getattr(args, 'precipitation', None),
        'fog_density': getattr(args, 'fog_density', None),
        'fog_distance': getattr(args, 'fog_distance', None),
        'cloudiness': getattr(args, 'cloudiness', None),
    }
    
    client, world, spectator, map, vehicle, camera = make_carla_client(
        args.host,
        args.port,
        use_sync_mode=not getattr(args, "no_sync_mode", False),
        fixed_delta=getattr(args, "fixed_delta_seconds", 0.05),
        weather_args=weather_args,
        time_of_day=getattr(args, 'time_of_day', None),
    )

    # #Add spectator camera to get the view to move with the car 
    # camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # # camera_transform = carla.Transform(carla.Location(x=-10,z=10), carla.Rotation(-20,0,0))
    # camera_transform = carla.Transform(carla.Location(x=-10,z=7), carla.Rotation(-20,0,0))
    # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # spectator.set_transform(camera.get_transform())

    print(client)
    print('Carla client connected.')
    # settings = make_carla_settings(args)
    # Now we load these settings into the server. The server replies
    # with a scene description containing the available start spots for
    # the player. Here we can provide a CarlaSettings object or a
    # CarlaSettings.ini file as string.
    # scene = client.load_settings(settings)
    # print(scene)
    # Refer to the player start folder in the WorldOutliner to see the
    # player start information
    # player_start = PLAYER_START_INDEX

    # Notify the server that we want to start the episode at the
    # player_start index. This function blocks until the server is ready
    # to start the episode.
    # print('Starting new episode at %r...' % scene.map_name)
    # client.start_episode(player_start)

    #############################################
    # Load Configurations
    #############################################

    # Load configuration file (options.cfg) and then parses for the various
    # options. Here we have two main options:
    # live_plotting and live_plotting_period, which controls whether
    # live plotting is enabled or how often the live plotter updates
    # during the simulation run.
    config = configparser.ConfigParser()
    config.read(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))
    demo_opt = config['Demo Parameters']

    # Get options
    enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
    enable_live_plot = enable_live_plot == 'True'
    live_plot_period = float(demo_opt.get('live_plotting_period', 0))

    # Allow command-line override for batch runs (avoid GUI overhead)
    if getattr(args, 'disable_live_plotting', False):
        enable_live_plot = False

    # Set options
    live_plot_timer = Timer(live_plot_period)

    #############################################
    # Initialize visualization window
    # Shows only relighted front camera with YOLO bounding boxes
    #############################################
    global camera_display, relighting_service, ENABLE_RELIGHTING
    disable_visualization = getattr(args, 'disable_visualization', False)
    if not disable_visualization:
        pygame.init()
    cam_width = 800
    cam_height = 600
    # Show only relighted front camera with YOLO bounding boxes
    if disable_visualization:
        camera_display = None
        print("Visualization disabled (--disable-visualization). Running headless (no pygame window).")
    else:
        camera_display = pygame.display.set_mode((cam_width, cam_height))
        pygame.display.set_caption("Relighted Front Camera with YOLO Detection")
    
    #############################################
    # Initialize relighting service
    # Optimization C: No separate Tkinter window - integrated into Pygame
    #############################################
    first_relit_received = False  # Reset flag
    
    if ENABLE_RELIGHTING and RELIGHTING_AVAILABLE:
        print("Initializing relighting service...")
        try:
            # Check if async mode is enabled (affects torch.compile() usage)
            use_async = getattr(args, 'async_relighting', False)

            # Disable torch.compile() in async mode to avoid CUDA graph capture conflicts
            enable_compile = not use_async
            if use_async:
                print("  Async mode detected: torch.compile() will be disabled to avoid CUDA conflicts")

            print(f"Loading relighting model: base={args.base_model}, lora={args.lora}")
            pipeline = load_pipeline(args.base_model, args.lora, compile_transformer=enable_compile)
            print("✓ New relighting model loaded successfully!")

            # Get size reduction factor from command-line argument
            # (long side fed to the model = 1024 / factor, e.g. 4 -> 256 px)
            size_factor = getattr(args, 'relight_size_factor', 4)
            max_dim = int(1024 // size_factor)
            relighter = Relighter(pipeline, max_dim=max_dim)

            if use_async:
                # Use async relighting service (non-blocking): 1 inference step per frame,
                # queue of 2 frames that drops the oldest when full.
                # Note: torch.compile() is disabled in async mode to avoid CUDA graph capture conflicts
                relighting_service = AsyncRelighter(relighter, num_inference_steps=1, max_queue_size=2)
                relighting_service.start()
                print(f"✓ Async relighting service initialized successfully!")
                print(f"  Image size reduction factor: {size_factor}x (max dimension: {max_dim}px)")
                print(f"  Async mode: Background thread processing (non-blocking)")
                print(f"  Note: torch.compile() disabled to avoid CUDA conflicts (async is more important)")
                print(f"  Relighted view with YOLO detection displayed in Pygame window")
            else:
                # Use synchronous relighting (blocking, 1 inference step per relit frame)
                relighting_service = relighter
                print(f"✓ Relighting service initialized successfully!")
                print(f"  Image size reduction factor: {size_factor}x (max dimension: {max_dim}px)")
                print(f"  Sync mode: Blocking processing (use --async-relighting for non-blocking)")
                print(f"  Relighted view with YOLO detection displayed in Pygame window")
        except Exception as e:
            print(f"Error initializing relighting service: {e}")
            print("Continuing without relighting...")
            ENABLE_RELIGHTING = False
            relighting_service = None
    else:
        if not RELIGHTING_AVAILABLE:
            print("Relighting modules not available. Skipping relighting initialization.")
        relighting_service = None

    #############################################
    # Load Waypoints
    #############################################
    # Opens the waypoint file and stores it to "waypoints"
    waypoints_file = WAYPOINTS_FILENAME
    waypoints_np   = None
    with open(waypoints_file) as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle,
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        waypoints_np = np.array(waypoints)
    print('Waypoints loaded')
    # print(waypoints_np)

    # Optional: scale reference speed profile so the maximum becomes a target cap
    # (e.g., 120 km/h), preserving the relative shape of the speed profile.
    # Waypoints are stored as [x, y, v] where v is speed in m/s.
    # This modifies ONLY the reference speed column, not the trajectory geometry.
    max_speed_kph = getattr(args, 'max_speed_kph', None)
    if max_speed_kph is not None and waypoints_np.shape[1] >= 3:
        try:
            max_speed_mps = float(max_speed_kph) / 3.6  # km/h -> m/s
            before_max = float(np.max(waypoints_np[:, 2]))
            if before_max > 1e-3:
                scale = min(1.0, max_speed_mps / before_max)
                waypoints_np[:, 2] = waypoints_np[:, 2] * scale
                after_max = float(np.max(waypoints_np[:, 2]))
                print(f"Max speed scaling enabled: target {max_speed_kph} km/h "
                      f"({max_speed_mps:.2f} m/s). "
                      f"Scale factor: {scale:.3f}. "
                      f"Waypoints v_max: {before_max:.2f} -> {after_max:.2f} m/s")
            else:
                print(f"Max speed scaling requested ({max_speed_kph} km/h) "
                      f"but waypoint speeds are near zero (v_max={before_max:.4f} m/s). Skipping.")
        except Exception as e:
            print(f"Warning: could not apply max_speed_kph={max_speed_kph}: {e}")

    for w in waypoints_np:
        world.debug.draw_string(carla.Location(x=w[0],y = w[1]), 'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=255, b=0, a =200), life_time=500.0,
                                    persistent_lines=True)

    #############################################
    # Spawn random static obstacles (ONLY orange traffic cones) near the path
    #############################################
    # Obstacles are placed upright on the road surface, slightly offset
    # left/right from the centerline, and only after an initial segment
    # of the trajectory to avoid intersecting the ego spawn.
    # IMPORTANT: Only spawn traffic cones to avoid car flying out from other props.
    # Get obstacle probability from command-line argument or use default
    obstacle_probability = getattr(args, 'obstacle_probability', 0.00)  # Default: no obstacles
    min_obstacle_index = int(0.01 * waypoints_np.shape[0])  # skip first 0% of path

    # ONLY use traffic cone blueprints - no fallback to other props
    # This prevents spawning large/heavy objects that could launch the car
    bp_lib = world.get_blueprint_library()
    obstacle_blueprints = bp_lib.filter('static.prop.trafficcone01*')
    
    # Filter to only orange traffic cones if available
    # CARLA traffic cone naming typically includes color or type
    # Filter for orange cones specifically to match YOLO detection
    orange_cone_blueprints = []
    for bp in obstacle_blueprints:
        bp_id = bp.id.lower()
        # Look for orange traffic cone variants
        # Common names: trafficcone01, trafficcone02, etc.
        # Some may have color in the name, but most are orange by default
        # Accept all traffic cone variants as they're typically orange
        if 'trafficcone' in bp_id:
            orange_cone_blueprints.append(bp)
    
    # Use filtered list if we found orange cones, otherwise use all traffic cones
    if len(orange_cone_blueprints) > 0:
        obstacle_blueprints = orange_cone_blueprints
        print(f"Found {len(obstacle_blueprints)} orange traffic cone blueprint(s)")
    elif len(obstacle_blueprints) > 0:
        print(f"Using {len(obstacle_blueprints)} traffic cone blueprint(s) (assuming orange)")
    else:
        print("WARNING: No traffic cone blueprints found! No obstacles will be spawned.")
        obstacle_blueprints = []

    obstacles = []
    if len(obstacle_blueprints) > 0:
        for idx, w in enumerate(waypoints_np):
            if idx < min_obstacle_index:
                continue
            if np.random.rand() >= obstacle_probability:
                continue

            # Project onto the road to get a valid base waypoint
            wp_loc = carla.Location(x=w[0], y=w[1], z=0.0)
            road_wp = map.get_waypoint(wp_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if road_wp is None:
                continue

            base_transform = road_wp.transform

            # Compute a conservative random lateral offset so cones are
            # near the path but not directly under the wheels.
            # Keep offset small (0.5–1.5 m) to reduce extreme interactions.
            lateral_offset = np.random.uniform(0.5, 3.5)
            # lateral_offset = np.random.uniform(0.0, 3.5)
            # lateral_offset=0.0

            # Randomly choose left (-1) or right (+1)
            side = -1.0 if np.random.rand() < 0.5 else 1.0
            lateral_offset *= side


            yaw_rad = math.radians(base_transform.rotation.yaw)
            # Right vector in world coordinates
            right_x = math.sin(yaw_rad)
            right_y = -math.cos(yaw_rad)

            spawn_loc = carla.Location(
                x=base_transform.location.x + right_x * lateral_offset,
                y=base_transform.location.y + right_y * lateral_offset,
                # Place exactly on the road surface to avoid dropping and tipping.
                z=base_transform.location.z
            )

            # Keep cones perfectly upright; only align yaw with lane heading.
            spawn_rot = carla.Rotation(
                pitch=0.0,
                yaw=base_transform.rotation.yaw,
                roll=0.0
            )

            transform = carla.Transform(spawn_loc, spawn_rot)

            blueprint = np.random.choice(obstacle_blueprints)
            try:
                obstacle = world.spawn_actor(blueprint, transform)
                # Enable physics on the obstacle for more realistic impacts, if supported.
                try:
                    obstacle.set_simulate_physics(True)
                except Exception:
                    pass
                obstacles.append(obstacle)
            except Exception:
                # If spawn fails (e.g. collision at spawn), skip this waypoint
                pass
    
    print(f"Spawned {len(obstacles)} orange traffic cone obstacle(s) along the path")
    
    # Store total cone count globally for percentage calculation
    total_cone_count = len(obstacles)

    # Because the waypoints are discrete and our controller performs better
    # with a continuous path, here we will send a subset of the waypoints
    # within some lookahead distance from the closest point to the vehicle.
    # Interpolating between each waypoint will provide a finer resolution
    # path and make it more "continuous". A simple linear interpolation
    # is used as a preliminary method to address this issue, though it is
    # better addressed with better interpolation methods (spline
    # interpolation, for example).
    # More appropriate interpolation methods will not be used here for the
    # sake of demonstration on what effects discrete paths can have on
    # the controller. It is made much more obvious with linear
    # interpolation, because in a way part of the path will be continuous
    # while the discontinuous parts (which happens at the waypoints) will
    # show just what sort of effects these points have on the controller.
    # Can you spot these during the simulation? If so, how can you further
    # reduce these effects?

    # Linear interpolation computations
    # Compute a list of distances between waypoints
    wp_distance = []   # distance array
    for i in range(1, waypoints_np.shape[0]):
        wp_distance.append(
                np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                        (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
    wp_distance.append(0)  # last distance is 0 because it is the distance
                            # from the last waypoint to the last waypoint

    # Linearly interpolate between waypoints and store in a list
    wp_interp      = []    # interpolated values
                            # (rows = waypoints, columns = [x, y, v])
    wp_interp_hash = []    # hash table which indexes waypoints_np
                            # to the index of the waypoint in wp_interp
    interp_counter = 0     # counter for current interpolated point index

    print("numpy original: ",waypoints_np.shape[0])

    #
    delete_items = []
    for i in range(waypoints_np.shape[0] - 1):
        
        wp_vector = waypoints_np[i+1] - waypoints_np[i]
        if wp_vector[0] == 0:
            delete_items.append(i+1)

    waypoints_np = np.delete(waypoints_np, delete_items, 0)
    print("numpy deleted: ", waypoints_np.shape[0])
    for i in range(waypoints_np.shape[0] - 1):
        # Add original waypoint to interpolated waypoints list (and append
        # it to the hash table)
        wp_interp.append(list(waypoints_np[i]))
        wp_interp_hash.append(interp_counter)
        interp_counter+=1

        # Interpolate to the next waypoint. First compute the number of
        # points to interpolate based on the desired resolution and
        # incrementally add interpolated points until the next waypoint
        # is about to be reached.
        num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                        float(INTERP_DISTANCE_RES)) - 1)
        wp_vector = waypoints_np[i+1] - waypoints_np[i]
        wp_uvector = wp_vector / np.linalg.norm(wp_vector)
        for j in range(num_pts_to_interp):
            next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
            wp_interp.append(list(waypoints_np[i] + next_wp_vector))
            interp_counter+=1
    print('Interpolate Finish')
    # add last waypoint at the end
    wp_interp.append(list(waypoints_np[-1]))
    wp_interp_hash.append(interp_counter)
    interp_counter+=1

    #############################################
    # Controller 2D Class Declaration
    #############################################
    # This is where we take the controller2d.py class
    # and apply it to the simulator
    # controller = controller2d.Controller2D(waypoints, args.control_method)
    controller = controller2d.Controller2D(waypoints, args.control_method, world)
    print('Controller loaded')
    #############################################
    # Determine simulation average timestep (and total frames)
    #############################################
    # Ensure at least one frame is used to compute average timestep
    num_iterations = ITER_FOR_SIM_TIMESTEP
    if (ITER_FOR_SIM_TIMESTEP < 1):
        num_iterations = 1
    print("Closed")
    
    # Gather current data from the CARLA server. This is used to get the
    # simulator starting game time. Note that we also need to
    # send a command back to the CARLA server because synchronous mode
    # is enabled.
    # measurement_data, sensor_data = client.read_data()

    # Determine simulation timestep depending on sync/async mode.
    sim_settings = world.get_settings()
    if sim_settings.synchronous_mode and sim_settings.fixed_delta_seconds not in (None, 0.0):
        # In synchronous mode with fixed_delta_seconds, the timestep is known.
        SIMULATION_TIME_STEP = sim_settings.fixed_delta_seconds
        # Advance one tick to establish a starting timestamp
        world.tick()
        timestamp = world.get_snapshot().timestamp
        sim_start_stamp = timestamp.elapsed_seconds
        send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
    else:
        # In asynchronous mode, estimate average timestep over several frames.
        sim_duration = 0.0
        # Initial timestamp
        timestamp = world.wait_for_tick()
        sim_start_stamp = timestamp.elapsed_seconds
        last_stamp = sim_start_stamp
        for i in range(num_iterations):
            timestamp = world.wait_for_tick()
            now_stamp = timestamp.elapsed_seconds
            sim_duration += (now_stamp - last_stamp)
            last_stamp = now_stamp
        # Avoid division by zero
        SIMULATION_TIME_STEP = sim_duration / float(max(num_iterations, 1)) if sim_duration > 0 else 1.0 / 20.0
        send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)

    print("SERVER SIMULATION STEP APPROXIMATION: " + str(SIMULATION_TIME_STEP))
    
    # Get simulation timeout from args or use default
    global SIMULATION_TIMEOUT
    sim_timeout = getattr(args, 'simulation_timeout', None)
    if sim_timeout is not None:
        SIMULATION_TIMEOUT = float(sim_timeout)
        print(f"Simulation timeout set to: {SIMULATION_TIMEOUT} seconds")
    else:
        SIMULATION_TIMEOUT = TOTAL_RUN_TIME  # Use default timeout
    
    TOTAL_EPISODE_FRAMES = int((SIMULATION_TIMEOUT + WAIT_TIME_BEFORE_START) /\
                            SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER
    print(f"Total simulation time: {SIMULATION_TIMEOUT} seconds")
    print("total frame: ",TOTAL_EPISODE_FRAMES)
    #############################################
    # Frame-by-Frame Iteration and Initialization
    #############################################
    # Store pose history starting from the start position
    # measurement_data, sensor_data = client.read_data()
    # start_x, start_y, start_yaw = get_current_pose(measurement_data)
    start_x = round(vehicle.get_transform().location.x,2)
    start_y = round(vehicle.get_transform().location.y,2)
    start_yaw = round(vehicle.get_transform().rotation.yaw,2)

    send_control_command(vehicle,throttle=0.0, steer=0, brake=1.0)
    x_history     = [start_x]
    y_history     = [start_y]
    yaw_history   = [start_yaw]
    time_history  = [0]
    speed_history = [0]
    desired_speed_history = [0]  # Track desired/reference speed for speed error calculation
    print(start_x,start_y,start_yaw)
    #############################################
    # Vehicle Trajectory Live Plotting Setup
    #############################################
    # Uses the live plotter to generate live feedback during the simulation
    # The two feedback includes the trajectory feedback and
    # the controller feedback (which includes the speed tracking).
    lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
    lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

    ###
    # Add 2D position / trajectory plot
    ###
    trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
            title='Vehicle Trajectory',
            figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
            edgecolor="black",
            rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

    trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed
                                        # coordinate system the X
                                        # axis in the graph is flipped
    trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

    # Add waypoint markers
    trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                                linestyle="-", marker="", color='g')
    # Add trajectory markers
    trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                x0=[start_x]*TOTAL_EPISODE_FRAMES,
                                y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                color=[1, 0.5, 0])
    # Add lookahead path
    trajectory_fig.add_graph("lookahead_path",
                                window_size=INTERP_MAX_POINTS_PLOT,
                                x0=[start_x]*INTERP_MAX_POINTS_PLOT,
                                y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                color=[0, 0.7, 0.7],
                                linewidth=4)
    # Add starting position marker
    trajectory_fig.add_graph("start_pos", window_size=1,
                                x0=[start_x], y0=[start_y],
                                marker=11, color=[1, 0.5, 0],
                                markertext="Start", marker_text_offset=1)
    # Add end position marker
    trajectory_fig.add_graph("end_pos", window_size=1,
                                x0=[waypoints_np[-1, 0]],
                                y0=[waypoints_np[-1, 1]],
                                marker="D", color='r',
                                markertext="End", marker_text_offset=1)
    # Add car marker
    trajectory_fig.add_graph("car", window_size=1,
                                marker="s", color='b', markertext="Car",
                                marker_text_offset=1)

    ###
    # Add 1D speed profile updater
    ###
    forward_speed_fig =\
            lp_1d.plot_new_dynamic_figure(title="Forward Speed (km/h)")
    forward_speed_fig.add_graph("forward_speed",
                                label="forward_speed",
                                window_size=TOTAL_EPISODE_FRAMES)
    forward_speed_fig.add_graph("reference_signal",
                                label="reference_Signal",
                                window_size=TOTAL_EPISODE_FRAMES)

    # Add throttle signals graph
    throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle (%)")
    throttle_fig.add_graph("throttle",
                            label="throttle",
                            window_size=TOTAL_EPISODE_FRAMES)
    # Add brake signals graph
    brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake (%)")
    brake_fig.add_graph("brake",
                            label="brake",
                            window_size=TOTAL_EPISODE_FRAMES)

    # Add steering signals graph
    steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer (Degree)")
    steer_fig.add_graph("steer",
                            label="steer",
                            window_size=TOTAL_EPISODE_FRAMES)

    # Add collision counter graph
    collision_fig = lp_1d.plot_new_dynamic_figure(title="Collisions")
    collision_fig.add_graph("collision_count",
                            label="collision_count",
                            window_size=TOTAL_EPISODE_FRAMES)

    # live plotter is disabled, hide windows
    if not enable_live_plot:
        lp_traj._root.withdraw()
        lp_1d._root.withdraw()

    # Iterate the frames until the end of the waypoints is reached or
    # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
    # ouptuts the results to the controller output directory.
    reached_the_end = False
    skip_first_frame = True
    closest_index    = 0  # Index of waypoint that is currently closest to
                            # the car (assumed to be the first index)
    closest_distance = 0  # Closest distance of closest waypoint to car
    
    # Lap time tracking
    lap_start_time = None  # Will be set when controller actually starts (after WAIT_TIME_BEFORE_START)
    lap_end_time = None    # Will be set when simulation ends

    # #Add spectator camera to get the view to move with the car 
    # camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # # camera_transform = carla.Transform(carla.Location(x=-10,z=10), carla.Rotation(-20,0,0))
    # camera_transform = carla.Transform(carla.Location(x=-10,z=7), carla.Rotation(-20,0,0))
    # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)


    sim_settings = world.get_settings()

    # Initialize performance profiler
    profiler = PerformanceProfiler(
        enabled=getattr(args, 'enable_profiling', False),
        report_interval=100
    )
    if profiler.enabled:
        print("=" * 80)
        print("PERFORMANCE PROFILING ENABLED")
        print("Statistics will be reported every 100 frames")
        print("=" * 80)
    
    # Get relighting frame skip setting
    relight_skip_frames = getattr(args, 'relight_skip_frames', 1)
    if ENABLE_RELIGHTING and relight_skip_frames > 1:
        print(f"Relighting will process every {relight_skip_frames} frames (skip {relight_skip_frames-1} frames)")

    for frame in range(TOTAL_EPISODE_FRAMES):
        profiler.start_frame()
        
        
        # Optimization B: Restructured main loop following recommended order:
        # 1. world.tick() (Advance physics)
        # 2. Get vehicle state
        # 3. Inference: Run Relighting + YOLO (Keep on GPU)
        # 4. Control: Update controller2d based on detection
        # 5. Visualization: Update Pygame (The lowest priority)
        
        # Initialize variables for this frame
        current_x = 0.0
        current_y = 0.0
        current_yaw = 0.0
        current_speed = 0.0
        current_timestamp = 0.0
        cmd_throttle = 0.0
        cmd_steer = 0.0
        cmd_brake = 0.0
        cmd_reverse = False

        # STEP 1: Advance physics simulation (world.tick)
        def carla_tick():
            nonlocal current_timestamp
            if sim_settings.synchronous_mode:
                world.tick()
                current_timestamp = world.get_snapshot().timestamp.elapsed_seconds
            else:
                timestamp = world.wait_for_tick()
                current_timestamp = timestamp.elapsed_seconds
        
        profiler.time_stage('carla_tick', carla_tick)

        # STEP 2: Get vehicle state
        def get_vehicle_state():
            nonlocal current_x, current_y, current_yaw, current_speed
            current_x = vehicle.get_transform().location.x
            current_y = vehicle.get_transform().location.y
            current_yaw = math.radians(vehicle.get_transform().rotation.yaw)
            # compensate for CoG 
            current_x, current_y = controller.get_shifted_coordinate(current_x, current_y, current_yaw, math.sqrt(4**2+0.5**2))
            current_speed = math.sqrt(vehicle.get_velocity().x**2+vehicle.get_velocity().y**2)
        
        profiler.time_stage('vehicle_state', get_vehicle_state)

        # STEP 3: Inference - Run Relighting + YOLO (Keep on GPU when possible)
        # This is the critical path for real-time performance.
        # IMPORTANT: Perception (relighting + YOLO) must run even when visualization is disabled.
        # The pygame window and blitting are purely optional.
        # Handle pygame window events (allow user to close camera window)
        if not disable_visualization and pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()

        # Show only relighted front camera with YOLO bounding boxes
        # Perception runs regardless of whether camera_display exists; visualization is optional.
        cam_names = ["front"]  # Only process front camera
        relit_front_image = None  # Store relit front camera image
        global last_relit_image, last_relit_np
        use_visualization = camera_display is not None

        # For progress-based screenshot saving (captured later, after progress update)
        front_raw_image_for_screenshot = None
        front_relit_image_for_screenshot = None  # numpy RGB if available
        front_annotated_image_for_screenshot = None  # numpy RGB with YOLO boxes if available
        
        # Process front camera only
        for name in cam_names:
            # Time camera frame retrieval (just getting the frame from dict)
            cam_retrieval_start = time.time()
            image = latest_camera_frames.get(name)
            profiler.record_timing('camera_retrieval', time.time() - cam_retrieval_start)
            
            if image is None:
                continue

            # EVENT-DRIVEN RELIGHTING LOGIC (if enabled):
            # 1. First run YOLO on raw image to get c_max
            # 2. Check hysteresis based on raw c_max
            # 3. Conditionally run relighting based on state
            # 4. Use appropriate detections (raw or from relit image)
            
            # Prepare raw image (apply CLAHE if enabled, but not relighting yet)
            raw_image_for_yolo = image
            if ENABLE_CLAHE and CLAHE_AVAILABLE and name == "front" and not ENABLE_RELIGHTING:
                # CLAHE only if relighting is not enabled (they're mutually exclusive)
                try:
                    clahe_start = time.time()
                    raw_image_for_yolo = apply_clahe(image)
                    profiler.record_timing('clahe', time.time() - clahe_start)
                    if frame == 0:
                        print("=" * 60)
                        print("✓ CLAHE enhancement enabled")
                        print("=" * 60)
                except Exception as e:
                    print(f"Error during CLAHE processing: {e}")
                    import traceback
                    traceback.print_exc()
                    raw_image_for_yolo = image
            
            # Initialize variables for this frame
            image_for_yolo = raw_image_for_yolo
            raw_c_max = None
            raw_detections = None
            relighting_triggered_this_frame = False
            
            # EVENT-DRIVEN MODE: Run YOLO on raw image first to get c_max for hysteresis check
            if ENABLE_EVENT_DRIVEN and ENABLE_RELIGHTING and relighting_service is not None and name == "front":
                # Step 1: Run YOLO on raw image to get c_max
                raw_yolo_start = time.time()
                _, raw_detections = detect_obstacles(raw_image_for_yolo)
                profiler.record_timing('yolo_detection', time.time() - raw_yolo_start)
                
                # Step 2: Extract c_max from raw detections
                if raw_detections is not None and hasattr(raw_detections, 'conf') and len(raw_detections.conf) > 0:
                    try:
                        conf_tensor = raw_detections.conf
                        if hasattr(conf_tensor, 'max'):
                            raw_c_max = float(conf_tensor.max().item())
                        elif hasattr(conf_tensor, '__len__') and len(conf_tensor) > 0:
                            if hasattr(conf_tensor, 'cpu'):
                                raw_c_max = float(np.max(conf_tensor.cpu().numpy()))
                            else:
                                raw_c_max = float(np.max(conf_tensor))
                    except Exception as e:
                        if frame % 100 == 0:
                            print(f"Warning: Could not extract c_max from raw YOLO detections: {e}")
                
                # Step 3: Hysteresis check on raw image confidence (relight.ConfidenceTrigger):
                # activate when raw c_max < tau_low, deactivate when raw c_max > tau_high,
                # frames with no detection (c_max is None) keep the current state.
                # Note: is_relighting_active is already declared as global at function start
                was_relighting_active = is_relighting_active
                is_relighting_active = relight_trigger.update(raw_c_max)
                # "Triggered" = a detection was seen this frame and relighting is (still) active
                relighting_triggered_this_frame = raw_c_max is not None and is_relighting_active
                if frame % 30 == 0 and is_relighting_active != was_relighting_active:  # Log occasionally
                    if is_relighting_active:
                        print(f"[Frame {frame}] Event-driven: ACTIVATED (raw_c_max={raw_c_max:.3f} < {TAU_LOW})")
                    else:
                        print(f"[Frame {frame}] Event-driven: DEACTIVATED (raw_c_max={raw_c_max:.3f} > {TAU_HIGH})")

                # Step 4: Execute relighting if active
                if is_relighting_active:
                    # Run relighting on raw image (every frame when active)
                    try:
                        relight_start = time.time()
                        relit_pil = relight_carla_image(relighting_service, image, num_inference_steps=1)
                        profiler.record_timing('relighting', time.time() - relight_start)

                        if relit_pil is not None:
                            # Convert PIL to numpy array for YOLO
                            relit_np = np.array(relit_pil)
                            # Store for potential reuse
                            last_relit_image = relit_pil
                            last_relit_np = relit_np
                            image_for_yolo = relit_np
                            relit_front_image = relit_pil
                            
                            if not first_relit_received:
                                first_relit_received = True
                                print("=" * 60)
                                print("✓ First relit image received (event-driven mode)")
                                print(f"  Image size: {relit_pil.size}")
                                print("=" * 60)
                        else:
                            # Relighting failed, use raw image
                            image_for_yolo = raw_image_for_yolo
                            raw_detections = None  # Will re-run YOLO below
                    except Exception as e:
                        print(f"Error during event-driven relighting: {e}")
                        import traceback
                        traceback.print_exc()
                        image_for_yolo = raw_image_for_yolo
                        raw_detections = None  # Will re-run YOLO below
                else:
                    # Not active: use raw detections (already computed above)
                    image_for_yolo = raw_image_for_yolo
                    # raw_detections already computed, will use them below
            
            # CONTINUOUS RELIGHTING MODE (original behavior)
            elif ENABLE_RELIGHTING and relighting_service is not None and name == "front":
                # Check if using async relighting
                is_async = isinstance(relighting_service, AsyncRelighter)

                if is_async:
                    # ASYNC MODE: Non-blocking relighting
                    # Request relighting (non-blocking, queues the RGB frame; oldest dropped when full)
                    should_request = (frame % relight_skip_frames == 0)
                    if should_request:
                        queued = relighting_service.submit(carla_image_to_rgb_array(image))
                        if queued:
                            profiler.record_timing('relighting', 0.0)  # No blocking time
                        else:
                            # Queue full, image dropped (use last available)
                            pass

                    # Get latest relit image (thread-safe, non-blocking)
                    relit_pil, relit_np = relighting_service.latest()

                    # Track async relighting processing time (non-blocking, but we track it)
                    # Mark as async=True so it doesn't count toward overhead calculation
                    async_stats = relighting_service.stats()
                    if async_stats['avg_time'] > 0:
                        profiler.record_timing('async_relighting', async_stats['avg_time'], is_async=True)
                    
                    if relit_np is not None:
                        # Use the latest relit image
                        image_for_yolo = relit_np
                        relit_front_image = relit_pil
                        
                        # Mark that we've received the first relit image
                        if not first_relit_received:
                            first_relit_received = True
                            print("=" * 60)
                            print("✓ First relit image received successfully!")
                            if relit_pil is not None:
                                print(f"  Image size: {relit_pil.size}")
                            print("=" * 60)
                    else:
                        # No relit image available yet (first frame or still processing)
                        if not first_relit_received and frame == 0:
                            print("=" * 60)
                            print("Waiting for first relit image... (processing in background)")
                            print("=" * 60)
                else:
                    # SYNC MODE: Blocking relighting (original behavior)
                    should_relight = (frame % relight_skip_frames == 0)
                    
                    if should_relight:
                        # Process new relighting (blocking)
                        try:
                            if not first_relit_received:
                                print("=" * 60)
                                print("Processing first relit image... This may take 10-30 seconds.")
                                print("=" * 60)
                            
                            # Time relighting separately
                            # Note: CUDA lock only needed if torch.compile() is enabled (sync mode)
                            relight_start = time.time()
                            # Relight the image (this is synchronous and will block until done)
                            relit_pil = relight_carla_image(relighting_service, image, num_inference_steps=1)
                            profiler.record_timing('relighting', time.time() - relight_start)

                            if relit_pil is not None:
                                # Convert PIL to numpy array for YOLO (RGB format)
                                relit_np = np.array(relit_pil)
                                # Store for reuse on skipped frames
                                last_relit_image = relit_pil
                                last_relit_np = relit_np
                                # Use numpy array directly (detect_obstacles now accepts numpy arrays)
                                image_for_yolo = relit_np
                                relit_front_image = relit_pil  # Store relit image
                                
                                # Mark that we've received the first relit image
                                if not first_relit_received:
                                    first_relit_received = True
                                    print("=" * 60)
                                    print("✓ First relit image received successfully!")
                                    print(f"  Image size: {relit_pil.size}")
                                    print("=" * 60)
                                
                                if frame % 30 == 0:  # Print every 30 frames to avoid spam
                                    print(f"Relighting successful - relit image size: {relit_pil.size}")
                            else:
                                if not first_relit_received:
                                    print("Warning: Relighting returned None. Retrying...")
                        except Exception as e:
                            print(f"Error during relighting: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fallback to original image
                            image_for_yolo = image
                    else:
                        # Reuse last relit image when skipping frames
                        if last_relit_np is not None:
                            image_for_yolo = last_relit_np
                            relit_front_image = last_relit_image
                            # Record 0 time for relighting when skipping
                            profiler.record_timing('relighting', 0.0)
                        # else: use original image (first frame or relighting failed)
            
            # Run YOLO detection (if not already done in event-driven mode)
            annotated_img = None
            detections = None
            c_max = None
            has_detections = False
            
            if ENABLE_EVENT_DRIVEN and ENABLE_RELIGHTING and name == "front":
                # Event-driven mode: use raw detections if not active, or run YOLO on relit image if active
                if is_relighting_active:
                    # Relighting is active: run YOLO on relit image
                    yolo_start = time.time()
                    annotated_img, detections = detect_obstacles(image_for_yolo)
                    profiler.record_timing('yolo_detection', time.time() - yolo_start)
                else:
                    # Not active: use raw detections (already computed above)
                    detections = raw_detections
                    # Generate annotated image from raw detections for visualization
                    if raw_detections is not None:
                        # Re-run detect_obstacles to get annotated image (or create one from raw)
                        yolo_start = time.time()
                        annotated_img, _ = detect_obstacles(raw_image_for_yolo)
                        profiler.record_timing('yolo_detection', time.time() - yolo_start)
            else:
                # Continuous mode or no relighting: run YOLO normally
                yolo_start = time.time()
                annotated_img, detections = detect_obstacles(image_for_yolo)
                profiler.record_timing('yolo_detection', time.time() - yolo_start)
            
            # Extract c_max (maximum confidence) from YOLO detections
            # For event-driven mode, use raw_c_max if available, otherwise extract from current detections
            if ENABLE_EVENT_DRIVEN and ENABLE_RELIGHTING and raw_c_max is not None:
                c_max = raw_c_max  # Use raw c_max for event-driven decision tracking
            elif detections is not None and hasattr(detections, 'conf') and len(detections.conf) > 0:
                try:
                    # Get maximum confidence from all detections
                    conf_tensor = detections.conf
                    if hasattr(conf_tensor, 'max'):
                        c_max = float(conf_tensor.max().item())
                    elif hasattr(conf_tensor, '__len__') and len(conf_tensor) > 0:
                        # Fallback: convert to numpy/float if needed
                        if hasattr(conf_tensor, 'cpu'):
                            c_max = float(np.max(conf_tensor.cpu().numpy()))
                        else:
                            c_max = float(np.max(conf_tensor))
                    has_detections = True
                except Exception as e:
                    # If extraction fails, c_max remains None
                    if frame % 100 == 0:  # Log occasionally
                        print(f"Warning: Could not extract c_max from YOLO detections: {e}")
            
            # Set has_detections flag
            if detections is not None and hasattr(detections, 'conf') and len(detections.conf) > 0:
                has_detections = True
            
            # Store detections from front camera for obstacle avoidance
            if name == "front" and detections is not None:
                # Store front camera detections globally for waypoint modification
                front_camera_detections = detections
                # Store the actual image used for YOLO (may be relit & resized)
                front_camera_image = image_for_yolo
                # Track its size for correct 2D→3D projection
                if isinstance(image_for_yolo, np.ndarray):
                    front_yolo_img_height, front_yolo_img_width = image_for_yolo.shape[:2]
                else:
                    # Fallback for CARLA Image
                    front_yolo_img_width = getattr(image, "width", 800)
                    front_yolo_img_height = getattr(image, "height", 600)
                
                # Record YOLO c_max data (with its own timestamp for alignment)
                if record_mode and name == "front":
                    yolo_cmax_data.append({
                        'timestamp': current_timestamp,
                        'c_max': c_max,
                        'frame': frame,
                        'has_detections': has_detections,
                        'num_detections': len(detections.xyxy) if hasattr(detections, 'xyxy') and detections.xyxy is not None else 0,
                        'relighting_triggered': relighting_triggered_this_frame if ENABLE_EVENT_DRIVEN else False,
                        'is_relighting_active': is_relighting_active if ENABLE_EVENT_DRIVEN else False,
                    })

            # Cache latest images for progress screenshots (front camera only)
            if name == "front":
                front_raw_image_for_screenshot = image
                front_annotated_image_for_screenshot = annotated_img
                # If relighting is active, image_for_yolo is typically a numpy RGB array
                if ENABLE_RELIGHTING and isinstance(image_for_yolo, np.ndarray):
                    front_relit_image_for_screenshot = image_for_yolo
            
            # Visualization: only show relighted front camera with YOLO bounding boxes
            if use_visualization and name == "front":
                img_conv_start = time.time()
                if annotated_img is not None:
                    # Convert annotated numpy array to pygame surface
                    # annotated_img is already RGB from YOLO with bounding boxes
                    # pygame expects (width, height) format
                    annotated_surf = pygame.surfarray.make_surface(
                        annotated_img.swapaxes(0, 1)
                    )
                    # Always resize to match window size (cam_width x cam_height)
                    # This ensures consistent visualization width regardless of relighting resize factor
                    annotated_surf = pygame.transform.scale(annotated_surf, (cam_width, cam_height))
                    camera_display.blit(annotated_surf, (0, 0))
                else:
                    # Fallback to original image if detection fails
                    surf = carla_image_to_surface(image)
                    if surf is not None:
                        # Always resize to match window size for consistency
                        surf = pygame.transform.scale(surf, (cam_width, cam_height))
                        camera_display.blit(surf, (0, 0))
                profiler.record_timing('image_conversion', time.time() - img_conv_start)
        
        # STEP 5: Visualization - Update Pygame (lowest priority)
        # Time pygame display flip
        if use_visualization:
            pygame_start = time.time()
            pygame.display.flip()
            profiler.record_timing('pygame_render', time.time() - pygame_start)

        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        # spectator.set_transform(carla.Transform(transform.location + carla.Location(x=0,y = -10,z=20),
        # carla.Rotation(-20,90,0)))
 
        spectator.set_transform(camera.get_transform())

        # spectator.set_transform(carla.Transform(transform.location + carla.Location(x=0,y = -10,z=15),
        # carla.Rotation(-20,90,0)))

        # CAMERA SETTING


        # Update pose, timestamp
        # current_x, current_y, current_yaw = \
        #     get_current_pose(measurement_data)
        # current_speed = measurement_data.player_measurements.forward_speed
        # current_timestamp = float(measurement_data.game_timestamp) / 1000.0

        # Shift x, y coordinates
        if args.control_method == 'PurePursuit':
            # length = -1*controller.b
            length = -1.2
        elif args.control_method == 'Stanley' or args.control_method == 'MPC' or args.control_method == 'NMPC':
            # length = controller.a
            length = 1.2
        else:
            length = 0.0
        # Time other operations (history updates, debug drawing, etc.)
        other_ops_start = time.time()
        
        shifted_x, shifted_y = controller.get_shifted_coordinate(current_x, current_y, current_yaw, length)
        world.debug.draw_string(carla.Location(x=shifted_x,y = shifted_y), 'X', draw_shadow=False,
                            color=carla.Color(r=0, g=0, b=255, a =255), life_time=0.01,
                            persistent_lines=True)
        # Wait for some initial time before starting the demo
        if current_timestamp <= WAIT_TIME_BEFORE_START:
            send_control_command(vehicle,throttle=0.0, steer=0, brake=1.0)
            profiler.record_timing('other_operations', time.time() - other_ops_start)
            profiler.end_frame()
            continue
        else:
            # Controller starts here - record lap start time (only once)
            # Set to 0 since this is the start of the actual lap (after wait period)
            if lap_start_time is None:
                lap_start_time = 0.0
            current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START

        # Store history
        x_history.append(current_x)
        y_history.append(current_y)
        yaw_history.append(current_yaw)
        speed_history.append(current_speed)
        # Track desired speed (updated by controller.update_controls() above)
        desired_speed_history.append(controller._desired_speed)
        time_history.append(current_timestamp)
        
        # Store tracking data for record mode (per timestamp, aligned with full tracking)
        if record_mode:
            # Get the most recent c_max (from YOLO detection in this frame or previous)
            # Since YOLO runs on front camera, we'll match by frame or use the most recent
            current_c_max = None
            # Find c_max from the most recent YOLO detection (same frame or closest previous)
            for yolo_entry in reversed(yolo_cmax_data):
                if yolo_entry['frame'] <= frame:
                    current_c_max = yolo_entry['c_max']
                    break
            
            tracking_data.append({
                'timestamp': current_timestamp,
                'frame': frame,
                'x': current_x,
                'y': current_y,
                'yaw': current_yaw,
                'speed': current_speed,
                'desired_speed': controller._desired_speed,
                'c_max': current_c_max,  # May be None if no YOLO detection yet
                'throttle': cmd_throttle,
                'steer': cmd_steer,
                'brake': cmd_brake,
            })
        
        profiler.record_timing('other_operations', time.time() - other_ops_start)
        
        ###
        # STEP 4: Control - Update controller2d based on detection
        # Controller update (this uses the controller2d.py implementation)
        ###

        # To reduce the amount of waypoints sent to the controller,
        # provide a subset of waypoints that are within some
        # lookahead distance from the closest point to the car. Provide
        # a set of waypoints behind the car as well.

        # Find closest waypoint index to car. First increment the index
        # from the previous index until the new distance calculations
        # are increasing. Apply the same rule decrementing the index.
        # The final index should be the closest point (it is assumed that
        # the car will always break out of instability points where there
        # are two indices with the same minimum distance, as in the
        # center of a circle)
        def find_closest_waypoint():
            nonlocal closest_index, closest_distance
            closest_distance = np.linalg.norm(np.array([
                    waypoints_np[closest_index, 0] - current_x,
                    waypoints_np[closest_index, 1] - current_y]))

            new_distance = closest_distance
            # print(new_distance)
            new_index = closest_index
            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index = new_index
                new_index += 1
                if new_index >= waypoints_np.shape[0]:  # End of path
                    break
                new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))
            new_distance = closest_distance
            new_index = closest_index
            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index = new_index
                new_index -= 1
                if new_index < 0:  # Beginning of path
                    break
                new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))
        
        profiler.time_stage('waypoint_finding', find_closest_waypoint)

        # Track progress as a percentage of reference waypoint index.
        # This is deterministic w.r.t. the reference path and is stable across lighting conditions
        # when using the same random seed for obstacle placement.
        denom = max(1, waypoints_np.shape[0] - 1)
        track_progress = float(closest_index) / float(denom)
        if frame % 50 == 0:
            print(f"[Frame {frame}] Track progress: {track_progress*100:.1f}% (closest_index={closest_index}/{denom})")

        # Save progress-based screenshots when thresholds are crossed.
        # Saves into: <output_dir>/screenshots/progress_010p_frame_000123.png
        if screenshot_progress_targets and len(screenshot_progress_captured) < len(screenshot_progress_targets):
            for p in screenshot_progress_targets:
                if p in screenshot_progress_captured:
                    continue
                if track_progress >= p:
                    try:
                        create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
                        screenshots_dir = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'screenshots')
                        if not os.path.exists(screenshots_dir):
                            os.makedirs(screenshots_dir)
                        pct_int = int(round(p * 100.0))
                        fname = f"progress_{pct_int:03d}p_frame_{frame:06d}.png"
                        out_path = os.path.join(screenshots_dir, fname)

                        if screenshot_kind == 'both':
                            # Save RAW (dark) and RELIT images separately.
                            # IMPORTANT: For relit screenshot we bypass event-driven state and
                            # force a one-off relighting of the raw frame so the "after" view
                            # is always a true relighted version of the same moment.
                            raw_img = carla_image_to_rgb_array(front_raw_image_for_screenshot)
                            relit_img = None

                            # (Synchronous modes only: in async mode the pipeline belongs to the
                            # background thread, so the latest async relit frame is used instead.)
                            if (ENABLE_RELIGHTING and relighting_service is not None
                                    and not isinstance(relighting_service, AsyncRelighter)
                                    and front_raw_image_for_screenshot is not None):
                                try:
                                    relight_start_ss = time.time()
                                    relit_pil_ss = relight_carla_image(relighting_service, front_raw_image_for_screenshot, num_inference_steps=1)
                                    profiler.record_timing('relighting', time.time() - relight_start_ss)
                                    if relit_pil_ss is not None:
                                        relit_img = np.array(relit_pil_ss)
                                except Exception as e:
                                    print(f"[Screenshot] Error during forced relighting for screenshot at {p*100:.1f}%: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # Fallbacks: if forced relighting failed, use previously cached relit
                            # image (if any), then annotated, then raw.
                            if relit_img is None and front_relit_image_for_screenshot is not None:
                                relit_img = front_relit_image_for_screenshot
                            if relit_img is None and front_annotated_image_for_screenshot is not None:
                                relit_img = front_annotated_image_for_screenshot
                            if relit_img is None:
                                relit_img = raw_img

                            # Apply YOLO detection to both images to get bounding boxes
                            raw_annotated = None
                            relit_annotated = None
                            
                            if raw_img is not None:
                                raw_annotated, _ = detect_obstacles(raw_img)
                                if raw_annotated is None:
                                    raw_annotated = raw_img
                                    
                            if relit_img is not None:
                                relit_annotated, _ = detect_obstacles(relit_img)
                                if relit_annotated is None:
                                    relit_annotated = relit_img

                            if raw_annotated is not None:
                                raw_fname = f"progress_{pct_int:03d}p_raw_frame_{frame:06d}.png"
                                raw_path = os.path.join(screenshots_dir, raw_fname)
                                plt.imsave(raw_path, raw_annotated)
                                print(f"[Screenshot] RAW (pre-relight) with YOLO saved at {p*100:.1f}% -> {raw_path}")

                            if relit_annotated is not None:
                                relit_fname = f"progress_{pct_int:03d}p_relit_frame_{frame:06d}.png"
                                relit_path = os.path.join(screenshots_dir, relit_fname)
                                plt.imsave(relit_path, relit_annotated)
                                print(f"[Screenshot] RELIT (post-relight) with YOLO saved at {p*100:.1f}% -> {relit_path}")

                            if raw_annotated is not None or relit_annotated is not None:
                                screenshot_progress_captured.add(p)

                        else:
                            img_to_save = None
                            if screenshot_kind == 'annotated':
                                img_to_save = front_annotated_image_for_screenshot
                            elif screenshot_kind == 'relit':
                                img_to_save = front_relit_image_for_screenshot
                            elif screenshot_kind == 'raw':
                                img_to_save = carla_image_to_rgb_array(front_raw_image_for_screenshot)

                            # Fallbacks (avoid losing the moment)
                            if img_to_save is not None:
                                plt.imsave(out_path, img_to_save)
                                screenshot_progress_captured.add(p)
                                print(f"[Screenshot] Saved at {p*100:.1f}% -> {out_path}")

                        # Also save simulator window screenshot (CarlaUE4) if requested
                        if screenshot_sim_window:
                            sim_fname = f"sim_progress_{pct_int:03d}p_frame_{frame:06d}.png"
                            sim_out_path = os.path.join(screenshots_dir, sim_fname)
                            ok, sim_window_id_cache = _capture_simulator_window_png(
                                sim_out_path, window_name=sim_window_name, cached_window_id=sim_window_id_cache
                            )
                            if ok:
                                print(f"[Screenshot] Simulator window saved -> {sim_out_path}")
                            else:
                                print("[Screenshot] Simulator window capture failed (no DISPLAY, tools missing, or window not found).")
                        else:
                            print(f"[Screenshot] Warning: no image available at {p*100:.1f}% (skipping)")
                            screenshot_progress_captured.add(p)
                    except Exception as e:
                        print(f"[Screenshot] Error saving at {p*100:.1f}%: {e}")
                        import traceback
                        traceback.print_exc()
                        screenshot_progress_captured.add(p)

                    # If we captured all requested progress screenshots, end the run early (screenshot mode).
                    if exit_after_screenshots and len(screenshot_progress_captured) >= len(screenshot_progress_targets):
                        ended_early_for_screenshots = True
                        print("=" * 80)
                        print("EARLY EXIT REQUESTED (Screenshot Mode)")
                        print(f"Captured all progress screenshots: {[f'{pp*100:.1f}%' for pp in screenshot_progress_targets]}")
                        print("Ending episode early and writing outputs...")
                        print("=" * 80)
                        # Record lap end time for reporting
                        if lap_start_time is not None:
                            lap_end_time = current_timestamp
                        # Ensure we break out of frame loop
                        break

            # If we broke out to end early for screenshots, stop processing the rest of this frame loop
            if ended_early_for_screenshots:
                break
        
        # Continue with waypoint subset calculation (this is part of waypoint processing)


        # Once the closest index is found, return the path that has 1
        # waypoint behind and X waypoints ahead, where X is the index
        # that has a lookahead distance specified by
        # INTERP_LOOKAHEAD_DISTANCE
        waypoint_subset_first_index = closest_index - 1
        if waypoint_subset_first_index < 0:
            waypoint_subset_first_index = 0

        waypoint_subset_last_index = closest_index
        total_distance_ahead = 0
        while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
            total_distance_ahead += wp_distance[waypoint_subset_last_index]
            waypoint_subset_last_index += 1
            if waypoint_subset_last_index >= waypoints_np.shape[0]:
                waypoint_subset_last_index = waypoints_np.shape[0] - 1
                break


        # Use the first and last waypoint subset indices into the hash
        # table to obtain the first and last indicies for the interpolated
        # list. Update the interpolated waypoints to the controller
        # for the next controller update.
        new_waypoints = \
                wp_interp[wp_interp_hash[waypoint_subset_first_index]:\
                            wp_interp_hash[waypoint_subset_last_index] + 1]
        
        # STEP 3.5: Obstacle perception (YOLO) and MPC_DLC signaling
        # NOTE: For MPC_DLC, we let the controller handle collision avoidance internally.
        #       The main loop keeps the centerline waypoints untouched and only passes
        #       obstacle information into the controller via set_obstacle_flag.
        obstacle_positions = []
        if args.control_method == 'MPC_DLC':
            # For MPC_DLC, always use the original (centerline) waypoints
            modified_waypoints = new_waypoints
        else:
            # For other controllers, we may still modify waypoints for avoidance
            modified_waypoints = new_waypoints.copy()
        
        # Only perform obstacle avoidance if enabled
        if enable_collision_avoidance and front_camera_detections is not None and cam_front_ref is not None and front_camera_image is not None:
            try:
                # Debug: Check if detections exist (YOLO boxes object)
                try:
                    num_detections = len(front_camera_detections.xyxy) if front_camera_detections is not None else 0
                except:
                    num_detections = 0
                if frame % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f"[Frame {frame}] YOLO detections: {num_detections} objects detected")
                
                # Extract obstacle positions from YOLO detections
                # IMPORTANT: use the actual YOLO image size (relit+resized), not raw CARLA resolution.
                if front_yolo_img_width is not None and front_yolo_img_height is not None:
                    image_width = front_yolo_img_width
                    image_height = front_yolo_img_height
                else:
                    # Fallback to sensor size if something went wrong
                    image_width = getattr(front_camera_image, "width", 800)
                    image_height = getattr(front_camera_image, "height", 600)
                obstacle_positions = extract_obstacle_positions(
                    front_camera_detections, cam_front_ref, image_width, image_height
                )
                
                # Debug: Log obstacle extraction
                if frame % 30 == 0:
                    print(f"[Frame {frame}] Extracted {len(obstacle_positions)} obstacle positions from {num_detections} detections")
                    if len(obstacle_positions) > 0:
                        for i, obs in enumerate(obstacle_positions):
                            print(f"  Obstacle {i}: ({obs.x:.2f}, {obs.y:.2f}, {obs.z:.2f})")
                
                # Determine if we have any obstacles
                has_obstacle = len(obstacle_positions) > 0

                # Inform controller whether an obstacle is present.
                # For MPC_DLC, we pass detailed obstacle info so the controller can
                # choose the DLC direction, but we do NOT modify waypoints here.
                if hasattr(controller, "set_obstacle_flag"):
                    if args.control_method == 'MPC_DLC':
                        # For MPC_DLC, pass obstacle positions to determine which side to shift
                        vehicle_location = vehicle.get_transform().location
                        # Calculate closest waypoint index in the waypoint subset
                        # This will be used to determine path heading for obstacle side detection
                        closest_idx_subset = 0
                        min_dist_subset = float("inf")
                        for i in range(len(new_waypoints)):
                            dist = math.sqrt(
                                (new_waypoints[i][0] - vehicle_location.x)**2 +
                                (new_waypoints[i][1] - vehicle_location.y)**2
                            )
                            if dist < min_dist_subset:
                                min_dist_subset = dist
                                closest_idx_subset = i
                        
                        controller.set_obstacle_flag(
                            has_obstacle,
                            obstacle_positions=obstacle_positions if has_obstacle else None,
                            vehicle_location=vehicle_location,
                            waypoints=new_waypoints,
                            closest_idx=closest_idx_subset
                        )
                    else:
                        # For other controllers, just pass the flag
                        controller.set_obstacle_flag(has_obstacle)

                # For non-MPC_DLC controllers, optionally modify waypoints to avoid obstacles.
                # For MPC_DLC, collision avoidance is done entirely inside the controller,
                # so we leave the waypoints unchanged.
                if has_obstacle and args.control_method != 'MPC_DLC':
                    vehicle_location = vehicle.get_transform().location
                    modified_waypoints = modify_waypoints_for_obstacles(
                        new_waypoints, obstacle_positions, vehicle_location, avoidance_distance=3.0
                    )
                    
                    # Debug: Count how many waypoints were actually modified
                    num_modified = 0
                    max_shift = 0.0
                    for i, wp in enumerate(modified_waypoints):
                        if i < len(new_waypoints):
                            orig_wp = new_waypoints[i]
                            shift = math.sqrt((wp[0] - orig_wp[0])**2 + (wp[1] - orig_wp[1])**2)
                            if shift > 0.1:
                                num_modified += 1
                                max_shift = max(max_shift, shift)
                    
                    if frame % 30 == 0:
                        print(f"[Frame {frame}] Modified {num_modified}/{len(modified_waypoints)} waypoints (max shift: {max_shift:.2f}m)")
                    
                    # # Visualize obstacle positions (red circles) - increased lifetime for visibility
                    # for obs_pos in obstacle_positions:
                    #     world.debug.draw_point(
                    #         obs_pos,
                    #         size=0.3,
                    #         color=carla.Color(r=255, g=0, b=0, a=255),
                    #         life_time=1.0,  # Increased from 0.1 to 1.0 seconds
                    #         persistent_lines=False
                    #     )
                    
                    # # Visualize modified waypoints (purple dots/circles)
                    # # Compare with original waypoints to show only modified ones
                    # for i, wp in enumerate(modified_waypoints):
                    #     wp_x = wp[0]
                    #     wp_y = wp[1]
                        
                    #     # Check if this waypoint was modified (compare with original)
                    #     if i < len(new_waypoints):
                    #         orig_wp = new_waypoints[i]
                    #         orig_x = orig_wp[0]
                    #         orig_y = orig_wp[1]
                            
                    #         # Only visualize if waypoint was actually modified
                    #         if abs(wp_x - orig_x) > 0.1 or abs(wp_y - orig_y) > 0.1:
                    #             wp_location = carla.Location(x=wp_x, y=wp_y, z=0.0)
                    #             world.debug.draw_point(
                    #                 wp_location,
                    #                 size=0.2,  # Increased from 0.15
                    #                 color=carla.Color(r=255, g=0, b=255, a=255),  # Purple, full opacity
                    #                 life_time=1.0,  # Increased from 0.1 to 1.0 seconds
                    #                 persistent_lines=False
                    #             )
                    #     else:
                    #         # New waypoint (shouldn't happen, but handle it)
                    #         wp_location = carla.Location(x=wp_x, y=wp_y, z=0.0)
                    #         world.debug.draw_point(
                    #             wp_location,
                    #             size=0.2,
                    #             color=carla.Color(r=255, g=0, b=255, a=255),  # Purple
                    #             life_time=1.0,
                    #             persistent_lines=False
                    #         )
                else:
                    if frame % 30 == 0:
                        print(f"[Frame {frame}] No obstacle positions extracted (detections: {num_detections})")
            except Exception as e:
                print(f"Error in obstacle avoidance: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to original waypoints on error
                modified_waypoints = new_waypoints
        else:
            # Debug: Log why avoidance is not running
            if frame % 60 == 0:  # Less frequent to avoid spam
                reasons = []
                if not enable_collision_avoidance:
                    reasons.append("collision_avoidance disabled")
                if front_camera_detections is None:
                    reasons.append("no detections")
                if cam_front_ref is None:
                    reasons.append("no camera ref")
                if front_camera_image is None:
                    reasons.append("no camera image")
                if reasons:
                    print(f"[Frame {frame}] Obstacle avoidance NOT running: {', '.join(reasons)}")
        
        controller.update_waypoints(modified_waypoints)

        # Time controller update
        def update_controller():
            nonlocal cmd_throttle, cmd_steer, cmd_brake, cmd_reverse
            # Update the other controller values and controls
            controller.update_values(current_x, current_y, current_yaw,
                                        current_speed,
                                        current_timestamp, frame, closest_distance)
            # print("update____")
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake, cmd_reverse = controller.get_commands()
        
        profiler.time_stage('controller_update', update_controller)

        # Skip the first frame (so the controller has proper outputs)
        if skip_first_frame and frame == 0:
            pass
        else:
            # Update live plotter with new feedback
            trajectory_fig.roll("trajectory", current_x, current_y)
            trajectory_fig.roll("car", current_x, current_y)
            # When plotting lookahead path, only plot a number of points
            # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
            # to decrease load when live plotting.
            # IMPORTANT: use modified_waypoints (after obstacle avoidance),
            # so the red "target" path visualization reflects avoidance behavior.
            new_waypoints_np = np.array(modified_waypoints)
            path_indices = np.floor(np.linspace(0,
                                                new_waypoints_np.shape[0]-1,
                                                INTERP_MAX_POINTS_PLOT))

            trajectory_fig.update("lookahead_path",
                    new_waypoints_np[path_indices.astype(int), 0],
                    new_waypoints_np[path_indices.astype(int), 1],
                    new_colour=[0, 0.7, 0.7])

            forward_speed_fig.roll("forward_speed",
                                    current_timestamp,
                                    current_speed*3.6) # m/s to km/h
            forward_speed_fig.roll("reference_signal",
                                    current_timestamp,
                                    controller._desired_speed*3.6) # m/s to km/h

            throttle_fig.roll("throttle", current_timestamp, cmd_throttle*100)
            brake_fig.roll("brake", current_timestamp, cmd_brake*100)
            steer_fig.roll("steer", current_timestamp, cmd_steer*180/np.pi)
            # Collision count over time
            collision_fig.roll("collision_count", current_timestamp, collision_count)

            # Refresh the live plot based on the refresh rate
            # set by the options
            if enable_live_plot and \
                live_plot_timer.has_exceeded_lap_period():
                plot_start = time.time()
                lp_traj.refresh()
                lp_1d.refresh()
                live_plot_timer.lap()
                profiler.record_timing('live_plotting', time.time() - plot_start)

        # Output controller command to CARLA server
        def send_control():
            send_control_command(vehicle,throttle=cmd_throttle,
                                    steer=cmd_steer,
                                    brake=cmd_brake,
                                    reverse=cmd_reverse)
        
        profiler.time_stage('control_command', send_control)
        
        # End frame timing and print stats periodically
        profiler.end_frame()
        profiler.print_stats()

        # Check if simulation timeout has been reached (after wait time)
        # If timeout reached, mark as failed and exit to skip to next test
        if current_timestamp >= SIMULATION_TIMEOUT:
            print(f"\n{'='*80}")
            print(f"SIMULATION TIMEOUT REACHED")
            print(f"{'='*80}")
            print(f"  Simulation time: {current_timestamp:.2f} seconds")
            print(f"  Timeout limit: {SIMULATION_TIMEOUT:.2f} seconds")
            print(f"  Did not complete trajectory - marking as failed")
            print(f"{'='*80}\n")
            # Mark as failed (reached_the_end remains False)
            # Record lap end time
            if lap_start_time is not None:
                lap_end_time = current_timestamp
            break
        
        # the simulation will end.
        dist_to_last_waypoint = np.linalg.norm(np.array([
            waypoints[-1][0] - current_x,
            waypoints[-1][1] - current_y]))
        # print("dist_to_last_waypoint:",dist_to_last_waypoint)
        if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
            reached_the_end = True
            # Record lap end time when reaching the end
            if lap_start_time is not None:
                lap_end_time = current_timestamp
            print("End")
            break
        # if reached_the_end:
            # break

    # End of demo - Stop vehicle and Store outputs to the controller output
    # directory.
    # Record lap end time if not already recorded (time limit reached)
    if lap_start_time is not None and lap_end_time is None:
        # Use the last recorded timestamp as end time
        # time_history contains timestamps after WAIT_TIME_BEFORE_START adjustment
        if len(time_history) > 0:
            lap_end_time = time_history[-1]
        elif 'current_timestamp' in locals():
            lap_end_time = current_timestamp
        else:
            # Fallback: use SIMULATION_TIMEOUT
            lap_end_time = SIMULATION_TIMEOUT
    
    if reached_the_end:
        print("Reached the end of path. Writing to controller_output...")
    else:
        print(f"Simulation timeout reached ({SIMULATION_TIMEOUT}s) or exceeded assessment time. Writing to controller_output...")
        print("Test marked as FAILED - did not complete trajectory")
    # Stop the car
    send_control_command(vehicle,throttle=0.0, steer=0.0, brake=1.0)
    
    # Calculate and report lap time
    print("=" * 80)
    print("LAP TIME STATISTICS:")
    if lap_start_time is not None and lap_end_time is not None:
        lap_time = lap_end_time - lap_start_time
        print(f"  Lap start time: {lap_start_time:.2f} s")
        print(f"  Lap end time: {lap_end_time:.2f} s")
        print(f"  Total lap time: {lap_time:.2f} s ({lap_time/60:.2f} min)")
        if reached_the_end:
            print(f"  Status: Completed trajectory successfully")
        else:
            print(f"  Status: Time limit reached (did not complete trajectory)")
        
        # Save lap time to file
        create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
        lap_time_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'lap_time.txt')
        with open(lap_time_file, 'w') as f:
            f.write(f"Lap Time Statistics\n")
            f.write(f"{'='*60}\n")
            f.write(f"Lap start time: {lap_start_time:.6f} s\n")
            f.write(f"Lap end time: {lap_end_time:.6f} s\n")
            f.write(f"Total lap time: {lap_time:.6f} s\n")
            f.write(f"Total lap time: {lap_time/60:.6f} min\n")
            f.write(f"Status: {'Completed trajectory' if reached_the_end else 'Time limit reached'}\n")
            f.write(f"Trajectory completion: {'Yes' if reached_the_end else 'No'}\n")
        print(f"  Lap time saved to: {lap_time_file}")
    else:
        print("  Warning: Could not calculate lap time")
        if lap_start_time is None:
            print("    Controller never started (stuck in wait period?)")
        if lap_end_time is None:
            print("    Lap end time not recorded")
    print("=" * 80)
    
    # Calculate collision percentage: total_crash_counts / total_cone_instance_number
    print("=" * 80)
    print("COLLISION STATISTICS:")
    if total_cone_count > 0:
        collision_percentage = (collision_count / total_cone_count) * 100.0
        print(f"  Total traffic cones spawned: {total_cone_count}")
        print(f"  Unique cones hit: {collision_count}")
        if len(hit_cone_ids) > 0:
            # Show cone IDs (limit to first 20 for readability)
            cone_ids_list = sorted(hit_cone_ids)
            if len(cone_ids_list) <= 20:
                print(f"  Unique cone IDs hit: {cone_ids_list}")
            else:
                print(f"  Unique cone IDs hit: {cone_ids_list[:20]} ... (and {len(cone_ids_list) - 20} more)")
        else:
            print(f"  Unique cone IDs hit: [] (no collisions)")
        print(f"  Collision percentage: {collision_percentage:.2f}%")
    else:
        print("  No traffic cones were spawned.")
        print(f"  Unique cones hit: {collision_count}")
    print("=" * 80)
    
    # Save collision statistics to file for batch testing
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    collision_stats_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_stats.txt')
    with open(collision_stats_file, 'w') as f:
        f.write(f"Collision Statistics\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total traffic cones spawned: {total_cone_count}\n")
        f.write(f"Unique cones hit: {collision_count}\n")
        f.write(f"Collision percentage: {(collision_count / total_cone_count * 100.0) if total_cone_count > 0 else 0.0:.2f}%\n")
        f.write(f"Reached end: {reached_the_end}\n")
        f.write(f"Simulation timeout: {SIMULATION_TIMEOUT} seconds\n")
        f.write(f"Status: {'Success' if reached_the_end else 'Failed - Timeout'}\n")
        f.write(f"Simulation timeout: {SIMULATION_TIMEOUT} seconds\n")
        f.write(f"Status: {'Success' if reached_the_end else 'Failed - Timeout'}\n")
    
    # Calculate path tracking error (lateral deviation from reference trajectory)
    print("=" * 80)
    print("PATH TRACKING ERROR STATISTICS:")
    if len(x_history) > 0 and waypoints_np.shape[0] > 0:
        path_error_stats = calculate_path_tracking_error(x_history, y_history, waypoints_np[:, :2])
        print(f"  Mean lateral deviation: {path_error_stats['mean']:.4f} m")
        print(f"  Std deviation: {path_error_stats['std']:.4f} m")
        print(f"  Maximum deviation: {path_error_stats['max']:.4f} m")
        print(f"  Minimum deviation: {path_error_stats['min']:.4f} m")
        print(f"  Total trajectory points: {len(x_history)}")
        print(f"  Reference waypoints: {waypoints_np.shape[0]}")
        
        # Save path tracking error to file for analysis
        create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
        error_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'path_tracking_error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Path Tracking Error Statistics\n")
            f.write(f"{'='*60}\n")
            f.write(f"Mean lateral deviation: {path_error_stats['mean']:.6f} m\n")
            f.write(f"Std deviation: {path_error_stats['std']:.6f} m\n")
            f.write(f"Maximum deviation: {path_error_stats['max']:.6f} m\n")
            f.write(f"Minimum deviation: {path_error_stats['min']:.6f} m\n")
            f.write(f"Total trajectory points: {len(x_history)}\n")
            f.write(f"Reference waypoints: {waypoints_np.shape[0]}\n")
            f.write(f"\nPer-point lateral deviations:\n")
            for i, error in enumerate(path_error_stats['all_errors']):
                f.write(f"{i:6d}: {error:.6f} m\n")
        print(f"  Path tracking error saved to: {error_file}")
    else:
        print("  Warning: Insufficient trajectory data to calculate path tracking error")
        print(f"    Trajectory points: {len(x_history)}, Waypoints: {waypoints_np.shape[0] if 'waypoints_np' in locals() else 0}")
    print("=" * 80)
    
    # Calculate speed tracking error (deviation from desired speed)
    print("=" * 80)
    print("SPEED TRACKING ERROR STATISTICS:")
    if len(speed_history) > 0 and len(desired_speed_history) > 0:
        speed_error_stats = calculate_speed_tracking_error(speed_history, desired_speed_history)
        print(f"  Mean speed error: {speed_error_stats['mean']:.4f} m/s ({speed_error_stats['mean']*3.6:.4f} km/h)")
        print(f"  Std deviation: {speed_error_stats['std']:.4f} m/s ({speed_error_stats['std']*3.6:.4f} km/h)")
        print(f"  RMSE: {speed_error_stats['rmse']:.4f} m/s ({speed_error_stats['rmse']*3.6:.4f} km/h)")
        print(f"  Maximum error: {speed_error_stats['max']:.4f} m/s ({speed_error_stats['max']*3.6:.4f} km/h)")
        print(f"  Minimum error: {speed_error_stats['min']:.4f} m/s ({speed_error_stats['min']*3.6:.4f} km/h)")
        print(f"  Total speed samples: {len(speed_history)}")
        
        # Save speed tracking error to file for analysis
        create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
        speed_error_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'speed_tracking_error.txt')
        with open(speed_error_file, 'w') as f:
            f.write(f"Speed Tracking Error Statistics\n")
            f.write(f"{'='*60}\n")
            f.write(f"Mean speed error: {speed_error_stats['mean']:.6f} m/s ({speed_error_stats['mean']*3.6:.6f} km/h)\n")
            f.write(f"Std deviation: {speed_error_stats['std']:.6f} m/s ({speed_error_stats['std']*3.6:.6f} km/h)\n")
            f.write(f"RMSE: {speed_error_stats['rmse']:.6f} m/s ({speed_error_stats['rmse']*3.6:.6f} km/h)\n")
            f.write(f"Maximum error: {speed_error_stats['max']:.6f} m/s ({speed_error_stats['max']*3.6:.6f} km/h)\n")
            f.write(f"Minimum error: {speed_error_stats['min']:.6f} m/s ({speed_error_stats['min']*3.6:.6f} km/h)\n")
            f.write(f"Total speed samples: {len(speed_history)}\n")
            f.write(f"\nPer-sample speed errors:\n")
            f.write(f"{'Index':>6} {'Actual (m/s)':>15} {'Desired (m/s)':>15} {'Error (m/s)':>15} {'Error (km/h)':>15}\n")
            for i in range(len(speed_history)):
                actual = speed_history[i]
                desired = desired_speed_history[i] if i < len(desired_speed_history) else 0
                error = abs(actual - desired)
                f.write(f"{i:6d} {actual:15.6f} {desired:15.6f} {error:15.6f} {error*3.6:15.6f}\n")
        print(f"  Speed tracking error saved to: {speed_error_file}")
    else:
        print("  Warning: Insufficient speed data to calculate speed tracking error")
        print(f"    Speed samples: {len(speed_history)}, Desired speed samples: {len(desired_speed_history) if 'desired_speed_history' in locals() else 0}")
    print("=" * 80)
    
    # Stop async relighting service if it's running
    if ENABLE_RELIGHTING and relighting_service is not None:
        if isinstance(relighting_service, AsyncRelighter):
            print("Stopping async relighting service...")
            relighting_service.stop()
            stats = relighting_service.stats()
            print(f"  Processed: {stats['processed']}, Dropped: {stats['dropped']}, "
                  f"Avg relight time (last 100): {stats['avg_time']*1000:.1f} ms")
    
    # Optimization C: No Tkinter window to close (removed)
    
    # Store the various outputs
    store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
    store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
    store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
    store_trajectory_plot(brake_fig.fig, 'brake_output.png')
    store_trajectory_plot(steer_fig.fig, 'steer_output.png')
    write_trajectory_file(x_history, y_history, speed_history, time_history)
    
    # Save tracking data with c_max to CSV in record mode
    if record_mode and len(tracking_data) > 0:
        try:
            create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
            csv_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'tracking_data_with_cmax.csv')
            
            # Write CSV with all tracking information and c_max
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['timestamp', 'frame', 'x', 'y', 'yaw', 'speed', 'desired_speed', 
                            'c_max', 'throttle', 'steer', 'brake']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in tracking_data:
                    writer.writerow(row)
            
            print(f"  Tracking data with c_max saved to: {csv_file}")
            print(f"  Total records: {len(tracking_data)}")
            print(f"  Records with c_max: {sum(1 for r in tracking_data if r['c_max'] is not None)}")
            print(f"  Records without c_max: {sum(1 for r in tracking_data if r['c_max'] is None)}")
            
            # Also save separate YOLO c_max data for reference (with original timestamps)
            if len(yolo_cmax_data) > 0:
                yolo_csv_file = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'yolo_cmax_data.csv')
                with open(yolo_csv_file, 'w', newline='') as f:
                    fieldnames = ['timestamp', 'frame', 'c_max', 'has_detections', 'num_detections']
                    if ENABLE_EVENT_DRIVEN:
                        fieldnames.extend(['relighting_triggered', 'is_relighting_active'])
                    # extrasaction='ignore': rows always carry the event-driven keys, which are
                    # only written as columns in event-driven mode
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for row in yolo_cmax_data:
                        writer.writerow(row)
                print(f"  YOLO c_max data (with original timestamps) saved to: {yolo_csv_file}")
                print(f"  Total YOLO detections: {len(yolo_cmax_data)}")
        except Exception as e:
            print(f"Warning: Could not save tracking data to CSV: {e}")
            import traceback
            traceback.print_exc()
    
    # Take screenshot if requested
    screenshot_path = getattr(args, 'screenshot_path', None)
    if screenshot_path is not None and camera_display is not None:
        try:
            pygame.image.save(camera_display, screenshot_path)
            print(f"Screenshot saved to: {screenshot_path}")
        except Exception as e:
            print(f"Warning: Could not save screenshot: {e}")
    
    # Collect results for batch testing
    collision_percentage = (collision_count / total_cone_count * 100.0) if total_cone_count > 0 else 0.0
    lap_time = lap_end_time - lap_start_time if (lap_start_time is not None and lap_end_time is not None) else None
    
    # Get error stats (may not exist if calculation failed)
    path_error_mean = None
    path_error_std = None
    speed_error_mean = None
    speed_error_rmse = None
    
    if len(x_history) > 0 and waypoints_np.shape[0] > 0:
        try:
            path_error_stats = calculate_path_tracking_error(x_history, y_history, waypoints_np[:, :2])
            path_error_mean = path_error_stats['mean']
            path_error_std = path_error_stats['std']
        except:
            pass
    
    if len(speed_history) > 0 and len(desired_speed_history) > 0:
        try:
            speed_error_stats = calculate_speed_tracking_error(speed_history, desired_speed_history)
            speed_error_mean = speed_error_stats['mean']
            speed_error_rmse = speed_error_stats['rmse']
        except:
            pass
    
    results = {
        'condition_id': getattr(args, 'condition_id', 'unknown'),
        'time_of_day': getattr(args, 'time_of_day', 10),
        'weather': getattr(args, 'weather_preset', 'clear'),
        'obstacle_probability': obstacle_probability,
        'collision_avoidance': getattr(args, 'enable_collision_avoidance', True),
        'relighting': getattr(args, 'enable_relighting', False),
        'reached_end': reached_the_end,
        'collision_count': collision_count,
        'total_cones': total_cone_count,
        'collision_percentage': collision_percentage,
        'lap_time': lap_time,
        'path_error_mean': path_error_mean,
        'path_error_std': path_error_std,
        'speed_error_mean': speed_error_mean,
        'speed_error_rmse': speed_error_rmse,
    }
    
    # Explicitly destroy CARLA actors/sensors before exit to prevent
    # "sensor object went out of scope" warnings and C++ crash (return code -6)
    try:
        # Destroy camera sensors first (order matters for CARLA)
        for cam in camera_sensors:
            try:
                if cam is not None and cam.is_alive:
                    cam.destroy()
            except Exception as e:
                print(f"Warning during camera sensor cleanup: {e}")
        camera_sensors.clear()
        # Destroy collision sensor
        if collision_sensor is not None and collision_sensor.is_alive:
            try:
                collision_sensor.destroy()
            except Exception as e:
                print(f"Warning during collision sensor cleanup: {e}")
            collision_sensor = None
        # Destroy spectator camera
        if camera is not None and camera.is_alive:
            try:
                camera.destroy()
            except Exception as e:
                print(f"Warning during camera cleanup: {e}")
        # Destroy obstacles (traffic cones)
        for obs in obstacles:
            try:
                if obs is not None and obs.is_alive:
                    obs.destroy()
            except Exception as e:
                print(f"Warning during obstacle cleanup: {e}")
        # Destroy vehicle last (destroys any remaining attached actors)
        if vehicle is not None and vehicle.is_alive:
            try:
                vehicle.destroy()
            except Exception as e:
                print(f"Warning during vehicle cleanup: {e}")
        # Brief pause to let CARLA server process destroy commands before process exit
        time.sleep(0.3)
    except Exception as e:
        print(f"Warning during CARLA cleanup: {e}")
    
    return results

def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '--control-method',
        metavar='CONTROL_METHOD',
        dest='control_method',
        choices = {'PurePursuit','Stanley','MPC', 'NMPC', 'MPC_DLC'},
        default='MPC',
        help='Select control method for Lane Keeping Assist : PurePursuit, Stanley, MPC, NMPC, or MPC_DLC (double lane change MPC)')
    argparser.add_argument(
        '--no-sync-mode',
        action='store_true',
        help='Disable synchronous mode (simulation may run at variable speed).')
    argparser.add_argument(
        '--fixed-delta-seconds',
        type=float,
        default=0.1,
        help='Simulation fixed time step in seconds when sync mode is enabled (e.g., 0.05 for ~20 FPS).')
    argparser.add_argument(
        '--enable-relighting',
        action='store_true',
        dest='enable_relighting',
        help='Enable DiT-based relighting for dark scenes (requires FLUX.2 Klein model).')
    argparser.add_argument(
        '--event-driven-relighting',
        action='store_true',
        dest='event_driven_relighting',
        help='Enable event-driven relighting with confidence-triggered activation. '
             'Requires --enable-relighting. Uses hysteresis: activates when c_max < tau_low, '
             'deactivates when c_max > tau_high.')
    argparser.add_argument(
        '--tau-low',
        type=float,
        default=0.3,
        dest='tau_low',
        help='Lower threshold for event-driven relighting activation (default: 0.3). '
             'Relighting activates when raw YOLO confidence drops below this value.')
    argparser.add_argument(
        '--tau-high',
        type=float,
        default=0.5,
        dest='tau_high',
        help='Upper threshold for event-driven relighting deactivation (default: 0.5). '
             'Relighting deactivates when raw YOLO confidence rises above this value.')
    argparser.add_argument(
        '--enable-clahe',
        action='store_true',
        dest='enable_clahe',
        help='Enable CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement. '
             'Typically used as an alternative to relighting for low-light conditions.')
    argparser.add_argument(
        '--relight-size-factor',
        type=float,
        default=4,
        dest='relight_size_factor',
        help='Image size reduction factor for relighting (default: 4, meaning 4x smaller). '
             'Higher values = smaller images = faster but lower quality. '
             'Examples: 2=512px, 4=256px, 8=128px (max dimension).')
    argparser.add_argument(
        '--relight-skip-frames',
        type=int,
        default=1,
        dest='relight_skip_frames',
        help='Process relighting every N frames (default: 1 = every frame). '
             'Higher values = faster but less frequent relighting. '
             'Example: 3 = relight every 3rd frame (3x faster).')
    argparser.add_argument(
        '--enable-profiling',
        action='store_true',
        dest='enable_profiling',
        help='Enable performance profiling to identify bottlenecks (reports every 100 frames).')
    argparser.add_argument(
        '--carla-quality-level',
        type=str,
        choices=['Low', 'Epic'],
        default='Epic',
        dest='carla_quality_level',
        help='CARLA graphics quality level: Low (faster, no shadows) or Epic (default, detailed).')
    argparser.add_argument(
        '--carla-no-rendering',
        action='store_true',
        dest='carla_no_rendering',
        help='Enable CARLA no-rendering mode (disables all graphics, cameras return empty data).')
    argparser.add_argument(
        '--async-relighting',
        action='store_true',
        dest='async_relighting',
        help='Enable async relighting (processes in background thread, eliminates blocking).')
    argparser.add_argument(
        '--base-model',
        type=str,
        default=DEFAULT_BASE_MODEL,
        dest='base_model',
        help=f'Base diffusion model for relighting (Hub id or local path). Default: {DEFAULT_BASE_MODEL}.')
    argparser.add_argument(
        '--lora',
        type=str,
        default=DEFAULT_LORA,
        dest='lora',
        help='Delight LoRA for relighting (local folder or Hub repo id with pytorch_lora_weights.safetensors). '
             'Default: <repo>/weights/delight_lora (the paper model).')
    argparser.add_argument(
        '--yolo-weights',
        type=str,
        default=DEFAULT_YOLO_WEIGHTS,
        dest='yolo_weights',
        help='YOLO detector weights. Default: <repo>/weights/yolo26_carla.pt (YOLO26 fine-tuned on CARLA cones).')
    argparser.add_argument(
        '--disable-collision-avoidance',
        action='store_false',
        dest='enable_collision_avoidance',
        default=True,
        help='Disable collision avoidance using YOLO obstacle detection (default: enabled).')
    argparser.add_argument(
        '--weather',
        type=str,
        choices=['clear', 'rainy', 'heavy_rain', 'foggy', 'foggy_rainy', 'cloudy'],
        default='clear',
        dest='weather_preset',
        help='Weather preset: clear (default), rainy, heavy_rain, foggy, foggy_rainy, or cloudy.')
    argparser.add_argument(
        '--precipitation',
        type=float,
        default=None,
        dest='precipitation',
        help='Custom precipitation level (0-100). Overrides weather preset if set.')
    argparser.add_argument(
        '--fog-density',
        type=float,
        default=None,
        dest='fog_density',
        help='Custom fog density (0-100). Overrides weather preset if set.')
    argparser.add_argument(
        '--fog-distance',
        type=float,
        default=None,
        dest='fog_distance',
        help='Custom fog distance in meters. Overrides weather preset if set.')
    argparser.add_argument(
        '--cloudiness',
        type=float,
        default=None,
        dest='cloudiness',
        help='Custom cloudiness level (0-100). Overrides weather preset if set.')
    argparser.add_argument(
        '--max-speed-kph',
        type=float,
        default=None,
        dest='max_speed_kph',
        help='Cap reference waypoint speed profile to this max speed (in km/h). '
             'Example: 120 -> ~33.3 m/s. Applies only to the waypoint speed column.',
    )
    argparser.add_argument(
        '--time-of-day',
        type=int,
        default=None,
        dest='time_of_day',
        help='Time of day in 24-hour format (10=day, 18=night). Default: 10 (day).')
    argparser.add_argument(
        '--obstacle-probability',
        type=float,
        default=0.00,
        dest='obstacle_probability',
        help='Probability of spawning obstacles per waypoint (0.0-1.0). Default: 0.00 (no obstacles).')
    argparser.add_argument(
        '--condition-id',
        type=str,
        default=None,
        dest='condition_id',
        help='Condition ID for batch testing (e.g., "Baseline1", "Dark2").')
    argparser.add_argument(
        '--screenshot-path',
        type=str,
        default=None,
        dest='screenshot_path',
        help='Path to save screenshot during test (e.g., "screenshots/test1.png").')
    argparser.add_argument(
        '--screenshot-progress',
        nargs='+',
        type=float,
        default=None,
        dest='screenshot_progress',
        help='Save screenshots automatically when track progress reaches these values. '
             'Accepts fractions (0.1) or percents (10 for 10%%). '
             'Example: --screenshot-progress 10 50 90')
    argparser.add_argument(
        '--screenshot-kind',
        type=str,
        choices=['annotated', 'raw', 'relit', 'both'],
        default='annotated',
        dest='screenshot_kind',
        help='Which image to save for progress screenshots: '
             '"annotated" (YOLO boxes), "raw" (sensor RGB), or "relit" (generative sensor output). '
             'Default: annotated.')
    argparser.add_argument(
        '--screenshot-sim-window',
        action='store_true',
        dest='screenshot_sim_window',
        help='Also save a screenshot of the CARLA simulator (CarlaUE4) OS window at the same progress triggers. '
             'Best-effort: requires a desktop session (DISPLAY) and external tools (xdotool+import preferred).')
    argparser.add_argument(
        '--no-exit-after-screenshots',
        action='store_true',
        dest='no_exit_after_screenshots',
        help='By default, if --screenshot-progress is set, the run will end early after capturing ALL requested '
             'progress screenshots (useful for paper figures). Set this flag to keep running the full lap.')
    argparser.add_argument(
        '--sim-window-name',
        type=str,
        default='CarlaUE4',
        dest='sim_window_name',
        help='Window name regex used to find the CARLA simulator window for screenshots (default: "CarlaUE4").')
    argparser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        dest='output_dir',
        help='Output directory for this test run (overrides CONTROLLER_OUTPUT_FOLDER).')
    argparser.add_argument(
        '--simulation-timeout',
        type=float,
        default=None,
        dest='simulation_timeout',
        help='Simulation time timeout in seconds. If lap not completed within this time, test will be marked as failed. Default: 300 seconds.')
    argparser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        dest='random_seed',
        help='Random seed for reproducible obstacle placement and paired tests.')
    argparser.add_argument(
        '--disable-live-plotting',
        action='store_true',
        dest='disable_live_plotting',
        help='Disable LivePlotter windows/refresh (recommended for batch runs).')
    argparser.add_argument(
        '--disable-visualization',
        action='store_true',
        dest='disable_visualization',
        help='Disable pygame visualization window (recommended for batch runs).')
    argparser.add_argument(
        '--mode',
        type=str,
        choices=['exp', 'record'],
        default='exp',
        dest='mode',
        help='Operation mode: "exp" for experiment mode (default), "record" for recording mode (saves tracking data with YOLO c_max to CSV).')
    args = argparser.parse_args()
    
    # Override output directory if specified
    if args.output_dir is not None:
        global CONTROLLER_OUTPUT_FOLDER
        CONTROLLER_OUTPUT_FOLDER = args.output_dir

    print(args)

    # Load the YOLO detector once (used by detect_obstacles)
    global model
    print(f"Loading YOLO detector from {args.yolo_weights}")
    model = YOLO(args.yolo_weights)

    ##Modifiable Variables from -1 to -5, as left to right
    # targetLane = -3
    targetLane = -3

    # Set global relighting flag from command-line argument
    global ENABLE_RELIGHTING, ENABLE_EVENT_DRIVEN, TAU_LOW, TAU_HIGH, is_relighting_active
    ENABLE_RELIGHTING = getattr(args, 'enable_relighting', False)
    ENABLE_EVENT_DRIVEN = getattr(args, 'event_driven_relighting', False)
    TAU_LOW = getattr(args, 'tau_low', 0.3)
    TAU_HIGH = getattr(args, 'tau_high', 0.5)
    
    # Validate event-driven mode
    if ENABLE_EVENT_DRIVEN and not ENABLE_RELIGHTING:
        print("=" * 60)
        print("WARNING: --event-driven-relighting requires --enable-relighting!")
        print("Event-driven mode will be DISABLED.")
        print("=" * 60)
        ENABLE_EVENT_DRIVEN = False
    
    # Initialize relighting state (starts inactive)
    is_relighting_active = False
    
    if ENABLE_RELIGHTING:
        print("=" * 60)
        if ENABLE_EVENT_DRIVEN:
            print("EVENT-DRIVEN RELIGHTING ENABLED")
            print(f"  Hysteresis thresholds: τ_low={TAU_LOW}, τ_high={TAU_HIGH}")
            print("  Relighting activates when raw c_max < τ_low")
            print("  Relighting deactivates when raw c_max > τ_high")
        else:
            print("CONTINUOUS RELIGHTING ENABLED via command-line argument")
        print("=" * 60)
    else:
        print("Relighting is DISABLED. Use --enable-relighting to enable.")
    
    # Set global CLAHE flag from command-line argument
    global ENABLE_CLAHE
    ENABLE_CLAHE = getattr(args, 'enable_clahe', False)
    if ENABLE_CLAHE:
        if CLAHE_AVAILABLE:
            print("=" * 60)
            print("CLAHE ENABLED via command-line argument")
            print("=" * 60)
        else:
            print("=" * 60)
            print("WARNING: CLAHE requested but OpenCV (cv2) is not available!")
            print("CLAHE will be DISABLED.")
            print("=" * 60)
            ENABLE_CLAHE = False
    else:
        print("CLAHE is DISABLED. Use --enable-clahe to enable.")

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
    import traceback
    
    # Execute when server connection is established
    while True:
        try:
            results = exec_waypoint_nav_demo(args)
            print(args)
            print('Done.')
            return results

        except Exception as e:
            print("break out with exception:", e)
            traceback.print_exc()
            # Important for batch testing: propagate failure to the calling process
            # so scripts can correctly mark this run as failed.
            sys.exit(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')