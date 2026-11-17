#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
import math
import sys
import os
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class Controller2D(object):
    def __init__(self, waypoints, control_method, world):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._set_reverse        = False
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        # self._kP                 = 0.7        Stanley 36kph
        # self._kI                 = 0.05
        # self._kD                 = 0.1  not used in calculate_throttle

        # self._kP                 = 3.0   Stanley 99kph(no _kD)
        # self._kI                 = 0.2
        # self._kD                 = 0.2

        # self._kP                 = 0.95
        # self._kI                 = 0.01
        # self._kD                 = 0.05
        # self._kB                 = 0.2

        self._kP                 = 1.2
        self._kI                 = 0.05
        self._kD                 = 0.15
        self._kB                 = 0.25

        self._Kpp                = 4.5
        self._Kvf                = 1.0
        # self._Kcte               = 0.7
        self._Kcte               = 0.5
        # self._Kmpc               = 0.3
        # self._Kmpc               = 0.55
        self._Kmpc               = 0.7

        self._eps_lookahead      = 10e-3
        self._closest_distance   = 0
        self._wheelbase          = 2.4689
        self._control_method     = control_method
        # self._steering_diff      = np.linspace(-10, 10, 21, endpoint = True)
        # self._steering_diff      = np.linspace(-5, 5, 21, endpoint = True)
        # self._steering_diff      = np.linspace(-7, 7, 31, endpoint = True)
        self._steering_diff      = np.linspace(-15, 15, 31, endpoint = True)
        self.world               = world
        self._desired_speed_mult = 0.9

        # Lane geometry / DLC parameters
        # self._lane_width         = 1.2 # approximate lane width (meters)
        # self._lane_width         = 2.5 # approximate lane width (meters)
        self._lane_width   = 1.0
        # Simple state flag for DLC (used by MPC_DLC)
        self._dlc_active         = False
        # DLC offset direction: +1 = shift left, -1 = shift right, 0 = center
        self._dlc_offset_sign    = 0     # Direction of lateral shift

        # NMPC-specific parameters (aggressive lateral control, aligned with MPC _Kmpc=0.55, ±15°)
        self._nmpc_horizon_steps = 15       # Reduced horizon to prevent early turning
        self._nmpc_dt = 0.05               # Time step (match sim: 0.05 or 0.1)
        self._nmpc_min_lookahead = 2.0    # Minimum lookahead distance (m) before starting horizon
        self._nmpc_max_steer = np.deg2rad(50)  # Allow sharper steering (MPC uses ±15°)
        self._nmpc_w_cte = 20.0             # Strong path tracking (aggressive)
        self._nmpc_w_heading = 20.0         # Fast heading alignment
        self._nmpc_w_steer = 0.15          # Low penalty on steering (allow large corrections)
        self._nmpc_w_steer_rate = 0.15     # Low smoothness penalty (faster steering response)

    def update_values(self, x, y, yaw, speed, timestamp, frame, closest_distance):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        self._closest_distance  = closest_distance
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed * self._desired_speed_mult -  1

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints


    def set_obstacle_flag(self, has_obstacle: bool, obstacle_positions=None, vehicle_location=None, waypoints=None, closest_idx=None):
        """
        Set obstacle flag and determine which side to shift to.
        
        Args:
            has_obstacle: Whether obstacle is detected
            obstacle_positions: List of carla.Location objects (optional, for determining shift direction)
            vehicle_location: Current vehicle location (optional)
            waypoints: Current waypoints (optional)
            closest_idx: Closest waypoint index (optional, will be calculated if not provided)
        """
        if has_obstacle:
            self._dlc_active = True
            # Fill the timer (e.g., 40 frames = ~2 seconds to pass the obstacle)
            if hasattr(self.vars, 'dlc_hold_frames'):
                self.vars.dlc_hold_frames = 40
            # Cache obstacle positions for internal DLC planning
            try:
                if obstacle_positions:
                    self.vars.dlc_obstacles = [(obs_pos.x, obs_pos.y) for obs_pos in obstacle_positions]
                else:
                    self.vars.dlc_obstacles = []
            except Exception:
                self.vars.dlc_obstacles = []
            
            # Determine which side to shift to based on obstacle positions
            if (obstacle_positions and len(obstacle_positions) > 0 and 
                vehicle_location and waypoints and len(waypoints) > 0):
                
                # Calculate closest_idx if not provided
                if closest_idx is None or closest_idx >= len(waypoints):
                    closest_idx = 0
                    min_dist = float("inf")
                    for i in range(len(waypoints)):
                        dist = self.get_distance(
                            vehicle_location.x, vehicle_location.y,
                            waypoints[i][0], waypoints[i][1]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                
                if closest_idx < len(waypoints):
                    try:
                        # Calculate average obstacle position relative to path
                        # Get path heading at closest waypoint
                        if closest_idx < len(waypoints) - 1:
                            wp_curr = waypoints[closest_idx]
                            wp_next = waypoints[closest_idx + 1]
                            dx = wp_next[0] - wp_curr[0]
                            dy = wp_next[1] - wp_curr[1]
                            path_yaw = np.arctan2(dy, dx)
                            
                            # Calculate perpendicular vector (left side of path)
                            perp_x = -np.sin(path_yaw)  # Left side
                            perp_y = np.cos(path_yaw)
                            
                            # For each obstacle, calculate lateral offset from path
                            left_obstacles = 0
                            right_obstacles = 0
                            
                            for obs_pos in obstacle_positions:
                                # Vector from waypoint to obstacle
                                obs_dx = obs_pos.x - wp_curr[0]
                                obs_dy = obs_pos.y - wp_curr[1]
                                
                                # Project onto perpendicular direction to get lateral offset
                                lateral_offset = obs_dx * perp_x + obs_dy * perp_y
                                
                                if lateral_offset > 0:
                                    left_obstacles += 1
                                else:
                                    right_obstacles += 1
                            
                            # Choose opposite side: if more obstacles on left, shift right (negative)
                            if left_obstacles >= right_obstacles:
                                self._dlc_offset_sign = -1  # Shift right
                            else:
                                self._dlc_offset_sign = 1   # Shift left
                        else:
                            # Default: shift right if we can't determine
                            self._dlc_offset_sign = -1
                    except Exception as e:
                        # If calculation fails, default to shift right
                        print(f"Warning: Could not determine DLC shift direction: {e}")
                        self._dlc_offset_sign = -1
            else:
                # Default: shift right if no obstacle position info
                self._dlc_offset_sign = -1
        else:
            # No obstacle: prepare to return to center
            # Don't immediately deactivate - let timer handle it
            # Also clear cached obstacle list
            if hasattr(self.vars, 'dlc_obstacles'):
                self.vars.dlc_obstacles = []

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake, self._set_reverse

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def calculate_throttle(self, t, v, v_desired):
        # Using PID Controller
        time_step = t - self.vars.t_previous
        
        # Avoid division by zero
        if time_step < 1e-6:
            time_step = 0.01  # Default to 10ms if time step is too small
        
        speed_error = v_desired - v
        
        # Proportional term
        p_term = self._kP * speed_error
        
        # Integral term (with anti-windup protection)
        i_term = self.vars.i_term_previous + self._kI * time_step * speed_error
        # Clamp integral term to prevent windup
        i_term = np.clip(i_term, -1.0, 1.0)
        self.vars.i_term_previous = i_term
        
        # Derivative term (FIX: proper derivative calculation)
        # Need to store previous error for derivative calculation
        if not hasattr(self.vars, 'speed_error_previous'):
            self.vars.speed_error_previous = 0.0
        error_derivative = (speed_error - self.vars.speed_error_previous) / time_step
        d_term = self._kD * error_derivative
        self.vars.speed_error_previous = speed_error
        
        u = p_term + i_term + d_term
        if u>=0:
            Tp = np.clip(u, 0.0, 1.0)  # Clamp throttle to [0, 1]
            Bp = 0.0
        else:
            Tp = 0.0
            Bp = np.clip((-1)*u*self._kB, 0.0, 1.0)  # Clamp brake to [0, 1]
        return Tp, Bp

    def get_shifted_coordinate(self, x, y, yaw, length):
        x_shifted = x + length*np.cos(yaw)
        y_shifted = y + length*np.sin(yaw)
        # print(x,y,yaw,x_shifted,y_shifted)
        return x_shifted, y_shifted

    def get_lookahead_dis(self, v):
        return self._Kvf*v

    def get_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def get_goal_waypoint_index(self, x, y, waypoints, lookahead_dis):
        for i in range(len(waypoints)):
            dis = self.get_distance(x, y, waypoints[i][0], waypoints[i][1])
            if abs(dis - lookahead_dis) <= self._eps_lookahead:
                return i
        return len(waypoints)-1

    def get_alpha(self, v1, v2, lookahead_dis):
        inner_prod = v1[0]*v2[0] + v1[1]*v2[1]
        return np.arccos(inner_prod/lookahead_dis)

    def get_steering_direction(self, v1, v2):
        corss_prod = v1[0]*v2[1] - v1[1]*v2[0]
        if corss_prod >= 0 :
            return -1
        return 1

    def get_heading_error(self, waypoints, current_yaw):
        waypoint_delta_x = waypoints[1][0] - waypoints[0][0]
        waypoint_delta_y = waypoints[1][1] - waypoints[0][1]
        waypoint_heading = np.arctan(waypoint_delta_y/waypoint_delta_x)
        heading_error_mod = divmod((waypoint_heading - current_yaw), np.pi)[1]
        if heading_error_mod > np.pi/2 and heading_error_mod < np.pi:
            heading_error_mod -= np.pi
        return heading_error_mod

    def get_cte_heading_error(self, v):
        Cerror = 0.001
        proportional_cte_error = self._Kcte * self._closest_distance
        cte_heading_error = np.arctan(proportional_cte_error/(v+Cerror))
        cte_heading_error_mod = divmod(cte_heading_error, np.pi)[1]
        if cte_heading_error_mod > np.pi/2 and cte_heading_error_mod < np.pi:
            cte_heading_error_mod -= np.pi
        return cte_heading_error_mod

    def get_predicted_wheel_location(self, x, y, steering_angle, yaw, v):
        wheel_heading = yaw + steering_angle
        wheel_traveled_dis = v * (self._current_timestamp - self.vars.t_previous)
        return [x + wheel_traveled_dis * np.cos(wheel_heading), y + wheel_traveled_dis * np.sin(wheel_heading)]

    def _get_dlc_waypoint(self, waypoints, index, offset):
        """
        Shift a waypoint laterally by 'offset' meters using a perpendicular
        direction to the PATH heading, not the vehicle yaw.
        This prevents feedback loops where steering changes the target direction.
        
        Args:
            waypoints: List of waypoints
            index: Index of the waypoint to shift
            offset: Lateral offset in meters (positive = shift left)
        
        Returns:
            [new_x, new_y]: Shifted waypoint coordinates
        """
        wp_curr = waypoints[index]
        # Get next waypoint to calculate path direction
        next_idx = min(index + 1, len(waypoints) - 1)
        wp_next = waypoints[next_idx]
        
        # If we are at the end of the path, fallback to 0 yaw
        if index == next_idx:
            path_yaw = 0.0
        else:
            dx = wp_next[0] - wp_curr[0]
            dy = wp_next[1] - wp_curr[1]
            path_yaw = np.arctan2(dy, dx)
        
        # Perpendicular to path heading (path_yaw + 90 degrees)
        perp_yaw = path_yaw + self._pi / 2.0
        new_x = wp_curr[0] + offset * np.cos(perp_yaw)
        new_y = wp_curr[1] + offset * np.sin(perp_yaw)
        return [new_x, new_y]

    def _dlc_plan_offsets_astar(self, waypoints, base_indices, horizon_steps):
        """
        Tiny dynamic-programming / A*-like planner over discrete lateral offsets
        for the next horizon_steps, to bias the reference waypoints away from
        predicted obstacle locations while keeping the path smooth.

        Offsets are chosen from {-lane_width, 0, +lane_width}.
        """
        # If we don't have obstacle info, just keep using the current offset.
        if not hasattr(self.vars, 'dlc_obstacles') or not self.vars.dlc_obstacles:
            return None  # caller will fall back to uniform offset

        offsets = [-self._lane_width, 0.0, self._lane_width]
        num_offsets = len(offsets)
        big = 1e9

        # DP tables: cost[step][offset_idx], parent[step][offset_idx]
        cost = [[big] * num_offsets for _ in range(horizon_steps)]
        parent = [[-1] * num_offsets for _ in range(horizon_steps)]

        # Safety radius around obstacles
        safety_radius = 3.0
        safety_radius_sq = safety_radius * safety_radius

        # Initialize step 0
        ref_idx0 = base_indices[0]
        for oi, off in enumerate(offsets):
            shifted = self._get_dlc_waypoint(waypoints, ref_idx0, off)
            x0, y0 = shifted[0], shifted[1]

            # Distance penalty to nearest obstacle
            min_obs_dist_sq = big
            for ox, oy in self.vars.dlc_obstacles:
                d_sq = (x0 - ox) ** 2 + (y0 - oy) ** 2
                if d_sq < min_obs_dist_sq:
                    min_obs_dist_sq = d_sq

            obs_cost = 0.0
            if min_obs_dist_sq < safety_radius_sq:
                # High cost when inside safety radius
                obs_cost = 1e4 * (safety_radius_sq - min_obs_dist_sq)

            # Small lateral cost to prefer centerline when possible
            lateral_cost = 0.5 * (off ** 2)

            cost[0][oi] = obs_cost + lateral_cost
            parent[0][oi] = -1

        # Propagate through horizon
        for step in range(1, horizon_steps):
            ref_idx = base_indices[step]
            for oi, off in enumerate(offsets):
                shifted = self._get_dlc_waypoint(waypoints, ref_idx, off)
                xk, yk = shifted[0], shifted[1]

                # Distance penalty to nearest obstacle
                min_obs_dist_sq = big
                for ox, oy in self.vars.dlc_obstacles:
                    d_sq = (xk - ox) ** 2 + (yk - oy) ** 2
                    if d_sq < min_obs_dist_sq:
                        min_obs_dist_sq = d_sq

                obs_cost = 0.0
                if min_obs_dist_sq < safety_radius_sq:
                    obs_cost = 1e4 * (safety_radius_sq - min_obs_dist_sq)

                # Transition from previous offsets
                best_prev_cost = big
                best_prev_idx = -1
                for p_oi, p_off in enumerate(offsets):
                    base_c = cost[step - 1][p_oi]
                    if base_c >= big:
                        continue
                    # Penalize large lateral jumps between steps
                    jump = off - p_off
                    jump_cost = 2.0 * (jump ** 2)
                    c = base_c + obs_cost + jump_cost
                    if c < best_prev_cost:
                        best_prev_cost = c
                        best_prev_idx = p_oi

                cost[step][oi] = best_prev_cost
                parent[step][oi] = best_prev_idx

        # Backtrack best sequence at final step
        last_step = horizon_steps - 1
        best_final_cost = big
        best_final_idx = -1
        for oi in range(num_offsets):
            if cost[last_step][oi] < best_final_cost:
                best_final_cost = cost[last_step][oi]
                best_final_idx = oi

        if best_final_idx == -1:
            return None

        best_offsets = [0.0] * horizon_steps
        idx = best_final_idx
        for step in range(last_step, -1, -1):
            best_offsets[step] = offsets[idx]
            idx = parent[step][idx]
            if idx < 0 and step > 0:
                # Fallback: zero offset for remaining previous steps
                for k in range(step - 1, -1, -1):
                    best_offsets[k] = 0.0
                break

        return best_offsets

    def _nmpc_bicycle_model_step(self, x, y, yaw, v, steer, dt, L):
        """Kinematic bicycle model: one integration step.
        Standard formula: yaw_new = yaw + (v/L) * tan(steer) * dt
        Note: We negate steer here because the optimizer works in inverted convention,
        then we negate the output to match CARLA's expected convention.
        """
        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        # Negate steering for internal optimization (will be negated again at output)
        yaw_new = yaw + (v / L) * np.tan(-steer) * dt
        # Normalize yaw to [-pi, pi]
        yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))
        return x_new, y_new, yaw_new

    def _nmpc_get_reference_path(self, waypoints, start_idx, distances):
        """Get reference (x, y, yaw) at given distances along path from start_idx."""
        if start_idx >= len(waypoints) - 1:
            wp = waypoints[-1]
            return [(wp[0], wp[1], 0.0)] * len(distances)
        refs = []
        for target_dist in distances:
            cum_dist = 0.0
            idx = start_idx
            found = False
            while idx < len(waypoints) - 1:
                dx = waypoints[idx + 1][0] - waypoints[idx][0]
                dy = waypoints[idx + 1][1] - waypoints[idx][1]
                seg_len = np.sqrt(dx**2 + dy**2)
                if seg_len < 1e-9:
                    idx += 1
                    continue
                if cum_dist + seg_len >= target_dist:
                    t = (target_dist - cum_dist) / seg_len
                    rx = waypoints[idx][0] + t * dx
                    ry = waypoints[idx][1] + t * dy
                    ryaw = np.arctan2(dy, dx)
                    refs.append((rx, ry, ryaw))
                    found = True
                    break
                cum_dist += seg_len
                idx += 1
            if not found:
                wp = waypoints[-1]
                ryaw = refs[-1][2] if refs else np.arctan2(
                    waypoints[-1][1] - waypoints[-2][1],
                    waypoints[-1][0] - waypoints[-2][0]) if len(waypoints) >= 2 else 0.0
                refs.append((wp[0], wp[1], ryaw))
        return refs

    def _nmpc_cost(self, steer, x, y, yaw, v, ref_points, dt, L, prev_steer):
        """NMPC cost function: path deviation + control effort + smoothness."""
        try:
            steer = float(np.asarray(steer).flat[0])
        except (IndexError, TypeError):
            steer = float(steer)
        px, py, pyaw = x, y, yaw
        cost = 0.0
        for k, (ref_x, ref_y, ref_yaw) in enumerate(ref_points):
            px, py, pyaw = self._nmpc_bicycle_model_step(px, py, pyaw, v, steer, dt, L)
            cte_sq = (px - ref_x)**2 + (py - ref_y)**2
            # Heading error: difference between predicted yaw and reference yaw
            # In CARLA left-handed: if path curves left, ref_yaw is negative, vehicle needs to turn left (negative steer)
            heading_err = pyaw - ref_yaw
            heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
            # Use absolute value for heading error cost (direction doesn't matter, only magnitude)
            cost += self._nmpc_w_cte * cte_sq + self._nmpc_w_heading * heading_err**2
        cost += self._nmpc_w_steer * steer**2
        cost += self._nmpc_w_steer_rate * (steer - prev_steer)**2
        return cost

    def _calculate_mpc_steering(self, x, y, yaw, waypoints, v):
        """
        Original MPC heuristic: point-shooting to a lookahead waypoint.
        Used by both the plain 'MPC' mode and as a fallback for 'MPC_DLC'
        when no lane change is active, so that the vehicle can complete
        a lap with the proven-stable behavior.
        """
        world = self.world

        # Find closest waypoint
        closest_idx = 0
        min_dist = float("inf")
        for i in range(len(waypoints)):
            dist = self.get_distance(x, y, waypoints[i][0], waypoints[i][1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Calculate lookahead waypoint based on closest point + horizon
        remaining_waypoints = len(waypoints) - closest_idx
        horizon_idx = closest_idx + int(self._Kmpc * remaining_waypoints)
        horizon_idx = min(horizon_idx, len(waypoints) - 1)  # Ensure within bounds

        last_waypoint = [waypoints[horizon_idx][0], waypoints[horizon_idx][1]]

        # draw target waypoint
        world.debug.draw_string(
            carla.Location(x=last_waypoint[0], y=last_waypoint[1]),
            'O',
            draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0, a=20),
            life_time=0.01,
            persistent_lines=True
        )
        # draw vehicle position
        world.debug.draw_string(
            carla.Location(x=x, y=y),
            'O',
            draw_shadow=False,
            color=carla.Color(r=0, g=255, b=0, a=20),
            life_time=0.01,
            persistent_lines=True
        )

        steering_list = self.vars.steering_previous + self._steering_diff * self._pi / 180
        min_dis = float("inf")
        steering = self.vars.steering_previous

        for i in range(len(steering_list)):
            predicted_wheel_location = self.get_predicted_wheel_location(
                x, y, steering_list[i], yaw, v
            )
            dis_to_last_waypoint = self.get_distance(
                predicted_wheel_location[0],
                predicted_wheel_location[1],
                last_waypoint[0],
                last_waypoint[1],
            )
            if dis_to_last_waypoint < min_dis:
                min_dis = dis_to_last_waypoint
                steering = steering_list[i]
        return steering

    def calculate_steering(self, x, y, yaw, waypoints, v):
        if self._control_method == 'PurePursuit':
            lookahead_dis = self.get_lookahead_dis(v)
            print(lookahead_dis)
            idx = self.get_goal_waypoint_index(x, y, waypoints, lookahead_dis)
            world = self.world
            # draw target
            world.debug.draw_string(carla.Location(x=waypoints[idx][0],y = waypoints[idx][1]), 'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0, a =20), life_time=0.001,
                                    persistent_lines=True)
            # draw vehicle position
            world.debug.draw_string(carla.Location(x,y), 'O', draw_shadow=False,
                                    color=carla.Color(r=0, g=255, b=0, a =20), life_time=0.001,
                                    persistent_lines=True)

            v1 = [waypoints[idx][0] - x, waypoints[idx][1] - y]
            v2 = [np.cos(yaw), np.sin(yaw)]
            alpha = self.get_alpha(v1, v2, lookahead_dis)
            if math.isnan(alpha):
                alpha = self.vars.alpha_previous
            if not math.isnan(alpha):
                self.vars.alpha_previous = alpha
            steering = self.get_steering_direction(v1, v2)*np.arctan((2*self._wheelbase*np.sin(alpha))/(self._Kpp*v))
            if math.isnan(steering):
                steering = self.vars.steering_previous
            if not math.isnan(steering):
                self.vars.steering_previous = steering
            return steering
        elif self._control_method == 'Stanley':
            world = self.world
            # draw target
            world.debug.draw_string(carla.Location(x=waypoints[300][0],y = waypoints[300][1]), 'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0, a =20), life_time=0.001,
                                    persistent_lines=True)
            # draw vehicle position
            world.debug.draw_string(carla.Location(x,y), 'O', draw_shadow=False,
                                    color=carla.Color(r=0, g=255, b=0, a =20), life_time=0.001,
                                    persistent_lines=True)

            v1 = [waypoints[300][0] - x, waypoints[300][1] - y]
            v2 = [np.cos(yaw), np.sin(yaw)]
            heading_error = self.get_heading_error(waypoints, yaw)
            cte_error = self.get_steering_direction(v1, v2)*self.get_cte_heading_error(v)
            steering =  heading_error + cte_error
            return steering
        elif self._control_method == 'MPC':
            # Use the shared MPC heuristic implementation
            return self._calculate_mpc_steering(x, y, yaw, waypoints, v)

        elif self._control_method == 'MPC_DLC':
            world = self.world

            # If no lane change is active and we are effectively on the centerline,
            # fall back to the proven MPC steering so we can reliably finish a lap.
            # This gives you a baseline identical to MPC when there are no obstacles,
            # while still enabling DLC behavior once obstacles trigger it.
            if (not self._dlc_active) and (abs(self.vars.current_offset) < 1e-3):
                return self._calculate_mpc_steering(x, y, yaw, waypoints, v)

            # 1) Find closest waypoint
            closest_idx = 0
            min_dist = float("inf")
            # Use cached closest_idx as starting point for efficiency
            start_idx = getattr(self.vars, 'closest_idx', 0)
            search_range = 100
            start_search = max(0, start_idx - search_range)
            end_search = min(len(waypoints), start_idx + search_range)
            
            for i in range(start_search, end_search):
                dist = self.get_distance(x, y, waypoints[i][0], waypoints[i][1])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Store for use by obstacle detection logic
            self.vars.closest_idx = closest_idx

            # Snap to zero if close enough, to enable the stable fallback
            if not self._dlc_active and abs(self.vars.current_offset) < 0.05:
                self.vars.current_offset = 0.0

            # --- NEW: Tick down the hold timer ---
            if self.vars.dlc_hold_frames > 0:
                self.vars.dlc_hold_frames -= 1
            else:
                self._dlc_active = False  # Timer expired, safe to return to center!
                self._dlc_offset_sign = 0  # Reset offset direction

            # 2) Smooth lateral offset transition
            # Calculate target offset with direction
            if self._dlc_active:
                target_offset = self._lane_width * self._dlc_offset_sign
            else:
                target_offset = 0.0
            
            # Use adaptive smoothing: faster when activating, slower when returning to center
            if abs(target_offset) > abs(self.vars.current_offset):
                # Activating DLC: faster response
                smoothing_factor = 0.1
            else:
                # Returning to center: smoother transition
                smoothing_factor = 0.05
                # smoothing_factor = 0.3
            
            self.vars.current_offset += (target_offset - self.vars.current_offset) * smoothing_factor
            active_offset = self.vars.current_offset

            # 3) Setup horizon variables
            # Detect if we're in a large corner by checking path curvature ahead
            # Look ahead ~20-30m to see if path is curving significantly
            corner_detection_lookahead = min(30, len(waypoints) - closest_idx)
            is_large_corner = False
            if corner_detection_lookahead >= 5:
                # Check heading change over next 10-15 waypoints
                check_range = min(15, corner_detection_lookahead)
                if closest_idx + check_range < len(waypoints):
                    wp_start = waypoints[closest_idx]
                    wp_end = waypoints[closest_idx + check_range]
                    heading_start = np.arctan2(
                        waypoints[min(closest_idx + 1, len(waypoints) - 1)][1] - wp_start[1],
                        waypoints[min(closest_idx + 1, len(waypoints) - 1)][0] - wp_start[0]
                    )
                    heading_end = np.arctan2(
                        wp_end[1] - waypoints[closest_idx + check_range - 1][1],
                        wp_end[0] - waypoints[closest_idx + check_range - 1][0]
                    )
                    heading_change = abs(np.arctan2(np.sin(heading_end - heading_start), np.cos(heading_end - heading_start)))
                    # If heading changes more than 30 degrees over ~15 waypoints, it's a large corner
                    is_large_corner = heading_change > np.deg2rad(30)
            
            # Adaptive horizon: longer in corners for better prediction
            if is_large_corner:
                horizon_steps = 12  # Longer horizon for corners
            else:
                horizon_steps = 10  # Increased from 5 to 10 for better stability
            
            dt = self._nmpc_dt
            v_safe = max(v, 0.5)

            # --- OPTIMIZATION: Pre-calculate the reference path ONCE ---
            # First compute the base waypoint indices along the centerline for each
            # step in the horizon, using v_safe and dt (speed-dependent lookahead).
            # Increase minimum lookahead in corners for better visibility
            min_lookahead = 3.0 if is_large_corner else 2.0
            base_indices = []
            closest_wp_x = waypoints[closest_idx][0]
            closest_wp_y = waypoints[closest_idx][1]
            for step in range(horizon_steps):
                # In corners, use slightly longer lookahead to see further ahead
                lookahead_mult = 1.2 if is_large_corner else 1.0
                lookahead_dist = max(min_lookahead, v_safe * dt * (step + 1) * lookahead_mult)
                ref_idx = closest_idx
                for j in range(closest_idx, len(waypoints)):
                    if self.get_distance(closest_wp_x, closest_wp_y, waypoints[j][0], waypoints[j][1]) >= lookahead_dist:
                        ref_idx = j
                        break
                ref_idx = min(ref_idx, len(waypoints) - 1)
                base_indices.append(ref_idx)

            # Optionally use a tiny A*-like planner over lateral offsets to bias
            # the reference waypoints away from obstacles when they are present.
            planned_offsets = None
            if hasattr(self.vars, 'dlc_obstacles') and self.vars.dlc_obstacles:
                planned_offsets = self._dlc_plan_offsets_astar(
                    waypoints, base_indices, horizon_steps
                )

            ref_waypoints = []
            for step in range(horizon_steps):
                ref_idx = base_indices[step]
                if planned_offsets is not None and step < len(planned_offsets):
                    step_offset = planned_offsets[step]
                else:
                    step_offset = active_offset
                ref_waypoints.append(self._get_dlc_waypoint(waypoints, ref_idx, step_offset))

            # 4) Evaluate steering candidates
            # Allow a wider steering search window when DLC is active, in large corners,
            # or we are still significantly offset from the centerline
            if self._dlc_active or abs(active_offset) > 0.3 or is_large_corner:
                steering_span_deg = 40.0# wider window for DLC / large offset / corners
                steering_samples = 61
            else:
                steering_span_deg = 15.0  # original MPC window
                steering_samples = len(self._steering_diff)

            dlc_steering_diff = np.linspace(-steering_span_deg, steering_span_deg, steering_samples, endpoint=True)
            steering_candidates = self.vars.steering_previous + dlc_steering_diff * self._pi / 180.0
            best_cost = float("inf")
            best_steer = self.vars.steering_previous

            # Get center waypoints for comparison (when no obstacle)
            center_ref_waypoints = []
            if abs(active_offset) < 0.1:  # Near center, use center waypoints
                for step in range(horizon_steps):
                    lookahead_dist = max(2.0, v_safe * dt * (step + 1))
                    ref_idx = closest_idx
                    for j in range(closest_idx, len(waypoints)):
                        if self.get_distance(closest_wp_x, closest_wp_y, waypoints[j][0], waypoints[j][1]) >= lookahead_dist:
                            ref_idx = j
                            break
                    ref_idx = min(ref_idx, len(waypoints) - 1)
                    center_ref_waypoints.append([waypoints[ref_idx][0], waypoints[ref_idx][1]])

            # Choose cost weights depending on whether we are in DLC / large-offset
            # mode, in a large corner, or cruising near the centerline.
            # In DLC/corner mode we relax steering penalties so the controller can use larger angles if needed.
            if self._dlc_active or abs(active_offset) > 0.3 or is_large_corner:
                # Lower penalties in corners/DLC to allow stronger steering
                steering_effort_w = 0.05  # Increased from 0.01 for smoother steering
                steering_rate_w = 0.2     # Increased from 0.05 to penalize rapid changes
            else:
                # Nominal cruising penalties - highly prioritize smoothness
                steering_effort_w = 0.1
                steering_rate_w = 0.5

            for steer_cand in steering_candidates:
                cost = 0.0
                px, py, pyaw = x, y, yaw

                for step in range(horizon_steps):
                    # Predict state
                    px, py, pyaw = self._nmpc_bicycle_model_step(px, py, pyaw, v, steer_cand, dt, self._wheelbase)
                    
                    # Fetch pre-calculated reference point
                    ref_wp = ref_waypoints[step]

                    # Distance cost to shifted waypoint (when avoiding obstacle)
                    dist_sq = (px - ref_wp[0]) ** 2 + (py - ref_wp[1]) ** 2
                    cost += dist_sq
                    
                    # Additional penalty for deviation from center when no obstacle
                    if abs(active_offset) < 0.1 and step < len(center_ref_waypoints):
                        center_wp = center_ref_waypoints[step]
                        center_dist_sq = (px - center_wp[0]) ** 2 + (py - center_wp[1]) ** 2
                        # Strong penalty for deviating from center when no obstacle
                        # cost += 2.0 * center_dist_sq
                        cost += 1.5 * center_dist_sq
                    
                    # Heading alignment cost (penalize heading error)
                    # Stronger weight in corners for better path following
                    if step < len(ref_waypoints) - 1:
                        # Calculate desired heading from reference path
                        ref_next = ref_waypoints[min(step + 1, len(ref_waypoints) - 1)]
                        desired_yaw = np.arctan2(ref_next[1] - ref_wp[1], ref_next[0] - ref_wp[0])
                        heading_error = pyaw - desired_yaw
                        # Normalize to [-pi, pi]
                        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                        # Stronger heading alignment in corners
                        heading_weight = 1.0 if is_large_corner else 0.5
                        cost += heading_weight * (heading_error ** 2)

                # Steering effort regularization
                cost += steering_effort_w * (steer_cand ** 2)
                
                # Steering rate penalty (smoothness)
                steer_rate = abs(steer_cand - self.vars.steering_previous)
                cost += steering_rate_w * (steer_rate ** 2)

                if cost < best_cost:
                    best_cost = cost
                    best_steer = steer_cand

            # Draw the full shifted reference path (purple lines when avoiding, green when on center)
            # path_color = carla.Color(r=128, g=0, b=128, a=20) if abs(active_offset) > 0.1 else carla.Color(r=0, g=255, b=0, a=20)
            path_color = carla.Color(r=128, g=0, b=128, a=1) if abs(active_offset) > 0.1 else carla.Color(r=255, g=255, b=255, a=1)
            for step in range(len(ref_waypoints) - 1):
                p1 = ref_waypoints[step]
                p2 = ref_waypoints[step + 1]
                
                # Elevate the Z coordinate slightly (z=0.5) so it doesn't clip into the road
                world.debug.draw_line(
                    carla.Location(x=p1[0], y=p1[1], z=15),
                    carla.Location(x=p2[0], y=p2[1], z=15),
                    thickness=0.2,
                    color=path_color,
                    life_time=0.15
                )
            
            # Draw status indicator
            status_text = f"DLC: {'ON' if self._dlc_active else 'OFF'} | Offset: {active_offset:.2f}m"
            world.debug.draw_string(
                carla.Location(x=x, y=y, z=2.0),
                status_text,
                draw_shadow=True,
                color=carla.Color(r=255, g=255, b=0, a=255),
                life_time=0.15,
                persistent_lines=False
            )
            
            # Apply low-pass filter to output steering for smoother control
            target_steer = -best_steer
            alpha = 0.6  # 60% new value, 40% historical value
            smoothed_steer = alpha * target_steer + (1.0 - alpha) * self.vars.steering_previous
            
            return smoothed_steer

        elif self._control_method == 'NMPC':
            # --- 1. Find Closest Waypoint Index ---
            min_dist = float("inf")
            curr_closest_idx = getattr(self.vars, 'closest_idx', 0)
            search_range = 100
            start_search = max(0, curr_closest_idx - search_range)
            end_search = min(len(waypoints), curr_closest_idx + search_range)
            for i in range(start_search, end_search):
                dist = self.get_distance(x, y, waypoints[i][0], waypoints[i][1])
                if dist < min_dist:
                    min_dist = dist
                    curr_closest_idx = i
            self.vars.closest_idx = curr_closest_idx

            # --- 2. NMPC: Fixed-distance reference horizon (avoids shrinking at track end) ---
            dt = self._nmpc_dt
            if self._start_control_loop and hasattr(self.vars, 't_previous'):
                ts = self._current_timestamp - self.vars.t_previous
                if 0.01 <= ts <= 0.2:
                    dt = ts
            N = self._nmpc_horizon_steps
            L = self._wheelbase
            v_safe = max(v, 0.5)  # Avoid div by zero
            # Reference distances: min_lookahead + v*dt, min_lookahead + 2*v*dt, ..., min_lookahead + N*v*dt
            # Add minimum lookahead to prevent premature turning
            distances = [self._nmpc_min_lookahead + v_safe * dt * k for k in range(1, N + 1)]
            ref_points = self._nmpc_get_reference_path(waypoints, curr_closest_idx, distances)

            prev_steer = self.vars.steering_previous

            if SCIPY_AVAILABLE:
                # --- 3. Numerical optimization (true NMPC) ---
                def cost_fn(steer):
                    return self._nmpc_cost(steer, x, y, yaw, v_safe, ref_points, dt, L, prev_steer)
                res = minimize(
                    cost_fn,
                    x0=[prev_steer],
                    method='L-BFGS-B',
                    bounds=[(-self._nmpc_max_steer, self._nmpc_max_steer)],
                    options={'maxiter': 50}
                )
                best_steering = float(res.x[0]) if res.success else prev_steer
            else:
                # --- Fallback: discrete search (heuristic) ---
                best_steering = prev_steer
                min_cost = float("inf")
                for steer_cand in self._steering_diff * np.pi / 180:
                    c = self._nmpc_cost(steer_cand, x, y, yaw, v_safe, ref_points, dt, L, prev_steer)
                    if c < min_cost:
                        min_cost = c
                        best_steering = steer_cand

            # CARLA steering convention: need to negate to match PurePursuit behavior
            # PurePursuit uses get_steering_direction which handles sign, NMPC needs manual negation
            return -best_steering
        else:
            return 0

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.
            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)
            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0
            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous',0.0)
        self.vars.create_var('i_term_previous',0.0)
        self.vars.create_var('alpha_previous',0.0)
        self.vars.create_var('steering_previous',0.0)
        self.vars.create_var('speed_error_previous', 0.0)
        self.vars.create_var('closest_idx', 0)  # <--- ADD THIS LINE
        self.vars.create_var('current_offset', 0.0)  # For smooth DLC offset transition
        self.vars.create_var('dlc_hold_frames', 0) # For DLC hold counter
        self.vars.create_var('dlc_offset_sign', 0)  # Direction of lateral shift (-1=right, +1=left, 0=center)
        self.vars.create_var('stuck_timer', 0)
        self.vars.create_var('recovery_timer', 0)
        
        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.
                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]:
                                          Returns the 3rd waypoint's y position
                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """

            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.

            throttle_output, brake_output = self.calculate_throttle(t, v, v_desired)

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            # print("Goal point idx =",idx)
            # Change the steer output with the lateral controller.
            steer_output    = self.calculate_steering(x, y, yaw, waypoints, v)

            # ------------------------------------------------------------------
            # STUCK DETECTION AND RECOVERY LOGIC (Added for Wall Recovery)
            # ------------------------------------------------------------------
            # Check if stuck: low speed but attempting to move (throttle > 0.1)
            # Threshold: < 0.5 m/s speed while throttle > 10%
            if v < 0.5 and throttle_output > 0.1:
                self.vars.stuck_timer += 1
            else:
                # Slowly decay stuck timer so intermittent stops don't trigger it
                self.vars.stuck_timer = max(0, self.vars.stuck_timer - 1)
            
            # Trigger recovery if stuck for ~2 seconds (assuming 20fps -> 40 frames)
            if self.vars.stuck_timer > 40:
                self.vars.recovery_timer = 60  # Reverse for ~3 seconds
                self.vars.stuck_timer = 0
            
            # Execute recovery behavior
            reverse_cmd = False
            if self.vars.recovery_timer > 0:
                self.vars.recovery_timer -= 1
                reverse_cmd = True
                throttle_output = 0.5
                brake_output = 0.0
                # Steer straight while reversing to pull away from obstacle
                steer_output = 0.0
                # Reset PID terms to prevent windup during recovery
                self.vars.i_term_previous = 0.0
                self.vars.speed_error_previous = 0.0

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1 to 1)
            self.set_brake(brake_output)        # in percent (0 to 1)
            self._set_reverse = reverse_cmd     # Set reverse gear state

            self.vars.steering_previous = steer_output  # For NMPC/MPC/Stanley next iteration

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.t_previous = t