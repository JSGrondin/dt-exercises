import numpy as np
import math


class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    def compute_control_action_lp(self, pose, last_v, last_omega): #9332
        """ Given an filtered segments, computes pure pursuit control action (tuple of linear and angular
        velocity.

        Returns:
            v (:obj:`float`): requested linear velocity in meters/second
            omega (:obj:`float`): requested angular velocity in radians/second
        """

        phi = pose.phi
        d = pose.d

        v_max = 0.15  # we can make it constant or we can make it as a function of sin_alpha, velocity or robot
        k = 0.5

        if np.isnan(d) or np.isinf(d) or np.isnan(phi) or np.isinf(phi) or d == None or phi == None:
            omega = last_omega
            v = last_v

        else:
            # in big turns, we rely only on phi to steer, and on straight lines, we rectify position relative to middle
            v = max(0.05, v_max * np.cos(phi))
            L = max(0.1, k * v)
            if (np.abs(d) >= L):
                omega = last_omega
                v = last_v
            else:
                if -0.10 <= phi <= 0.10:
                    alpha = -phi - np.arcsin(d/L)
                else:
                    alpha = -phi

                sin_alpha = np.sin(alpha)
                omega = sin_alpha / k

        return v, omega

    def compute_control_action_fs(self, segments, pose, last_v, last_omega):
        """ Given an filtered segments, computes pure pursuit control action (tuple of linear and angular
        velocity.

        Returns:
            v (:obj:`float`): requested linear velocity in meters/second
            omega (:obj:`float`): requested angular velocity in radians/second
        """

        v_max = 0.3  # we can make it constant or we can make it as a function of sin_alpha, velocity or robot
        v_min = 0.05
        L_min = 0.04
        k = 0.25
        l_w = 0.12 #0.215/2 #0.185
        delta = 0.04
        accel=  0.08

        ref_traj_pt_list = self.segments_to_ref_traj(segments, l_w=l_w)

        if len(ref_traj_pt_list) >= 2:
            # get linear equation of reference trajectory
            a, b = self.lin_regression(ref_traj_pt_list)         # Y = a + bX

            # calculate angle of lane
            sigma = np.arctan(b)

            v = min(max(v_min, v_max * np.cos(sigma)), last_v+accel)
            L = max(L_min, k * v)

            fp_s, fp_m, fp_l = self.get_follow_points_on_ref_traj(ref_traj_pt_list, L=L, delta=delta)

            if len(fp_s) >= 3:
                fp, n = self.get_mean_fp(fp_s)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha = fp[1] / d_fp
                omega = (sin_alpha) / k

            elif len(fp_m) >= 3:
                fp, n = self.get_mean_fp(fp_m)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha = fp[1] / d_fp
                omega = (sin_alpha) / k

            elif len(fp_l) >= 3:
                fp, n = self.get_mean_fp(fp_l)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha = fp[1] / d_fp
                omega = (sin_alpha) / k

            else:
                fp, n = self.get_mean_fp(ref_traj_pt_list)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha = fp[1] / d_fp
                omega = (sin_alpha) / k

        else:
            omega = last_omega
            v = last_v

        return v, omega



    def compute_control_action_combined(self, segments, pose, last_v, last_omega):
        """ Given an filtered segments, computes pure pursuit control action (tuple of linear and angular
        velocity.

        Returns:
            v (:obj:`float`): requested linear velocity in meters/second
            omega (:obj:`float`): requested angular velocity in radians/second
        """

        fs_correct = False
        lp_correct = False

        v_max = 0.3  # we can make it constant or we can make it as a function of sin_alpha, velocity or robot
        v_min = 0.1
        L_min = 0.04
        k = 0.25
        l_w = 0.12 #0.215/2 #0.185
        delta = 0.04
        accel = 0.08

        conf_fs = 0.7

        phi = pose.phi
        d = pose.d

        ref_traj_pt_list = self.segments_to_ref_traj(segments, l_w=l_w)


        if len(ref_traj_pt_list) >= 2:
            fs_correct = True
            # get linear equation of reference trajectory
            a, b = self.lin_regression(ref_traj_pt_list)         # Y = a + bX

            # calculate angle of lane
            sigma = np.arctan(b)

            v = min(max(v_min, v_max * np.cos(sigma)), last_v+accel)
            L = max(L_min, k * v)

            fp_s, fp_m, fp_l = self.get_follow_points_on_ref_traj(ref_traj_pt_list, L=L, delta=delta)

            if len(fp_s) >= 3:
                fp, n = self.get_mean_fp(fp_s)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha_fs = fp[1] / d_fp

            elif len(fp_m) >= 3:
                fp, n = self.get_mean_fp(fp_m)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha_fs = fp[1] / d_fp

            elif len(fp_l) >= 3:
                fp, n = self.get_mean_fp(fp_l)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha_fs = fp[1] / d_fp

            else:
                fp, n = self.get_mean_fp(ref_traj_pt_list)
                d_fp = np.sqrt(fp[0] ** 2 + fp[1] ** 2)

                sin_alpha_fs = fp[1] / d_fp

        else:
            L = L_min
            v = last_v


        if not (np.isnan(d) or np.isinf(d) or np.isnan(phi) or np.isinf(phi) or d == None or phi == None or \
                (np.abs(d) >= L)):
            lp_correct = True
            if -0.10 <= phi <= 0.10:
                alpha = -phi - np.arcsin(d / L)
            else:
                alpha = -phi
            sin_alpha_lp = np.sin(alpha)

        if fs_correct and lp_correct:
            sin_alpha = (sin_alpha_fs * conf_fs) + ((1-conf_fs) * sin_alpha_lp)
            omega = (sin_alpha) / k
        elif lp_correct:
            omega = (sin_alpha_lp) / k
        elif fs_correct:
            omega = (sin_alpha_fs) / k
        else:
            omega = last_omega
            v = last_v

        return v, omega


    # def compute_follow_point(self, segments, lad, delta):
    #     """
    #     Find follow point (or look ahead point) in lane ahead of robot on reference path at a distance of more or less
    #      L, given the filtered segments)
    #     """
    #     # only consider segments that ± delta from look ahead distance
    #
    #     inlier_segments = self.get_inlier_segments(segments, lad, delta)
    #     yellow_x_mean = 0
    #     yellow_y_mean = 0
    #     yellow_n = 0
    #     white_x_mean = 0
    #     white_y_mean = 0
    #     white_n = 0
    #
    #     for segment in inlier_segments:
    #         if segment.color == segment.WHITE:
    #             white_x_mean += segment.points[0].x + segment.points[1].x
    #             white_y_mean += segment.points[0].y + segment.points[1].y
    #             white_n += 2
    #         else: # segment is yellow
    #             yellow_x_mean += segment.points[0].x + segment.points[1].x
    #             yellow_y_mean += segment.points[0].y + segment.points[1].y
    #             yellow_n += 2
    #
    #     if yellow_n != 0 and white_n != 0:
    #         yellow_mean_v = np.array([yellow_x_mean / yellow_n, yellow_y_mean / yellow_n])
    #         white_mean_v = np.array([white_x_mean / white_n, white_y_mean / white_n])
    #         follow_point = (yellow_mean_v + white_mean_v) / 2
    #         self.last_follow_point = follow_point
    #         d_fp = np.sqrt(follow_point[0] ** 2 + follow_point[1] ** 2)
    #         self.last_d_fp = d_fp
    #     else:
    #         follow_point = self.last_follow_point
    #         d_fp = self.last_d_fp
    #
    #     return follow_point, d_fp

    # def compute_follow_point(self, segments, lad, delta):
    #     """
    #     Find follow point (or look ahead point) in lane ahead of robot on reference path at a distance of more or less
    #      L, given the filtered segments)
    #     """
    #
    #     # lane_width
    #     l_w = 0.21
    #
    #     # get linear equation for yellow and white segments (y = a + bx)
    #     a_y, b_y, n_y, a_w, b_w, n_w = self.lin_regression(segments)
    #
    #     if n_w > 0 and n_y > 0:
    #         phi_y = np.arctan(1/b_y)
    #         if np.sign(phi_y) == -1:
    #             phi_y = phi_y + np.pi
    #         phi_w = np.arctan(1/b_w)
    #         if np.sign(phi_w) == -1:
    #             phi_w = phi_w + np.pi
    #
    #         # weighted average of where the middle of the lane is on the x axis
    #         x_middle = n_y*(-a_y/b_y - l_w/2/np.cos(phi_y)) + n_w(-a_w/b_w + l_w/2/np.cos(phi_w)) / (n_y + n_w)
    #         phi = (n_y * phi_y + n_w * phi_w) / (n_w + n_y)
    #
    #
    #
    #
    #     elif n_w > 0:
    #         phi_w = np.arctan(1/b_w)
    #         if np.sign(phi_w) == -1:
    #             phi_w = phi_w + np.pi
    #         x_middle = (-a_w / b_w + l_w / 2 / np.cos(phi_w))
    #         phi = phi_w
    #
    #
    #
    #     elif n_y > 0:
    #         phi_y = np.arctan(1/b_y)
    #         if np.sign(phi_y) == -1:
    #             phi_y = phi_y + np.pi
    #         x_middle = (-a_y / b_y + l_w / 2 / np.cos(phi_y))
    #         phi = phi_y
    #
    #
    #
    #     else:
    #         follow_point = self.last_follow_point
    #
    #
    #     if yellow_n != 0 and white_n != 0:
    #
    #     else:
    #         follow_point = self.last_follow_point
    #         d_fp = self.last_d_fp
    #
    #     return follow_point, d_fp
    #
    #
    # def compute_alpha(self, segments, pose, lad, delta):
    #     """
    #     Find follow point (or look ahead point) in lane ahead of robot on reference path at a distance of more or less
    #      L, given the filtered segments)
    #     """
    #     phi = pose.phi
    #     d = pose.d
    #
    #     will_cause_error = (d >= lad) or np.isnan(d) or np.isinf(d) or d == None
    #
    #     # in big turns, we rely only on phi to steer, and on straight lines, we rectify position relative to middle
    #     if -0.06 <= phi <= 0.06 and not will_cause_error:
    #         alpha = -phi - np.arcsin(d/lad)
    #     else:
    #         alpha = -phi
    #
    #     return alpha
    #
    # def get_inlier_segments(self, segments, lad, delta):
    #     inlier_segments = []
    #     for segment in segments:
    #         d_s = self.get_distance_sgmt(segment)
    #         if lad + delta >= d_s >= lad - delta:
    #             inlier_segments.append(segment)
    #     return inlier_segments

    def get_follow_points_on_ref_traj(self, ref_traj_pt_list, L, delta):
        follow_points_small = []
        follow_points_medium = []
        follow_points_large = []
        for tup in ref_traj_pt_list:
            d_s = self.get_distance_tuple(tup)
            if L + delta >= d_s >= L - delta:
                follow_points_small.append(tup)
            if L + 2*delta >= d_s >= L - 2*delta:
                follow_points_medium.append(tup)
            if L + 3*delta >= d_s >= L - 3*delta:
                follow_points_large.append(tup)
        return follow_points_small, follow_points_medium, follow_points_large


    def get_distance_tuple(self, ref_pt):
        d_s = np.sqrt(ref_pt[0] ** 2 + ref_pt[1] ** 2)
        return d_s

    def get_mean_fp(self, follow_points):
        x=0
        y=0
        n=0
        for tup in follow_points:
            x += tup[0]
            y += tup[1]
            n += 1
        return (x, y), n



    # def get_distance_sgmt(self, segment):
    #     p1 = np.array([segment.points[0].x, segment.points[0].y])
    #     p2 = np.array([segment.points[1].x, segment.points[1].y])
    #     p_center = (p1 + p2) / 2
    #
    #     # d1 = np.sqrt(p1[0] ** 2 + p1[1] **2)
    #     # d2 = np.sqrt(p2[0] ** 2 + p2[1] ** 2)
    #     d_s = np.sqrt(p_center[0] ** 2 + p_center[1] ** 2)
    #     return d_s
    #
    def lin_regression(self, tup_list):
        # Y = a + bX

        sum_x=0
        sum_y=0
        sum_xy=0
        sum_x2=0
        n=0

        for tup in tup_list:
            sum_x += tup[0]
            sum_y += tup[1]
            sum_xy += tup[0]*tup[1]
            sum_x2 += tup[0]**2
            n += 1

        a = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)
        b = (n * sum_xy - sum_x*sum_y) / (n * sum_x2 - sum_x**2)

        return a, b

    # def segment_line_eq(self, segment):
    #
    #     x1 = segment.points[0].x
    #     x2 = segment.points[1].x
    #     y1 = segment.points[0].y
    #     y2 = segment.points[1].y
    #
    #     # get value of slope
    #     m = (x2 - x1) / (y2 - y1)
    #
    #     # given that x = my + b, we can use x1 and y1 in that equation to find b
    #     b = x1 - (m * y1)
    #
    #     return m, b

    def get_segment_centroid(self, segment):
        x = (segment.points[0].x + segment.points[1].x) / 2
        y = (segment.points[0].y + segment.points[1].y) / 2

        return x, y

    def get_segment_ortho_vec(self, segment):
        u1 = segment.points[1].x - segment.points[0].x
        u2 = segment.points[1].y - segment.points[0].y

        if segment.color == segment.WHITE:
            if np.sign(u1) == np.sign(u2):
                v1 = -1
                v2 = u1/u2
            else:
                v1 = 1
                v2 = -u1/u2
        else: # segment is yellow
            if np.sign(u1) == np.sign(u2):
                v1 = 1
                v2 = -u1/u2
            else:
                v1 = -1
                v2 = u1/u2
        # we want unit vector
        ed = np.sqrt(v1 ** 2 + v2 ** 2)

        return v1/ed, v2/ed

    def segments_to_ref_traj(self, segments, l_w):
        ref_traj_pt_list = []
        for segment in segments:
        # get centroid
            cent_x, cent_y = self.get_segment_centroid(segment)

        # get orthogonal unit vector
            ortho_unit_vec_x, ortho_unit_vec_y = self.get_segment_ortho_vec(segment)

        # get pt on reference trajectory using centroid and unit vector
            ref_x = cent_x + ortho_unit_vec_x * l_w
            ref_y = cent_y + ortho_unit_vec_y * l_w

        # append point to list
            ref_traj_pt_list.append((ref_x, ref_y))
        return ref_traj_pt_list




