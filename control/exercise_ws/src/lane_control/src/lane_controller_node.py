#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, SegmentList, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        self.segs_msg = None
        self.last_omega = 0
        self.last_v = 0.2

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        # self.params['~K_pp'] = DTParam(
        #     '~K_pp',
        #     param_type=ParamType.FLOAT,
        #     min_value=0.0,
        #     max_value=5.0
        # )
        # self.params['~theta_thres'] = rospy.get_param('~theta_thres', None)

        self.pp_controller = PurePursuitLaneController(self.params)

        # Initialize variables
        # self.last_s = None

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)


        self.sub_lineseglist = rospy.Subscriber("/agent/lane_filter_node/seglist_filtered", #"/agent/lane_filter_node/seglist_filtered"
                                                SegmentList,
                                                self.cbSegList,
                                                queue_size=1)

        self.log("Initialized!")

    def cbSegList(self, inputsegs_msg):
        """Callback receiving segment list messages

        Args:
            input_segs (:obj:`SegmentList`): Message containing information about the filtered segments in the
            ground projection plane.
        """

        self.segs_msg = inputsegs_msg

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        print(self.segs_msg is not None)

        if self.segs_msg is not None:
            v, omega = self.pp_controller.compute_control_action_combined(self.segs_msg.segments, self.pose_msg,
                                                                self.last_v, self.last_omega)

        else:
            v, omega = self.pp_controller.compute_control_action_lp(self.pose_msg,
                                                                self.last_v, self.last_omega)

        self.last_v = v
        self.last_omega = omega


        # TODO This needs to get changed
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
