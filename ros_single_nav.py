#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile, ReliabilityPolicy

import zmq

# TF
import tf2_ros
from tf_transformations import quaternion_matrix, translation_matrix

import message_filters

import numpy as np
from argparse import Namespace
from geometry_msgs.msg import Twist

import threading
import open3d as o3d
import open3d.visualization.gui as gui
from multiprocessing import Queue

from arguments import get_args
from agents.ros2_single_agent import ROS_Agent
from utils.vis_gui import ReconstructionWindow


# QoS
qos_profile_reliable = QoSProfile(depth=2)
qos_profile_reliable.reliability = ReliabilityPolicy.RELIABLE

# ZMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://192.168.100.1:5557")
print("socket connected(?) to port 5557")


def get_pose(trans, rot):
    trans_mat = translation_matrix([trans.x, trans.y, trans.z])
    rot_mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    transform_mat = np.dot(trans_mat, rot_mat)

    # lidar->camera extrinsics (as in your file)
    trans_mat = translation_matrix([0.20626980772421397,
                                    -0.00528269584842609,
                                    -0.015826777380052434])
    rot_mat = quaternion_matrix([0.5130067729487394,
                                 -0.5014588509249872,
                                 0.4940995610242004,
                                 -0.49115037975492243])
    T_lidar_camera = np.dot(trans_mat, rot_mat)
    return transform_mat @ T_lidar_camera


class FspNode(Node):
    def __init__(self, args, send_queue, receive_queue):
        super().__init__('fsp_node')

        self.args = args
        self.send_queue = send_queue
        self.receive_queue = receive_queue

        # Agent
        self.agent = ROS_Agent(self.args, 0, send_queue, receive_queue)

        # Obs buffer
        self.obs = {}

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1200.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # message_filters sync
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/robot1/camera/color/image_raw', qos_profile=qos_profile_reliable
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/robot1/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile_reliable
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.rgbd_callback)

        # CameraInfo
        self.create_subscription(
            CameraInfo,
            '/robot1/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers (optional; keep for monitoring)
        self.action_pub = self.create_publisher(Int32, '/robot_action', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Timestamp for TF lookup
        self.global_timestamp = None

        self.get_logger().info("FspNode init done.")

    def rgbd_callback(self, rgb_msg, depth_msg):
        h = rgb_msg.height
        w = rgb_msg.width

        rs_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(h, w, 3)
        self.obs['rgb'] = rs_rgb

        rs_depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w)
        self.obs['depth'] = rs_depth

        self.global_timestamp = depth_msg.header.stamp

    def camera_info_callback(self, msg):
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        camera_matrix = {'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy}
        self.obs['cam_K'] = Namespace(**camera_matrix)

    def timer_callback(self):
        if 'rgb' not in self.obs or 'depth' not in self.obs or 'cam_K' not in self.obs:
            return
        if self.global_timestamp is None:
            return

        # TF lookup
        try:
            transform = self.tf_buffer.lookup_transform(
                "camera_init_1",
                "body_1",
                self.global_timestamp,
                timeout=rclpy.duration.Duration(seconds=0.0)
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        trans = transform.transform.translation
        rot = transform.transform.rotation
        self.obs['pose'] = get_pose(trans, rot)

        # Agent step (your ROS_Agent returns discrete action 1/2/3)
        action = self.agent.mapping(self.obs)

        # Drive via ZMQ (as you already do)
        if self.agent.l_step < 35:
            socket.send(b"speedctl speed|0.0|0.0|0.5")
        elif action == 1:
            socket.send(b"speedctl speed|0.4|0.0|0.0")
        elif action == 2:
            socket.send(b"speedctl speed|0.0|0.0|0.5")
        elif action == 3:
            socket.send(b"speedctl speed|0.0|0.0|-0.5")

        # Publish action for monitoring
        msg = Int32()
        msg.data = int(action)
        self.action_pub.publish(msg)

        # Also publish a consistent cmd_vel for monitoring / alternative control
        vel = Twist()
        if self.agent.l_step < 35 or action == 2:
            vel.angular.z = 0.5
        elif action == 3:
            vel.angular.z = -0.5
        elif action == 1:
            vel.linear.x = 0.4
        self.cmd_vel_pub.publish(vel)

        self.get_logger().info(f"Action published: {action}")


def visualization_thread(args, send_queue, receive_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()


def main():
    rclpy.init()
    args = get_args()

    send_queue = Queue()
    receive_queue = Queue()

    visualization = threading.Thread(
        target=visualization_thread,
        args=(args, send_queue, receive_queue)
    )
    visualization.start()

    node = FspNode(args, send_queue, receive_queue)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    visualization.join()
    print("you successfully navigated to destination point")


if __name__ == "__main__":
    main()
