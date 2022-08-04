# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motor model for laikago."""
import numpy as np
import collections
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class LaikagoMotorModel(object):
	"""A simple motor model for Laikago.

	When in POSITION mode, the torque is calculated according to the difference
	between current and desired joint angle, as well as the joint velocity.
	For more information about PD control, please refer to:
	https://en.wikipedia.org/wiki/PID_controller.

	The model supports a HYBRID mode in which each motor command can be a tuple
	(desired_motor_angle, position_gain, desired_motor_velocity, velocity_gain,
	torque).

	"""

	def __init__(self,
				num_motors,
				kp,
				kd,
				torque_limits=None,
				motor_control_mode="Position"):
		self._num_motors = num_motors
		# Default kp and kd values
		self._kp = kp
		self._kd = kd
		self._torque_limits = torque_limits
		if torque_limits is not None:
			if isinstance(torque_limits, (collections.Sequence, np.ndarray)):
				self._torque_limits = np.asarray(torque_limits)
			else:
				self._torque_limits = np.full(self._num_motors, torque_limits)
		self._motor_control_mode = motor_control_mode
		self._strength_ratios = np.full(self._num_motors, 1)

	def set_strength_ratios(self, ratios):
		"""Set the strength of each motors relative to the default value.

		Args:
		ratios: The relative strength of motor output. A numpy array ranging from
		0.0 to 1.0.
		"""
		self._strength_ratios = ratios

	def set_motor_gains(self, kp, kd):
		"""Set the gains of all motors.

		These gains are PD gains for motor positional control. kp is the
		proportional gain and kd is the derivative gain.

		Args:
		kp: proportional gain of the motors.
		kd: derivative gain of the motors.
		"""
		self._kp = kp
		self._kd = kd

	def set_voltage(self, voltage):
		pass

	def get_voltage(self):
		return 0.0

	def set_viscous_damping(self, viscous_damping):
		pass

	def get_viscous_dampling(self):
		return 0.0

	def convert_to_torque(self,
						motor_commands,
						motor_angle,
						motor_velocity,
						true_motor_velocity,
						motor_control_mode):
		"""Convert the commands (position control or torque control) to torque.

		Args:
			motor_commands: The desired motor angle if the motor is in position
				control mode. The pwm signal if the motor is in torque control mode.
			motor_angle: The motor angle observed at the current time step. It is
				actually the true motor angle observed a few milliseconds ago (pd
				latency).
			motor_velocity: The motor velocity observed at the current time step, it
				is actually the true motor velocity a few milliseconds ago (pd latency).
			true_motor_velocity: The true motor velocity. The true velocity is used to
				compute back EMF voltage and viscous damping.
			motor_control_mode: A MotorControlMode enum.

		Returns:
			actual_torque: The torque that needs to be applied to the motor.
			observed_torque: The torque observed by the sensor.
		"""
		del true_motor_velocity
		if not motor_control_mode:
			motor_control_mode = self._motor_control_mode
		
		assert len(motor_commands) == self._num_motors, "The length of the command is inconsistent with the action_space."
		assert len(self._kp) != 0 and len(self._kd) != 0, "Values for kp and kd are missing for the motor."

		# No processing for motor torques
		if motor_control_mode != "Torque":
			kp = self._kp
			kd = self._kd
			desired_motor_angles = motor_commands
			desired_motor_velocities = np.full(self._num_motors, 0)
			motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (motor_velocity - desired_motor_velocities)
		else:
			motor_torques = motor_commands
		
		# Cap the max/min torque applied
		if self._torque_limits is not None:
			if len(self._torque_limits) != len(motor_torques):
				raise ValueError("Torque limits dimension does not match the number of motors.")
			motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits, self._torque_limits)

		observed_torque = motor_torques
		actual_torque = self._strength_ratios * motor_torques

		return actual_torque, observed_torque
