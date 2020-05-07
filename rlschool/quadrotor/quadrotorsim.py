import json
import numpy as np

PI = 3.1415926
GRAVITY_ACCELERATION = 9.80


class QuadrotorSim(object):
    def __init__(self, config_file=None):
        # State
        self._zero_state()

        # Other helper vars
        self._is_configured = False
        self._precision = 1e-3
        self._coordination_converter_to_world = self.rotation_matrix
        self._coordination_converter_to_body = np.linalg.inv(
            self.rotation_matrix)

    def _zero_state(self):
        self.global_position = np.array([0.0] * 3).astype(np.float32)
        self.global_velocity = np.array([0.0] * 3).astype(np.float32)
        self.body_acceleration = np.array([0.0] * 3).astype(np.float32)
        self.body_angular_velocity = np.array([0.0] * 3).astype(np.float32)
        self.propeller_angular_velocity = np.array([0.0] * 4).astype(
            np.float32)
        self.rotation_matrix = np.eye(3).astype(np.float32)
        self.power = 0.0

    def _save_state(self):
        current_position = np.copy(self.global_position)
        current_velocity = np.copy(self.global_velocity)
        current_body_acc = np.copy(self.body_acceleration)
        current_body_w = np.copy(self.body_angular_velocity)
        current_prop_w = np.copy(self.propeller_angular_velocity)
        current_rot_mat = np.copy(self.rotation_matrix)
        current_power = self.power

        return [current_position, current_velocity, current_body_acc,
                current_body_w, current_prop_w, current_rot_mat, current_power]

    def _restore_state(self, state):
        self.global_position, self.global_velocity, self.body_acceleration, \
            self.body_angular_velocity, self.propeller_angular_velocity, \
            self.rotation_matrix, self.power = state
        self._coordination_converter_to_world = self.rotation_matrix
        self._coordination_converter_to_body = np.linalg.inv(
            self.rotation_matrix)

    def _parse_cfg(self):
        self._precision = float(self.cfg['precision'])
        self._quality = float(self.cfg['quality'])

        self._inertia = np.zeros((3, 3)).astype(np.float32)
        self._inertia[0, 0] = float(self.cfg['inertia']['xx'])
        self._inertia[0, 1] = float(self.cfg['inertia']['xy'])
        self._inertia[0, 2] = float(self.cfg['inertia']['xz'])
        self._inertia[1, 0] = float(self.cfg['inertia']['xy'])
        self._inertia[1, 1] = float(self.cfg['inertia']['yy'])
        self._inertia[1, 2] = float(self.cfg['inertia']['yz'])
        self._inertia[2, 0] = float(self.cfg['inertia']['xz'])
        self._inertia[2, 1] = float(self.cfg['inertia']['yz'])
        self._inertia[2, 2] = float(self.cfg['inertia']['zz'])
        self._inverse_inertia = np.linalg.inv(self._inertia)

        self._drag_coeff_momentum = np.eye(3).astype(np.float32)
        self._drag_coeff_momentum[0, 0] = float(self.cfg['drag']['m_xx'])
        self._drag_coeff_momentum[1, 1] = float(self.cfg['drag']['m_yy'])
        self._drag_coeff_momentum[2, 2] = float(self.cfg['drag']['m_zz'])

        self._drag_coeff_force = np.eye(3).astype(np.float32)
        self._drag_coeff_force[0, 0] = float(self.cfg['drag']['f_xx'])
        self._drag_coeff_force[1, 1] = float(self.cfg['drag']['f_yy'])
        self._drag_coeff_force[2, 2] = float(self.cfg['drag']['f_zz'])

        self._gravity_center = np.zeros(3, dtype=np.float32)
        self._gravity_center[0] = float(self.cfg['gravity_center']['x'])
        self._gravity_center[1] = float(self.cfg['gravity_center']['y'])
        self._gravity_center[2] = float(self.cfg['gravity_center']['z'])

        self._thrust_coeff_ct_0 = float(self.cfg['thrust']['CT'][0])
        self._thrust_coeff_ct_1 = float(self.cfg['thrust']['CT'][1])
        self._thrust_coeff_ct_2 = float(self.cfg['thrust']['CT'][2])

        self._thrust_coeff_mm = float(self.cfg['thrust']['Mm'])
        self._thrust_coeff_jm = float(self.cfg['thrust']['Jm'])
        self._thrust_coeff_phi = float(self.cfg['thrust']['phi'])
        self._thrust_coeff_ra = float(self.cfg['thrust']['RA'])

        self._fail_max_velocity = float(self.cfg['fail']['velocity'])
        self._fail_max_range = float(self.cfg['fail']['range'])
        self._fail_max_angular_velocity = float(self.cfg['fail']['w'])

        self._propeller_coord = np.zeros((4, 3), dtype=np.float32)
        self._propeller_coord[0, 0] = float(self.cfg['propeller'][0]['x'])
        self._propeller_coord[0, 1] = float(self.cfg['propeller'][0]['y'])
        self._propeller_coord[0, 2] = float(self.cfg['propeller'][0]['z'])
        self._propeller_coord[1, 0] = float(self.cfg['propeller'][1]['x'])
        self._propeller_coord[1, 1] = float(self.cfg['propeller'][1]['y'])
        self._propeller_coord[1, 2] = float(self.cfg['propeller'][1]['z'])
        self._propeller_coord[2, 0] = float(self.cfg['propeller'][2]['x'])
        self._propeller_coord[2, 1] = float(self.cfg['propeller'][2]['y'])
        self._propeller_coord[2, 2] = float(self.cfg['propeller'][2]['z'])
        self._propeller_coord[3, 0] = float(self.cfg['propeller'][3]['x'])
        self._propeller_coord[3, 1] = float(self.cfg['propeller'][3]['y'])
        self._propeller_coord[3, 2] = float(self.cfg['propeller'][3]['z'])

        self._max_voltage = float(self.cfg['electric']['max_voltage'])
        self._min_voltage = float(self.cfg['electric']['min_voltage'])

    def _get_pitch_roll_yaw(self):
        roll = np.arctan2(self.rotation_matrix[2, 1],
                          self.rotation_matrix[2, 2])
        pitch = np.arctan2(
            -self.rotation_matrix[2, 0],
            np.sqrt(self.rotation_matrix[2, 1] ** 2 +
                    self.rotation_matrix[2, 2] ** 2))
        yaw = np.arctan2(self.rotation_matrix[1, 0],
                         self.rotation_matrix[0, 0])
        return pitch, roll, yaw

    def _run_internal(self, act):
        # Update propeller state
        prop_force = np.zeros(3).astype(np.float32)
        prop_torque = np.zeros(3).astype(np.float32)
        prop_powers = np.zeros(4).astype(np.float32)
        me = np.zeros(4).astype(np.float32)

        for i in range(4):
            eff_act = act[i]
            if eff_act > self._max_voltage:
                eff_act = self._max_voltage
            elif eff_act < self._min_voltage:
                eff_act = self._min_voltage

            phi_w = self._thrust_coeff_phi * self.propeller_angular_velocity[i]
            me[i] = self._thrust_coeff_phi / self._thrust_coeff_ra * \
                (eff_act - phi_w)
            prop_powers[i] = abs(me[i] / self._thrust_coeff_phi * eff_act)

            d_prop_w = 1.0 / self._thrust_coeff_jm * \
                (me[i] - self._thrust_coeff_mm)

            w_m = self.propeller_angular_velocity[i] + \
                self._precision * d_prop_w
            l_m = np.linalg.norm(self._propeller_coord[i])
            body_velocity = np.matmul(self._coordination_converter_to_body,
                                      self.global_velocity)
            bw_to_v = np.cross(
                self.body_angular_velocity, self._propeller_coord[i]) * l_m
            v_1 = body_velocity[2] + bw_to_v[2]
            sign = 1.0 if v_1 > 0 else -1.0

            thrust = self._thrust_coeff_ct_0 * w_m * w_m + \
                self._thrust_coeff_ct_1 * w_m * v_1 + \
                self._thrust_coeff_ct_2 * v_1 * v_1 * sign

            self.propeller_angular_velocity[i] = w_m
            prop_force[2] += thrust
            prop_torque += np.cross(
                -np.array([0.0, 0.0, thrust], dtype=np.float32),
                self._propeller_coord[i])

        prop_torque[2] += -me[0] + me[1] - me[2] + me[3]

        f_drag = -np.linalg.norm(self.global_velocity) * \
            np.matmul(
                np.matmul(self._drag_coeff_force,
                          self._coordination_converter_to_body),
                self.global_velocity)
        t_drag = -np.linalg.norm(self.body_angular_velocity) * \
            np.matmul(self._drag_coeff_momentum, self.body_angular_velocity)

        gravity_acc = np.array([0.0, 0.0, -GRAVITY_ACCELERATION],
                               dtype=np.float32)
        f_grav = np.matmul(self._coordination_converter_to_body,
                           gravity_acc) * self._quality
        t_grav = -np.cross(f_grav, self._gravity_center)

        f_all = prop_force + f_grav + f_drag
        t_all = prop_torque + t_grav + t_drag

        body_acc = f_all / self._quality
        acc = np.matmul(self._coordination_converter_to_world, body_acc)
        self.global_position += self.global_velocity * self._precision + \
            0.5 * self._precision * self._precision * acc
        self.global_velocity += self._precision * acc
        self.power = np.sum(prop_powers)

        tmp_w = self.body_angular_velocity + 0.5 * self._precision * \
            np.matmul(self._inverse_inertia, t_all)

        skew_sym_w = np.zeros((3, 3)).astype(np.float32)
        skew_sym_w[0, 1] = -tmp_w[2]
        skew_sym_w[0, 2] = tmp_w[1]
        skew_sym_w[1, 0] = tmp_w[2]
        skew_sym_w[1, 2] = -tmp_w[0]
        skew_sym_w[2, 0] = -tmp_w[1]
        skew_sym_w[2, 1] = tmp_w[0]

        self.rotation_matrix += self._precision * \
            np.matmul(self.rotation_matrix, skew_sym_w)
        self.body_angular_velocity += self._precision * \
            np.matmul(self._inverse_inertia, t_all)

        self._coordination_converter_to_world = self.rotation_matrix
        self._coordination_converter_to_body = np.linalg.inv(
            self.rotation_matrix)

        self._check_failure()

    def _check_failure(self):
        if np.linalg.norm(self.global_position) > self._fail_max_range:
            raise Exception('The quadrotor exists the valid zone')

        if np.linalg.norm(self.global_velocity) > self._fail_max_velocity:
            raise Exception('The quadrotor has too large velocity to recover')

        if np.linalg.norm(self.body_angular_velocity) > \
           self._fail_max_angular_velocity:
            raise Exception('The quadrotor has too large angular velocity')

    def get_config(self, config_file):
        with open(config_file, 'r') as f:
            self.cfg = json.load(f)

        try:
            self._parse_cfg()
        except Exception as e:
            raise RuntimeError('Error in loading configuration: ' + str(e))

        self._is_configured = True
        return {
            'action_space_low': self._min_voltage,
            'action_space_high': self._max_voltage,
            'range': self._fail_max_range
        }

    def reset(self):
        self._zero_state()
        self.coordination_converter_to_world = self.rotation_matrix
        self.coordination_converter_to_body = np.linalg.inv(
            self.rotation_matrix)

    def get_state(self):
        body_velocity = np.matmul(self._coordination_converter_to_body,
                                  self.global_velocity)
        return {
            'b_v_x': body_velocity[0],
            'b_v_y': body_velocity[1],
            'b_v_z': body_velocity[2]
        }

    def get_sensor(self):
        # TODO: add noisy
        gravity_acc = np.array([0, 0, -GRAVITY_ACCELERATION], dtype=np.float32)
        imu_acc = self.body_acceleration + \
            np.matmul(self._coordination_converter_to_body, gravity_acc)
        imu_gyro = self.body_angular_velocity
        barometer = self.global_position[2]
        pitch, roll, yaw = self._get_pitch_roll_yaw()
        return {
            'z': barometer,
            'acc_x': imu_acc[0],
            'acc_y': imu_acc[1],
            'acc_z': imu_acc[2],
            'gyro_x': imu_gyro[0],
            'gyro_y': imu_gyro[1],
            'gyro_z': imu_gyro[2],
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw
        }

    def step(self, act, dt):
        if not self._is_configured:
            raise RuntimeError('Simulator is not configured')

        if self._precision < 1e-8 or self._precision > dt:
            raise ValueError('Inproper parameter of precision')

        times = int(dt / self._precision)
        for _ in range(times):
            self._run_internal(act)

    def define_velocity_control_task(self, dt, nt, seed):
        np.random.seed(seed)
        current_state = self._save_state()

        velocity_lst = []
        for _ in range(nt):
            act = np.random.uniform(
                low=self._min_voltage, high=self._max_voltage, size=4)
            act = act.astype(np.float32)
            self.step(act, dt)

            body_velocity = np.matmul(
                self._coordination_converter_to_body,
                self.global_velocity)
            velocity_lst.append(list(body_velocity))

        self._restore_state(current_state)
        return velocity_lst
