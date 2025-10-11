# Import Gymnasium stuff
import gymnasium as gym
from gymnasium import Env
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
class TrainSpeedControl2(Env):

    def __init__(self):
        # --- 固定参数 ---
        self.dt = 1.0
        self.Mass = 360.0
        self.Max_traction_F = 0.0
        self.Episode_time = 600.0
        self.Running_time = 150.0
        self.sensor_range = 500.0

        # --- 环境参数 ---
        self.track_length = 2500.0
        self.station = 2000.0
        self.max_speed = 30  # ≈ 80 km/h

        # --- 限速表 (可在 reset 中随机化) ---
        self.speed_limit_positions = np.array([0.0, 500.0, 1200.0, 1800.0, self.station], dtype=float)
        self.speed_limits = np.array([22.222, 22.222, 22.222, 22.222, 0.0], dtype=float)

        # --- 状态上下限 ---
        self.specs = {
            'velocity_limits': [0, self.max_speed],
            'power_limits': [0, 75],
            'distance_limits': [0, self.track_length],
            'Episode_time': [0, self.Episode_time],
        }

        # 观测空间： [distance_left, time_left, velocity, speed_limit, distance_to_next_change, next_speed_limit]
        self.state_max = np.hstack((
            self.specs['distance_limits'][1],
            self.specs['Episode_time'][1],
            self.specs['velocity_limits'][1],
            self.specs['velocity_limits'][1],
            self.sensor_range,  # 最大探测距离
            self.specs['velocity_limits'][1]  # 未来限速
        ))
        self.state_min = np.hstack((
            self.specs['distance_limits'][0],
            self.specs['Episode_time'][0],
            self.specs['velocity_limits'][0],
            self.specs['velocity_limits'][0],
            0.0,
            0.0
        ))

        from gymnasium.spaces import Box
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=self.state_min, high=self.state_max, dtype=np.float64)

        self._max_episode_steps = int(self.Episode_time / self.dt)
        self.episode_count = 0
        self.reroll_frequency = 10  # 你已有的参数，保留

        # ===== 新增：势函数 shaping / 课程式终点 / 安全投影参数 =====
        self.gamma_shape = 0.99  # 与算法的折扣一致（或略小）
        self.kappa_shape = 0.10  # 势函数系数（前进奖励强度）

        # 课程式终点窗口（可在训练中逐渐收紧）
        self.terminal_pos_tol = 40.0  # m
        self.terminal_time_tol = 40.0  # s
        self.terminal_v_tol = 0.5  # m/s
        self.terminal_bonus = 300.0

        # 安全投影层的保守最大减速度（与 evaluation/前瞻保持一致）
        self.max_decel_est = 1.0  # 可用最大减速度(>0)
        self.safety_margin = 0.95  # 安全裕度 [0.90,0.98]
        self.brake_soften = 1.00  # 刹车软化系数 (<=1 更柔)
        self.v_cap_margin = 2.0  # 可行速度上界过渡带 (m/s)

        # 势函数需要的“上一步距离”与安全投影需要的“上一步前视”
        self.distance_left_prev = None
        self.last_d_next = 1e9
        self.last_v_next = self.max_speed

    # ====================== reset ======================
    def reset(self, *, seed=None, options=None):
        # （可选）随机化环境以做 curriculum / domain randomization
        # 例如按固定频率在若干预设模板之间切换：
        # if self.episode_count % self.reroll_frequency == 0:
        #     # 示例：随机站距、运行时间、限速图（保持单调不增至站）
        #     self.station = np.random.uniform(1800.0, 2200.0)
        #     self.Running_time = np.random.uniform(130.0, 160.0)
        #     self.speed_limit_positions = np.array([0.0, 500.0, 1200.0, max(1600.0, self.station-200.0), self.station], dtype=float)
        #     base = 22.222
        #     self.speed_limits = np.array([base, base, base, base, 0.0], dtype=float)

        # 计数器（如果需要）
        self.episode_count = getattr(self, "episode_count", 0) + 1

        # --- 状态变量初始化 ---
        self.position = 0.0
        self.distance_left = self.station
        self.velocity = 0.0
        self.acceleration = 0.0
        self.prev_acceleration = 0.0
        self.traction_power = 0.0
        self.action_clipped = 0.0
        self.jerk = 0.0
        self.prev_action = 0.0
        self.time = 0.0
        self.time_left = self.Running_time
        self.total_energy_kWh = 0.0
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        action_safe = 0.0
        action_raw = 0.0

        # 当前限速
        self.speed_limit = self._current_speed_limit()

        # 传感器信息（并写入给安全投影缓存）
        distance_to_next_change, next_speed_limit = self.sensor(range_m=self.sensor_range)
        self.last_d_next = distance_to_next_change
        self.last_v_next = next_speed_limit

        # 势函数 shaping 的“上一步距离”
        self.distance_left_prev = self.distance_left

        # （可选）根据训练进度收紧终点窗口/提高终点奖励
        # 例如每过若干回合微调一次：
        # if self.episode_count % 2000 == 0:
        #     self.terminal_pos_tol = max(10.0, self.terminal_pos_tol - 5.0)
        #     self.terminal_time_tol = max(10.0, self.terminal_time_tol - 5.0)
        #     self.terminal_bonus = min(1000.0, self.terminal_bonus + 50.0)

        info = {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'jerk': self.jerk,
            'time': self.time,
            'power': self.traction_power,
            'energy': self.total_energy_kWh,
            'reward': self.reward,
            'distance_to_next_change': distance_to_next_change,
            'next_speed_limit': next_speed_limit,
            'action_safe': action_safe,
            'action_raw': action_raw
        }

        # 观测（含传感器）
        state = np.hstack([
            self.distance_left,
            self.time_left,
            self.velocity,
            self.speed_limit,
            distance_to_next_change,
            next_speed_limit
        ])

        state = self._normalize_obs(state)

        return state, info

    # ====================== step ======================
    def step(self, action):
        assert self.action_space.contains(action)

        # --- 安全投影层：在 update_motion 之前 ---
        action_safe = self._safety_project_action(action)

        # 运动学更新（用投影后的动作）
        self.update_motion(action_safe)

        # 时间 & 剩余量更新
        self.time += self.dt
        self.time_left = max(self.Running_time - self.time, 0.0)
        self.distance_left = max(self.station - self.position, 0.0)

        # 限速 & 传感器
        self.speed_limit = self._current_speed_limit()
        distance_to_next_change, next_speed_limit = self.sensor(range_m=self.sensor_range)
        if self.station <= self.position:
            self.speed_limit = 0.0

        # --- 终止/截断：失败不立刻 done，只按 Episode_time 截断 ---
        pos_ok = abs(self.station - self.position) <= self.terminal_pos_tol
        time_ok = abs(self.Running_time - self.time) <= self.terminal_time_tol
        v_ok = self.velocity <= self.terminal_v_tol
        self.terminated = bool(pos_ok and time_ok and v_ok)
        self.truncated = bool(self.time > self.Episode_time)

        # --- 奖励：成本(负向) + 势函数shaping + 终点奖励 ---
        base_cost = self.evaluation_func(
            action_clipped=action_safe,
            dist_next=distance_to_next_change,
            next_limit=next_speed_limit
        )

        # 势函数 shaping：Φ(s) = -kappa * d_left
        if self.distance_left_prev is None:
            self.distance_left_prev = self.distance_left

        progress = max(0.0, self.distance_left_prev - self.distance_left)
        r_shape = self.kappa_shape * (progress / max(self.dt, 1e-6))
        # r_shape = self.kappa_shape * (
        #         self.distance_left_prev - self.gamma_shape * self.distance_left
        # )
        self.distance_left_prev = self.distance_left

        self.reward = -base_cost + r_shape

        if self.terminated:
            self.reward += self.terminal_bonus

        info = {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'jerk': self.jerk,
            'time': self.time,
            'power': self.traction_power,
            'energy': self.total_energy_kWh,
            'reward': self.reward,
            'distance_to_next_change': distance_to_next_change,
            'next_speed_limit': next_speed_limit,
            'action_safe': action_safe,
            'action_raw': self._proj_a_raw
        }

        state = np.hstack([
            self.distance_left,
            self.time_left,
            self.velocity,
            self.speed_limit,
            distance_to_next_change,
            next_speed_limit
        ])

        state = self._normalize_obs(state)

        return state, self.reward, self.terminated, self.truncated, info

    def evaluation_func(self, action_clipped, dist_next, next_limit):
        """
        Returns a scalar cost to be minimized (step() negates it).
        Use Huber/linear penalties (别用全tanh饱和)，
        强化超速/刹晚/能耗/舒适的“疼痛感”。
        """
        eps = 1e-6

        # --- 基本量 ---
        v = max(self.velocity, 0.0)
        v_lim = max(self.speed_limit, 0.0)
        v_next = max(next_limit, 0.0)
        d_left = max(self.distance_left, 0.0)
        d_next = max(dist_next, 0.0)

        a_brake = float(self.max_decel_est)  # 用于前瞻与安全投影的一致参数

        # --- Huber 损失 ---
        def huber(x, k=1.0):
            ax = np.abs(x)
            return np.where(ax < k, 0.5 * x * x / k, ax - 0.5 * k)

        # 1) 超速：超出1 m/s 后线性增痛
        overspeed = huber(max(v - v_lim, 0.0), k=1.0)

        # 2) 前瞻：下一限速的刹车距离不足
        if v > v_next:
            d_need_next = (v ** 2 - v_next ** 2) / (2.0 * a_brake + eps)
            next_brake_risk = huber(max(d_need_next - d_next, 0.0), k=20.0)
        else:
            next_brake_risk = 0.0

        # 3) 前瞻：到站刹停距离不足
        d_need_stop = (v ** 2) / (2.0 * a_brake + eps)
        stop_brake_risk = huber(max(d_need_stop - d_left, 0.0), k=20.0)

        # 4) 能耗：只惩罚牵引（正功率）
        pos_power = max(self.traction_power, 0.0)
        energy = 0.001 * pos_power  # 线性、系数小

        # 5) 舒适：用加速度的 Huber
        comfort = huber(abs(self.acceleration), k=0.5)

        # 6) 时间项（可选，弱化或关闭，主要靠势函数+终点奖励调时间）
        time_term = 0.0

        # 权重（先从这个起点调）
        w = dict(
            overspeed=3.0,
            next_brake=2.0,
            stop_brake=3.0,
            energy=0.1,
            comfort=0.1,
            time_term=0.0
        )

        cost = (
                w['overspeed'] * overspeed +
                w['next_brake'] * next_brake_risk +
                w['stop_brake'] * stop_brake_risk +
                w['energy'] * energy +
                w['comfort'] * comfort +
                w['time_term'] * time_term
        )
        return float(cost)

    def _safety_project_action(self, action):
        """
        安全投影（最小侵入，能产生负样本）：
        1) 计算到站/到下一限速的“所需制动比例”并强制 a <= -brake_frac；
        2) 前瞻可行速度包络：即便 v < v_next，也柔性限正牵引；
        3) 近站正牵引线性衰减到 0。
        记录 a_raw/a_safe/触发标志与各比例，便于调试与奖励轻罚。
        """
        # -------- 基本量与参数 --------
        a_raw = float(np.clip(action, -1.0, 1.0))
        a = a_raw
        eps = 1e-8

        v = max(self.velocity, 0.0)
        d_left = max(self.station - self.position, 0.0)

        a_max = float(getattr(self, "max_decel_est", 1.0))  # 最大可用减速度(正)
        safety_margin = float(getattr(self, "safety_margin", 0.95))  # 安全裕度
        soften = float(getattr(self, "brake_soften", 1.00))  # 刹车软化系数
        v_cap_margin = float(getattr(self, "v_cap_margin", 2.0))  # 包络过渡带

        # 传感器缓存（若上步没写入则兜底）
        d_next = float(getattr(self, "last_d_next", 1e9))
        v_next = float(getattr(self, "last_v_next", getattr(self, "speed_limit", 0.0)))

        # -------- 工具函数：所需制动比例 --------
        def req_brake_frac(v_cur, v_tar, d_avail):
            if v_cur <= v_tar:
                return 0.0
            a_req = (v_cur * v_cur - v_tar * v_tar) / (2.0 * max(d_avail, eps))
            frac = (soften * a_req) / max(a_max, eps)
            return float(np.clip(frac, 0.0, 1.0))

        # 可用距离（留裕度）
        d_stop_avail = safety_margin * d_left
        d_next_avail = safety_margin * max(d_next, 0.0)

        # 两类约束的制动比例
        frac_stop = req_brake_frac(v_cur=v, v_tar=0.0, d_avail=d_stop_avail)
        frac_next = req_brake_frac(v_cur=v, v_tar=v_next, d_avail=d_next_avail) if v > v_next else 0.0

        # 若必须刹车：至少给对应强度的负动作
        brake_frac = max(frac_stop, frac_next)
        if brake_frac > 0.0:
            a = min(a, -brake_frac)

        # -------- 前瞻可行速度包络（即便 v < v_next 也避免将来刹不回） --------
        # 上界确保：能在 d_stop_avail 刹到 0、能在 d_next_avail 刹到 v_next、且不超当前区段限速
        # v_cap_stop = np.sqrt(max(0.0, 2.0 * a_max * max(d_stop_avail, eps)))
        # v_cap_next = np.sqrt(max(0.0, v_next * v_next + 2.0 * a_max * max(d_next_avail, eps)))
        # v_lim = float(getattr(self, "speed_limit", np.inf))
        # v_cap = min(v_lim, v_cap_stop, v_cap_next)
        #
        # if v >= v_cap:
        #     # 已到不可行上界：禁止正牵引
        #     a = min(a, 0.0)
        #     cap_allow = 0.0
        # else:
        #     # 距上界越近，允许正牵引越小（线性过渡）
        #     cap_allow = float(np.clip((v_cap - v) / max(v_cap_margin, eps), 0.0, 1.0))
        #     if a > cap_allow:
        #         a = cap_allow

        # -------- 近站台：正牵引再线性衰减（与你原逻辑一致） --------
        if d_left < 200.0:
            allow_pos_near = max(0.0, d_left / 200.0)
            if a > allow_pos_near:
                a = allow_pos_near

        # -------- 记录调试字段（在 step() 里塞进 info 使用） --------
        self._proj_a_raw = a_raw
        self._proj_a_safe = a
        self._proj_triggered = int(a != a_raw)
        self._proj_brake_frac = brake_frac
        self._proj_frac_stop = frac_stop
        self._proj_frac_next = frac_next
        # self._proj_cap_allow = cap_allow
        # self._proj_v_cap = v_cap

        return a

    def update_motion(self, action_clipped):
        resistance = self.Calc_Resistance()

        # --- slope force ---
        g = 9.81
        pos = self.position
        L = self.station
        if 0 < pos < 300.0:
            gradient = -0.018
        elif L - 300.0 < pos < L:
            gradient = 0.018
        else:
            gradient = 0.0
        slope_force = - self.Mass * g * gradient

        # 牵引/制动力与功率
        if self.velocity > 0:
            if action_clipped >= 0:
                force = action_clipped * self.Calc_Max_traction_F()
                self.traction_power = force * self.velocity
            else:
                force = action_clipped * self.Calc_Max_braking_F()
                self.traction_power = 0.0

            self.acceleration = (force - resistance + slope_force) / self.Mass
            # 防止反向（速度不能跨过0）
            if self.velocity + self.acceleration * self.dt < 0:
                self.acceleration = -self.velocity / self.dt

        elif self.velocity == 0:
            if action_clipped > 0:
                force = action_clipped * self.Calc_Max_traction_F()
            else:
                force = 0.0
            self.acceleration = max(0.0, (force - resistance + slope_force) / self.Mass)
            self.traction_power = 0.0  # v=0 时功率视为0

        # 状态积分
        self.position += (0.5 * self.acceleration * self.dt ** 2 + self.velocity * self.dt)
        self.velocity += self.acceleration * self.dt
        self.total_energy_kWh += (self.traction_power * self.dt) / 3600.0

    import numpy as np

    def _current_speed_limit(self, pos=None):
        """返回当前位置生效的限速值 [m/s]"""
        if pos is None:
            pos = self.position
        idx = np.searchsorted(self.speed_limit_positions, pos, side='right') - 1
        idx = np.clip(idx, 0, len(self.speed_limits) - 1)
        return float(self.speed_limits[idx])

    def sensor(self, range_m=500.0):
        """
        返回：
        distance_to_next_change [m], next_speed_limit [m/s]
        若500m内无变更，则返回(500, 当前限速)
        """
        pos = self.position
        next_idx = np.searchsorted(self.speed_limit_positions, pos, side='right')
        if next_idx >= len(self.speed_limit_positions):
            dist_to_end = max(self.station - pos, 0.0)
            if dist_to_end <= range_m:
                return dist_to_end, float(self.speed_limits[-1])
            else:
                return range_m, self._current_speed_limit(pos)

        next_change_pos = float(self.speed_limit_positions[next_idx])
        dist_to_change = max(next_change_pos - pos, 0.0)
        if dist_to_change <= range_m:
            next_limit = float(self.speed_limits[next_idx])
            return dist_to_change, next_limit
        else:
            return range_m, self._current_speed_limit(pos)

    def _normalize_obs(self, state):
        return 2.0 * (state - self.state_min) / (self.state_max - self.state_min) - 1.0

    def Calc_Max_traction_F(self):
        """
        Calculate the traction force based on the speed in m/s.

        Parameters:
        - speed (float): Speed in km/h

        Returns:
        - float: Traction force in kN
        """
        speed = self.velocity * 3.6  # Convert speed from m/s to km/h
        f_t = 263.9  # Initial traction force value in kN (acceleration phase)
        p_max = f_t * 43 / 3.6  # Maximum power during acceleration in kW

        # If power exceeds the maximum power limit, then limit the traction force
        if speed > 0:
          if (f_t * speed / 3.6) > p_max:
              f_t = p_max / (speed / 3.6)

          # Additional condition to limit the traction force
          if f_t > (263.9 * 43 * 50 / (speed ** 2)):
              f_t = 263.9 * 43 * 50 / (speed ** 2)
        if speed == 0:
            f_t = 263.9  # Set traction force to initial value if speed is 0

        return f_t

    def Calc_Max_braking_F(self):
        """
        Calculate the braking force based on the speed.

        Parameters:
        - speed (float): Speed in km/h

        Returns:
        - float: Braking force in kN
        """

        speed = self.velocity * 3.6  # Convert speed from m/s to km/h
        if speed <= 0:
            f_b = 200
        else:
            if speed > 0 and speed <= 5:
                f_b = 200
            elif speed > 5 and speed <= 48.5:
                f_b = 389
            elif speed > 48.5 and speed <= 80:
                f_b = 913962.5 / (speed ** 2)
            else:
                f_b = 200  # Assumes no braking force calculation outside specified range

        # Apply a final modification factor to the braking force
        f_b = 1.6 * f_b

        return f_b

    def Calc_Resistance(self):
        """
        Calculate the basic resistance of a train running at a given speed.

        :param speed: Speed of the train in km/h
        :return: Basic resistance in kN
        """
        n = 24  # Number of axles
        N = 6  # Number of cars
        A = 10.64  # Cross-sectional area of the train in m^2
        speed = self.velocity * 3.6  # Convert speed from m/s to km/h

        f_r = (6.4 * self.Mass + 130 * n + 0.14 * self.Mass * abs(speed) +
              (0.046 + 0.0065 * (N - 1)) * A * speed**2) / 1000
        # f_r = 0.1 * f_r
        return f_r


    def render(self):
        pass