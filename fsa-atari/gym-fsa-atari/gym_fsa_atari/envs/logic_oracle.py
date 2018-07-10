from gym import spaces
import numpy as np


class LogicOracle():

    def __init__(self, game):
        self.game = game
        self.start = True
        self.last = None

    # returns a tuple
    def get_logic_state(self, ram):
        return (0, 1)

    # returns list of human interpretable definitions
    # of each logic state
    # not totally necessary to implement this
    def get_logic_meanings(self):
        return ['No meaning', 'No meaning']

    # returns the logic's observation space
    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([2, 2])


class LogicOracleSpaceInvaders(LogicOracle):

    def get_logic_state(self, ram):

        laser, saucer, dead = [0] * 3
        if ram[78] != 0:
            laser = 1

        if ram[76] == 4:
            saucer = 1

        if ram[75] == 6:
            dead = 1

        # max aliens is 37
        aliens = int(ram[17] / 10)

        lives = ram[73] - 1

        left = ram[16] % 2  # whether the aliens are heading left

        # return (laser, saucer, aliens, dead)
        return (laser, aliens)

    def get_logic_meanings(self):
        # return ['Laser Active', 'Saucer Present', 'Remaining Aliens', 'Player Death']
        return ['Laser Active', 'Remaining Aliens']

    def get_logic_observation_space(self):
        # return spaces.MultiDiscrete([2, 2, 4, 2])
        return spaces.MultiDiscrete([2, 4])


class LogicOracleMontezumaRevenge(LogicOracle):

    def __init__(self, game):
        self.doors_left = {1:1, 5:1, 17:1}
        self.doors_right = {1:1, 5:1, 17:1}
        self.game = game

    def get_logic_state(self, ram):

        in_air, ladder, rope, dead = [0] * 4

        if format(ram[30], '02x') == 'a5':
            in_air = 1
        if format(ram[30], '02x') in ['3e', '52']:
            ladder = 1
        if format(ram[30], '02x') in ['90', '7b']:
            rope = 1
        if format(ram[30], '02x') in ['ba', 'c9', 'dd', 'c8']:
            dead = 1

        left_door = int(format(ram[66], '08b')[4])
        right_door = int(format(ram[66], '08b')[5])

        inv = format(ram[65], '08b')  # torch, sword, sword, key, key, key, key, mallet

        room = ram[3]
        lives = ram[58]

        if left_door == 0:
            self.doors_left[room] = 0
        if right_door == 0:
            self.doors_right[room] = 0

        return (lives, in_air, ladder, rope, dead,
                int(inv[0]), int(inv[1]), int(inv[2]), int(inv[3]),
                int(inv[4]), int(inv[5]), int(inv[6]), int(inv[7]),
                room, self.doors_left[1], self.doors_right[1],
                self.doors_left[5], self.doors_right[5],
                self.doors_left[17], self.doors_right[17])

    def get_logic_meanings(self):
        return ['Lives Left', 'In Air', 'On Ladder', 'On Rope', 'Is Dead',
                'Torch', 'Sword 1', 'Sword 2', 'Key 1', 'Key 2', 'Key 3',
                'Key 4', 'Mallet', 'Current Room',
                'Door 1 left', 'Door 1 right', 'Door 5 left', 'Door 5 right',
                'Door 17 left', 'Door 17 right']

    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 24,
                                     2, 2, 2, 2, 2, 2])


class LogicOracleVenture(LogicOracle):

    def __init__(self, game):
        self.game = game
        self.rooms = {0: 0, 1: 1, 2: 2, 64: 3, 8: 4}

    def get_logic_state(self, ram):

        room = self.rooms[ram[19]]
        lives = ram[70]
        inv = format(ram[17], '08b')

        can_fire = 0
        if ram[73] != 0:
            can_fire = 1

        paused_state = 0  # when dying or entering room (monsters don't move)
        if ram[96] == 0:
            paused_state = 1

        hall_monster = 0
        if room != 4 and ram[92] == 0:
            hall_monster = 1

        return (lives, room, int(inv[7]), int(inv[6]), int(inv[5]), int(inv[4]),
                can_fire, paused_state, hall_monster)

    def get_logic_meanings(self):
        return ['Lives Remaining', 'Room Number', 'Treasure 1', 'Treasure 2',
                'Treasure 3', 'Treasure 4', 'Can Fire', 'In Paused State',
                'Hall Monster']

    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([4, 5, 2, 2, 2, 2, 2, 2, 2])


class LogicOraclePrivateEye(LogicOracle):

    def __init__(self, game):
        self.game = game
        self.gun = 0
        self.money = 0
        self.fiend = 0

    def get_logic_state(self, ram):

        # picked up item
        if ram[60] == 20:
            self.gun = 1
        if ram[60] == 24:
            self.money = 1
        if ram[60] == 28:
            self.fiend = 1

        # returned the item
        if self.gun == 1 and ram[0x5c] == 23:
            self.gun = 2
        if self.money == 1 and ram[0x5c] == 7:
            self.money = 2

        # dropped the item
        if ram[60] == 0 and ram[5] == 20 and self.gun == 1:
            self.gun = 0
        if ram[60] == 0 and ram[5] == 24 and self.money == 1:
            self.money = 0
        if ram[60] == 0 and ram[5] == 28 and self.fiend == 1:
            self.fiend = 0

        return (ram[0x5c], self.gun, self.money, self.fiend)

    def get_logic_meanings(self):
        return ['Room Number', 'Gun Status', 'Money Status', 'Found Le Fiend']

    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([32, 3, 3, 2])


class LogicOraclePitfall(LogicOracle):

    def get_logic_state(self, ram):

        if ram[0] == 160:
            lives = 2
        elif ram[0] == 128:
            lives = 1
        elif ram[0] == 0:
            lives = 0

        underground = 0
        if ram[105] > 40:
            underground = 1

        room = ram[1]
        jump, ladder, vine = [0]*3
        if ram[100] == 0 and ram[4] == 0:
            jump = 1
        if ram[100] == 7 or ram[100] == 8:
            ladder = 1
        if ram[100] == 6:
            vine = 1
        treasures = 31- ram[113]

        return (lives, room, jump, ladder, vine, underground, treasures)

    def get_logic_meanings(self):
        return ['Lives Remaining', 'Room Number', 'Jumping', 'On Ladder',
                'On Vine', 'Underground', 'Treasures Collected']

    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([3, 255, 2, 2, 2, 2, 32])


class LogicOracleSolaris(LogicOracle):

    def get_logic_state(self, ram):

        scanner, space, planet = [0]*3
        if ram[13] in [32, 104]:
            scanner = 1
        if ram[13] in [0, 2, 16]:
            space = 1
        if ram[13] in [72, 74, 78]:
            planet = 1
        return (ram[0x59], scanner, space, planet)

    def get_logic_meanings(self):
        return ['Remaining Lives', 'Scanner Screen', 'In Space', 'On Planet']

    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([6, 2, 2, 2])


