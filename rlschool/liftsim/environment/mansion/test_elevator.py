#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

"""
Unit test class
"""
from rlschool.liftsim.environment.mansion.person_generators.uniform_generator import UniformPersonGenerator
from rlschool.liftsim.environment.mansion import person_generators
from rlschool.liftsim.environment.mansion.person_generators import uniform_generator
from rlschool.liftsim.environment.mansion.utils import PersonType, MansionState, ElevatorState, ElevatorAction
from rlschool.liftsim.environment.mansion.elevator import Elevator
from rlschool.liftsim.environment.mansion.mansion_manager import MansionManager
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
import sys
import unittest
import mock


class TestElevator(unittest.TestCase):
    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_door_load_unload(self, mock_uniformgenerator):
        """
        stop at the target, load and unload corresponding passengers, open and close the door properly
        """
        max_floors = 8
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0)

        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")
        test_elevator._direction = 1
        test_elevator._current_position = 8.0
        test_elevator._target_floors = [3, 5]
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._loaded_person[2].append(
            PersonType(6, 40, 1, 3, world.raw_time))
        test_elevator._loaded_person[4].append(
            PersonType(7, 35, 1, 5, world.raw_time))
        test_elevator._load_weight = 80

        tmp_uniform_generator = UniformPersonGenerator()

        ret_person = []
        ret_person.append(PersonType(0, 50, 3, 5, world.raw_time))
        ret_person.append(PersonType(1, 30, 3, 1, world.raw_time))
        ret_person.append(PersonType(2, 60, 6, 4, world.raw_time))
        ret_person.append(PersonType(4, 55, 3, 4, world.raw_time))
        ret_person.append(PersonType(5, 65, 3, 6, world.raw_time))
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        test_mansion.run_mansion(dispatch)
        # print(test_mansion.state, "\nworld time is", world.raw_time)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 0.5)

        # mock generate_person again
        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion.run_mansion(dispatch)    # Door fully open, t = 1.0
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 1.0)

        for i in range(4):
            test_mansion.run_mansion(dispatch)
            # print(test_mansion.state, "\nworld time is", world.raw_time)
        state = test_mansion.state    # passenger 6 is unloaded, t = 3.0
        self.assertAlmostEqual(state.ElevatorStates[0].LoadWeight, 40)

        dispatch = []
        dispatch.append(ElevatorAction(0, 0))

        for i in range(4):
            test_mansion.run_mansion(dispatch)
            # print(test_mansion.state, "\nworld time is", world.raw_time)
        state = test_mansion.state    # passenger 0 and 4 are loaded, t = 5.0
        self.assertAlmostEqual(state.ElevatorStates[0].LoadWeight, 145)

        for i in range(4):
            test_mansion.run_mansion(dispatch)
        state = test_mansion.state    # passenger 5 is loaded, t = 7.0
        self.assertAlmostEqual(state.ElevatorStates[0].LoadWeight, 210)

        for i in range(4):
            test_mansion.run_mansion(dispatch)
            # print(test_mansion.state, "\nworld time is", world.raw_time)
        state = test_mansion.state    # the door is closed, going up, t = 9.0
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 1.0)

    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_overload(self, mock_uniformgenerator):
        """
        overload, two people enter together, check who can enter the elevator one by one
        after overload, if the dispatcher still dispatches the elevator to the current floor, ignore the dispatch
        """
        max_floors = 8
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0)

        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")
        test_elevator._direction = 1
        test_elevator._current_position = 8.0
        test_elevator._target_floors = [5]
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._loaded_person[5].append(
            PersonType(6, 750, 1, 6, world.raw_time))
        test_elevator._loaded_person[7].append(
            PersonType(7, 750, 1, 8, world.raw_time))
        test_elevator._load_weight = 1500

        tmp_uniform_generator = UniformPersonGenerator()

        ret_person = []
        ret_person.append(PersonType(0, 150, 3, 5, world.raw_time))
        ret_person.append(PersonType(1, 50, 3, 1, world.raw_time))
        ret_person.append(PersonType(2, 60, 5, 6, world.raw_time))
        ret_person.append(PersonType(4, 65, 3, 8, world.raw_time))
        ret_person.append(PersonType(5, 65, 3, 6, world.raw_time))
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        test_mansion.run_mansion(dispatch)

        # mock generate_person again
        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion.run_mansion(dispatch)    # Door fully open, t = 1.0

        dispatch = []
        dispatch.append(ElevatorAction(-1, 0))

        for i in range(4):
            test_mansion.run_mansion(dispatch)  # upload person 4, t = 3.0

        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].LoadWeight, 1565)

        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)  # t = 4.5
        state = test_mansion.state
        self.assertGreater(state.ElevatorStates[0].Velocity, 0.0)
        # print(test_mansion.state, "\nworld time is", world.raw_time)

    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_stop_at_dispatch(self, mock_uniformgenerator):
        """
        stop at the dispatch floor, open and close the door, then keep going to the target floor
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0)

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")
        test_elevator._direction = 1
        test_elevator._current_velocity = 2.0
        test_elevator._current_position = 4.0   # currently at 2 floor
        test_elevator._target_floors = [5]
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._loaded_person[5].append(
            PersonType(0, 50, 1, 6, world.raw_time))
        test_elevator._load_weight = 50

        # test_mansion
        tmp_uniform_generator = UniformPersonGenerator()

        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]

        # test
        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        for i in range(7):
            test_mansion.run_mansion(dispatch)  # stop at the dispatched floor
        # print(test_mansion.state, "\nworld time is", world.raw_time)
        # dispatch = []
        # dispatch.append(ElevatorAction(-1, 0))
        for i in range(2):
            # the door is fully open, t = 4.5
            test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 1.0)

        dispatch = []
        dispatch.append(ElevatorAction(0, 0))

        for i in range(6):
            # finish time open lag and close the door
            test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 0.0)

        for i in range(4):
            test_mansion.run_mansion(dispatch)  # then keep going up
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 2.0)

    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_dispatch_when_closing(self, mock_uniformgenerator):
        """
        dispatch the current floor when the door is closing
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0)

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")

        test_elevator._direction = 1
        test_elevator._current_position = 8.0
        test_elevator._target_floors = [4, 5]
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._loaded_person[3].append(
            PersonType(6, 40, 1, 4, world.raw_time))
        test_elevator._loaded_person[4].append(
            PersonType(7, 40, 1, 5, world.raw_time))
        test_elevator._load_weight = 80

        # test_mansion
        tmp_uniform_generator = UniformPersonGenerator()

        ret_person = []
        ret_person.append(PersonType(0, 50, 3, 5, world.raw_time))
        ret_person.append(PersonType(1, 50, 3, 1, world.raw_time))
        ret_person.append(PersonType(2, 60, 6, 4, world.raw_time))
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        # run_mansion
        test_mansion.run_mansion(dispatch)

        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion.run_mansion(dispatch)  # the door is open, t = 1.0
        # print(test_mansion.state, "\nworld time is", world.raw_time)

        dispatch = []
        dispatch.append(ElevatorAction(-1, 0))
        for i in range(4):
            test_mansion.run_mansion(dispatch)  # load person 0, t = 3.0

        # the door is closing, the door state = 0.5, t = 3.5
        test_mansion.run_mansion(dispatch)

        # come two more passengers
        ret_person = []
        ret_person.append(PersonType(4, 55, 3, 4, world.raw_time))
        ret_person.append(PersonType(5, 65, 3, 6, world.raw_time))
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        # the door is open, door_state = 1.0, time = 4.0
        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 1.0)

        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        dispatch = []
        dispatch.append(ElevatorAction(-1, 0))
        for i in range(4):
            test_mansion.run_mansion(dispatch)  # load the two passengers
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].LoadWeight, 250)

    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_dispatch_invalid(self, mock_uniformgenerator):
        """
        ignore the invalid dispatch (cannot stop at the dispatch)
        and decelerate when needed (test velocity_planner)
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0
        )

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_eleavtor")
        test_elevator._direction = 1
        test_elevator._current_velocity = 2.0
        test_elevator._current_position = 8.0  # currently at 3 floor
        test_elevator._target_floors = [5, 8]  # target 5 floor
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._loaded_person[4].append(
            PersonType(6, 40, 1, 5, world.raw_time))
        test_elevator._loaded_person[7].append(
            PersonType(7, 40, 1, 8, world.raw_time))
        test_elevator._load_weight = 80

        # test_mansion
        tmp_uniform_generator = UniformPersonGenerator()

        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(3, 1))

        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 2.0)

        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(
            state.ElevatorStates[0].Velocity,
            2.0)   # ignore the invalid dispatch

        for i in range(5):
            test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 0.0)

    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_no_dispatch(self, mock_uniformgenerator):
        """
        arrive at the target, no dispatch, hold still
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0
        )

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_eleavtor")
        test_elevator._direction = 1
        test_elevator._current_velocity = 0
        test_elevator._current_position = 8.0  # currently at 3 floor
        test_elevator._target_floors = [3]  # target 3 floor
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._loaded_person[2].append(
            PersonType(0, 40, 1, 3, world.raw_time))
        test_elevator._load_weight = 40

        # test_mansion
        tmp_uniform_generator = UniformPersonGenerator()

        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))

        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(-1, 0))

        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)  # open the door

        for i in range(4):
            test_mansion.run_mansion(dispatch)  # unload person 0

        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)  # close the door

        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 0.0)
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 0.0)
        self.assertAlmostEqual(state.ElevatorStates[0].Floor, 3.0)
        self.assertAlmostEqual(state.ElevatorStates[0].Direction, 0)

    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_dispatch_twice(self, mock_uniformgenerator):
        """
        no target, dispatch (3, 1) first, then (8, -1)
        decelerate then accelerate
        not accelerate immediately
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0
        )

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")
        test_elevator._direction = 1
        test_elevator._current_velocity = 2.0
        test_elevator._current_position = 9.0
        test_elevator._target_floors = list()
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._load_weight = 0

        # mansion
        tmp_uniform_generator = UniformPersonGenerator()
        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))
        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        # first, dispatch to 8 floor
        dispatch = []
        dispatch.append(ElevatorAction(4, 1))
        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)
        dispatch = []
        dispatch.append(ElevatorAction(8, -1))
        test_mansion.run_mansion(dispatch)    # accelerate at once
        test_mansion.run_mansion(dispatch)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 2.0)

    # checked
    # @unittest.skip("test")
    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_cancel_dispatch(self, mock_uniformgenerator):
        """
        no target, dispatch first, accelerate, then cancel dispatch, decelerate
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0
        )

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")
        test_elevator._direction = 0
        test_elevator._current_velocity = 0.0
        test_elevator._current_position = 8.0
        test_elevator._target_floors = list()
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._load_weight = 0

        # mansion
        tmp_uniform_generator = UniformPersonGenerator()
        ret_person = []
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))
        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(6, 1))
        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)  # t = 1.0
        dispatch = []
        dispatch.append(ElevatorAction(0, -1))
        for i in range(10):
            test_mansion.run_mansion(dispatch)

        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].DoorState, 0.0)
        self.assertAlmostEqual(state.ElevatorStates[0].Velocity, 0.0)

    @mock.patch("person_generators.uniform_generator.UniformPersonGenerator")
    def test_set_direction_0(self, mock_uniformgenerator):
        """
        When the elevator is stopped and empty, always set direction as 0 first, 
        then set as dispatch_target_direction
        """
        max_floors = 8
        # mansion_config
        world = MansionConfig(
            dt=0.50,
            number_of_floors=max_floors,
            floor_height=4.0
        )

        # test_elevator
        test_elevator = Elevator(start_position=0.0,
                                 mansion_config=world,
                                 name="test_elevator")
        test_elevator._direction = 1
        test_elevator._current_velocity = 0.0
        test_elevator._current_position = 8.0   # 3rd floor
        test_elevator._target_floors = list()
        test_elevator._loaded_person = [
            list() for i in range(
                test_elevator._number_of_floors)]
        test_elevator._load_weight = 0

        # mansion
        tmp_uniform_generator = UniformPersonGenerator()
        ret_person = []
        ret_person.append(PersonType(0, 50, 3, 1, world.raw_time))
        person_generators.uniform_generator.UniformPersonGenerator.generate_person = mock.Mock(
            return_value=(ret_person))
        test_mansion = MansionManager(
            elevator_number=1,
            person_generator=tmp_uniform_generator,
            mansion_config=world,
            name="test_mansion"
        )
        test_mansion._elevators = [test_elevator]
        dispatch = []
        dispatch.append(ElevatorAction(3, -1))
        test_mansion.run_mansion(dispatch)
        test_mansion.run_mansion(dispatch)  # t = 1.0
        # print(test_mansion.state, "\nworld time is", world.raw_time)
        state = test_mansion.state
        self.assertAlmostEqual(state.ElevatorStates[0].Direction, -1)


if __name__ == '__main__':
    unittest.main()
