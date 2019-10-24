#include <iostream>
#include <exception>
#include "common_tools.h"
#include "uranus_simulator.h"

namespace uranus{
namespace uranusim{
        Simulator::Simulator(){
            reset();
            _configured = false;
            _precision = 1.0e-4;
        }

        Simulator::~Simulator(){}

        void Simulator::reset(){
            _state = ZERO_STATE;
            _coordinate_converter_to_world = _state.rotation_matrix;
            _coordinate_converter_to_body = _state.rotation_matrix.inverse();
        }

        void Simulator::reset(State &state){
            _state = state;
            _coordinate_converter_to_world = _state.rotation_matrix;
            _coordinate_converter_to_body = _state.rotation_matrix.inverse();
        }

        void Simulator::reset(
                double g_x, double g_y, double g_z,
                double g_v_x, double g_v_y, double g_v_z,
                double w_x, double w_y, double w_z,
                double roll, double pitch, double yaw){
            _state.global_position(0) = g_x;
            _state.global_position(1) = g_y;
            _state.global_position(2) = g_z;
            _state.global_velocity(0) = g_v_x;
            _state.global_velocity(1) = g_v_y;
            _state.global_velocity(2) = g_v_z;
            _state.body_angular_velocity(0) = w_x;
            _state.body_angular_velocity(1) = w_y;
            _state.body_angular_velocity(2) = w_z;
            _state.propeller_angular_velocity(0) = 0;
            _state.propeller_angular_velocity(1) = 0;
            _state.propeller_angular_velocity(2) = 0;
            uranus::attitude_to_rotation(
                    roll, pitch, yaw,
                    _state.rotation_matrix);
        }

        int Simulator::check_failure(){
            if (_state.global_position.norm() > _fail_max_range){
                std::cerr << "The UAV exits the valid zone" << std::endl;
                return -1;
            }
            if (_state.body_angular_velocity.norm() > _fail_max_angular_velocity){
                std::cerr << "The UAV has too large angualr velocity to recover"
                    << std::endl;
                return -1;
            }
            if (_state.global_velocity.norm() > _fail_max_velocity){
                std::cerr << "The UAV has too large velocity to recover" << std::endl;
                return -1;
            }
            return 0;
        }

        int Simulator::get_config(const char *filename){
            try{
                boost::property_tree::ptree pt;
                boost::property_tree::read_xml(filename, pt);
                _quality = pt.get<double>("simulator.quality");
                Eigen::Matrix3f inertia;
                inertia(0, 0) = pt.get<double>("simulator.inertia.xx");
                inertia(0, 1) = pt.get<double>("simulator.inertia.xy");
                inertia(0, 2) = pt.get<double>("simulator.inertia.xz");
                inertia(1, 0) = pt.get<double>("simulator.inertia.xy");
                inertia(1, 1) = pt.get<double>("simulator.inertia.yy");
                inertia(1, 2) = pt.get<double>("simulator.inertia.yz");
                inertia(2, 0) = pt.get<double>("simulator.inertia.xz");
                inertia(2, 1) = pt.get<double>("simulator.inertia.yz");
                inertia(2, 2) = pt.get<double>("simulator.inertia.zz");
                _inverse_inertia = inertia.inverse();
                _drag_coeff_momentum = Eigen::Matrix3f::Identity();
                _drag_coeff_force = Eigen::Matrix3f::Identity();
                _drag_coeff_momentum(0, 0) = pt.get<double>("simulator.drag.m_xx");
                _drag_coeff_momentum(1, 1) = pt.get<double>("simulator.drag.m_yy");
                _drag_coeff_momentum(2, 2) = pt.get<double>("simulator.drag.m_zz");
                _drag_coeff_force(0, 0) = pt.get<double>("simulator.drag.f_xx");
                _drag_coeff_force(1, 1) = pt.get<double>("simulator.drag.f_yy");
                _drag_coeff_force(2, 2) = pt.get<double>("simulator.drag.f_zz");
                _gravity_center(0) = pt.get<double>("simulator.gravity_center.x");
                _gravity_center(1) = pt.get<double>("simulator.gravity_center.y");
                _gravity_center(2) = pt.get<double>("simulator.gravity_center.z");
                _thrust_coeff_ct_0 = pt.get<double>("simulator.thrust.CT.0");
                _thrust_coeff_ct_1 = pt.get<double>("simulator.thrust.CT.1");
                _thrust_coeff_ct_2 = pt.get<double>("simulator.thrust.CT.2");
                _thrust_coeff_mm = pt.get<double>("simulator.thrust.Mm");
                _thrust_coeff_jm = pt.get<double>("simulator.thrust.Jm");
                _thrust_coeff_phi = pt.get<double>("simulator.thrust.phi");
                _thrust_coeff_ra = pt.get<double>("simulator.thrust.RA");

                _fail_max_velocity = pt.get<double>("simulator.fail.velocity");
                _fail_max_range = pt.get<double>("simulator.fail.range");
                _fail_max_angular_velocity = pt.get<double>("simulator.fail.w");

                _propeller_coord(0, 0) = pt.get<double>("simulator.propeller.1.x");
                _propeller_coord(0, 1) = pt.get<double>("simulator.propeller.1.y");
                _propeller_coord(0, 2) = pt.get<double>("simulator.propeller.1.z");

                _propeller_coord(1, 0) = pt.get<double>("simulator.propeller.2.x");
                _propeller_coord(1, 1) = pt.get<double>("simulator.propeller.2.y");
                _propeller_coord(1, 2) = pt.get<double>("simulator.propeller.2.z");

                _propeller_coord(2, 0) = pt.get<double>("simulator.propeller.3.x");
                _propeller_coord(2, 1) = pt.get<double>("simulator.propeller.3.y");
                _propeller_coord(2, 2) = pt.get<double>("simulator.propeller.3.z");

                _propeller_coord(3, 0) = pt.get<double>("simulator.propeller.4.x");
                _propeller_coord(3, 1) = pt.get<double>("simulator.propeller.4.y");
                _propeller_coord(3, 2) = pt.get<double>("simulator.propeller.4.z");

                _max_voltage = pt.get<double>("simulator.electric.max_voltage");
                _min_voltage = pt.get<double>("simulator.electric.min_voltage");

                _precision = pt.get<double>("simulator.precision");
                _configured = true;
            }
            catch (std::exception& error){
                std::cerr << "Error in loading configuration" << std::endl;
                std::cerr << error.what() << std::endl;
                return -1;
            }
            return 0;
        }

        int Simulator::run_internal(ActionU &act){
            if (not _configured){
                return -1;
            }
            // update propellar state
            Eigen::Vector3f prop_force(0.0, 0.0, 0.0);
            Eigen::Vector3f prop_torque(0.0, 0.0, 0.0);
            std::vector<double> me;
            me.resize(4);
            for (int i = 0; i < 4; i++){
                double eff_act = act(i);
                if (act(i) > _max_voltage){
                    eff_act = _max_voltage;
                }
                else if (act(i) < _min_voltage){
                    eff_act = _min_voltage;
                }
                me[i] = _thrust_coeff_phi / _thrust_coeff_ra *
                    (eff_act - _thrust_coeff_phi
                     * _state.propeller_angular_velocity(i));
                double d_propeller_angular_velocity =
                    1.0 / _thrust_coeff_jm * (me[i] - _thrust_coeff_mm);

                // update the propellar angular velocity
                double w_m = _state.propeller_angular_velocity(i) +
                    _precision * d_propeller_angular_velocity;
                double l_m = _propeller_coord.row(i).norm();
                Eigen::Vector3f body_velocity = _coordinate_converter_to_body
                    * _state.global_velocity;
                double v_1 = body_velocity(2) +
                    (_state.body_angular_velocity.cross(_propeller_coord.row(i)) * l_m)(2);
                double sign = (v_1 > 0.0) ? 1.0 : -1.0;

                // calculate thrust
                double thrust = _thrust_coeff_ct_0 * w_m * w_m +
                    _thrust_coeff_ct_1 * w_m * v_1 +
                    _thrust_coeff_ct_2 * v_1 * v_1 * sign;
                _state.propeller_angular_velocity(i) = w_m;

                prop_force(2) += thrust;
                prop_torque += - Eigen::Vector3f(0.0, 0.0, thrust).cross(
                        _propeller_coord.row(i));
            }
            prop_torque(2) += - me[0] + me[1] - me[2] + me[3];

            // drag force
            Eigen::Vector3f f_drag = - _state.global_velocity.norm() * _drag_coeff_force
                * _coordinate_converter_to_body * _state.global_velocity;

            // drag torque
            Eigen::Vector3f t_drag = - _state.body_angular_velocity.norm()
                * _drag_coeff_momentum * _state.body_angular_velocity;
            // gravity force
            Eigen::Vector3f grav(0.0, 0.0, - GRAVITY_ACCELERATION);
            Eigen::Vector3f f_grav =  _coordinate_converter_to_body * grav * _quality;
            // gravity torque
            Eigen::Vector3f t_grav = -f_grav.cross(_gravity_center);

            // calculate summation forces and torques
            Eigen::Vector3f f_all = prop_force + f_grav + f_drag;
            Eigen::Vector3f t_all = prop_torque + t_grav + t_drag;

            // update the state
            _body_acceleration = f_all / _quality;
            Eigen::Vector3f acceleration = _coordinate_converter_to_world
                * _body_acceleration;
            _state.global_position += _state.global_velocity * _precision
                + 0.5 * _precision * _precision * acceleration;
            _state.global_velocity += _precision * acceleration;

            Eigen::Vector3f tmp_angular_velocity = _state.body_angular_velocity
                + 0.5 * _precision * _inverse_inertia * t_all;

            Eigen::Matrix3f skew_sym_angular_velocity;

            skew_sym_angular_velocity(0, 0) = 0.0;
            skew_sym_angular_velocity(0, 1) = - tmp_angular_velocity(2);
            skew_sym_angular_velocity(0, 2) = tmp_angular_velocity(1);
            skew_sym_angular_velocity(1, 0) = tmp_angular_velocity(2);
            skew_sym_angular_velocity(1, 1) = 0.0;
            skew_sym_angular_velocity(1, 2) = - tmp_angular_velocity(0);
            skew_sym_angular_velocity(2, 0) = - tmp_angular_velocity(1);
            skew_sym_angular_velocity(2, 1) = tmp_angular_velocity(0);
            skew_sym_angular_velocity(2, 2) = 0.0;
            _state.rotation_matrix += _precision * _state.rotation_matrix
                * skew_sym_angular_velocity;
            _state.body_angular_velocity += _precision * _inverse_inertia * t_all;

            // update coordinate converter
            _coordinate_converter_to_world = _state.rotation_matrix;
            _coordinate_converter_to_body = _state.rotation_matrix.inverse();

            if (check_failure()){
                std::cerr << "The quadrotor returns a failure";
                return -1;
            }
            return 0;
        }

        int Simulator::run_time(ActionU &act, double dt){
            unsigned int times = 0;
            if (not _configured){
                std::cerr << "The simulator is not configured" << std::endl;
                return -1;
            }

            if (_precision < 1.0e-8 or _precision > dt){
                std::cerr << "Inproper parameter, quit" << std::endl;
                return -1;
            }

            times = int(dt / _precision);
            for (unsigned int i = 0; i < times; i++){
                if (run_internal(act) != 0){
                    return -1;
                }
            }
            return 0;
        }

        void Simulator::get_printable_state(PrintableState &state){
            state.g_x = _state.global_position(0);
            state.g_y = _state.global_position(1);
            state.g_z = _state.global_position(2);
            state.g_v_x = _state.global_velocity(0);
            state.g_v_y = _state.global_velocity(1);
            state.g_v_z = _state.global_velocity(2);
            Eigen::Vector3f body_velocity = _coordinate_converter_to_body *
                _state.global_velocity;
            state.b_v_x = body_velocity(0);
            state.b_v_y = body_velocity(1);
            state.b_v_z = body_velocity(2);
            state.w_x = _state.body_angular_velocity(0);
            state.w_y = _state.body_angular_velocity(1);
            state.w_z = _state.body_angular_velocity(2);
            uranus::rotation_to_attitude(
                    _state.rotation_matrix, state.roll, state.pitch, state.yaw);
        }

        // Read output from sensor
        // To be updated: noises not added
        void Simulator::read_sensor(SensorOutput &output){
            Eigen::Vector3f body_velocity = _coordinate_converter_to_body *
                _state.global_velocity;
            output.imu_acc = _body_acceleration + _coordinate_converter_to_body
                * Eigen::Vector3f(0.0, 0.0,  GRAVITY_ACCELERATION);
            output.imu_gyro = _state.body_angular_velocity;
            output.vio = _coordinate_converter_to_body * _state.global_velocity;
        }
};
};
