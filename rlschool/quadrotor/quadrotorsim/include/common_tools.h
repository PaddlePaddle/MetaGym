#ifndef QUADROTOR_COMMON_TOOLS_H
#define QUADROTOR_COMMON_TOOLS_H
#include <Eigen/Dense>
#include <math.h>

namespace quadrotorsim {
void attitude_to_rotation(double roll, double pitch, double yaw,
                          Eigen::Matrix3f &rotation) {
    double c_x = cos(roll);
    double s_x = sin(roll);
    double c_y = cos(pitch);
    double s_y = sin(pitch);
    double c_z = cos(yaw);
    double s_z = sin(yaw);
    rotation(0, 0) = c_y * c_z;
    rotation(0, 1) = s_x * s_y * c_z - c_x * s_z;
    rotation(0, 2) = c_x * s_y * c_z + s_x * s_z;
    rotation(1, 0) = c_y * s_z;
    rotation(1, 1) = s_x * s_y * s_z + c_x * c_z;
    rotation(1, 2) = c_x * s_y * s_z - s_x * c_z;
    rotation(2, 0) = -s_y;
    rotation(2, 1) = s_x * c_y;
    rotation(2, 2) = c_x * c_y;
}

void rotation_to_attitude(Eigen::Matrix3f &rotation, double &roll,
                          double &pitch, double &yaw) {
    roll = atan2(rotation(2, 1), rotation(2, 2));
    pitch = atan2(-rotation(2, 0), sqrt(rotation(2, 1) * rotation(2, 1) +
                                        rotation(2, 2) * rotation(2, 2)));
    yaw = atan2(rotation(1, 0), rotation(0, 0));
}
} // namespace quadrotorsim
#endif
