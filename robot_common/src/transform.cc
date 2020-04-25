#include "transform.h"

namespace robot {
namespace common {

template <typename T>
Eigen::Matrix<T, 3, 1> EulerFromQuaternion(const Eigen::Quaternion<T>& quat) {
  T roll, pitch, yaw;

  T sinr_cosp = 2 * (quat.w() * quat.x() + quat.y() * quat.z());
  T cosr_cosp = 1 - 2 * (quat.x() * quat.x() + quat.y() * quat.y());
  roll = std::atan2(sinr_cosp, cosr_cosp);

  T sinp = 2 * (quat.w() * quat.y() - quat.z() * quat.x());
  if (std::abs(sinp) >= 1)
    pitch = std::copysign(M_PI / 2, sinp);
  else
    pitch = std::asin(sinp);

  T siny_cosp = 2 * (quat.w() * quat.z() + quat.x() * quat.y());
  T cosy_cosp = 1 - 2 * (quat.y() * quat.y() + quat.z() * quat.z());
  yaw = std::atan2(siny_cosp, cosy_cosp);

  return Eigen::Matrix<T, 3, 1>(roll, pitch, yaw);
}

template <typename T>
T GetYaw(const Eigen::Quaternion<T>& quat) {
  Eigen::Matrix<T, 3, 1> direction = quat * Eigen::Matrix<T, 3, 1>::UnitX();
  return static_cast<T>(std::atan2(direction.y(), direction.x()));
}

}  // namespace common
}  // namespace robot
