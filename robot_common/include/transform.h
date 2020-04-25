#ifndef ROBOT_COMMON_TRANSFORM_H_
#define ROBOT_COMMON_TRANSFORM_H_

#include <Eigen/Dense>

namespace robot {
namespace common {

template <typename T>
class SE3 {
 public:
  using Translation = Eigen::Matrix<T, 3, 1>;
  using Rotation = Eigen::Quaternion<T>;
  SE3() : translation_(Translation::Zero()), rotation_(Rotation::Identity()) {}
  SE3(const Translation& translation, const Rotation& rotation)
      : translation_(translation), rotation_(rotation) {}

  SE3 inverse() const { return SE3(rotation_.inverse() * (-translation_), rotation_.inverse()); }
  static SE3 Identity() { return SE3(); }

  template <typename F>
  SE3<F> cast() const {
    return SE3<F>(translation_.template cast<F>(), rotation_.template cast<F>());
  }

  Translation translation() const { return translation_; }
  Rotation rotation() const { return rotation_; }

 private:
  Translation translation_;
  Rotation rotation_;
};

template <typename T>
class SE2 {
 public:
  using Translation = Eigen::Matrix<T, 2, 1>;
  using Rotation = Eigen::Rotation2D<T>;
  SE2() : translation_(Translation::Zero()), rotation_(Rotation::Identity()) {}
  SE2(const Translation& translation, const Rotation& rotation)
      : translation_(translation), rotation_(rotation) {}

  SE2 inverse() const { return SE2(rotation_.inverse() * (-translation_), rotation_.inverse()); }
  static SE2 Identity() { return SE2(); }

  template <typename F>
  SE3<F> cast() const {
    return SE3<F>(translation_.template cast<F>(), rotation_.template cast<F>());
  }

 private:
  Translation translation_;
  Rotation rotation_;
};

template <typename T>
SE3<T> operator*(const SE3<T>& lhs, const SE3<T>& rhs) {
  return SE3<T>(lhs.translation() + lhs.rotation() * rhs.translation(),
                lhs.rotation() * rhs.rotation());
}

template <typename T>
SE2<T> operator*(const SE2<T>& lhs, const SE2<T>& rhs) {
  return SE2<T>(lhs.translation() + lhs.rotation() * rhs.translation(),
                lhs.rotation() * rhs.rotation());
}

typedef SE3<double> SE3d;
typedef SE3<float> SE3f;
typedef SE2<double> SE2d;
typedef SE2<float> SE2f;

template <typename T>
Eigen::Matrix<T, 3, 1> EulerFromQuaternion(const Eigen::Quaternion<T>& quat);

template <typename T>
T GetYaw(const Eigen::Quaternion<T>& quat);

}  // namespace common
}  // namespace robot

#endif
