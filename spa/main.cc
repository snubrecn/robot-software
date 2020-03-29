#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <iostream>

struct Pose {
  Eigen::Vector2d translation;
  Eigen::Rotation2Dd rotation;
};

struct Constraint {
  int source;
  int target;
  Pose relative_pose;
};

template <typename T>
T NormalizeAngleDifference(T difference) {
  const T kPi = T(M_PI);
  while (difference > kPi) difference -= 2. * kPi;
  while (difference < -kPi) difference += 2. * kPi;
  return difference;
}
class SpaCostFunctor {
 public:
  SpaCostFunctor(const Pose& observed)
      : x_(observed.translation.x()),
        y_(observed.translation.y()),
        theta_(observed.rotation.angle()) {}
  ~SpaCostFunctor() {}

  template <typename T>
  bool operator()(const T* const source_x, const T* const source_y,
                  const T* const source_theta, const T* const target_x,
                  const T* const target_y, const T* const target_theta,
                  T* residual) const {
    const T source_cos = cos(*source_theta);
    const T source_sin = sin(*source_theta);
    const T delta_x = *target_x - *source_x;
    const T delta_y = *target_y - *source_y;
    residual[0] = x_ - (source_cos * delta_x + source_sin * delta_y);
    residual[1] = y_ - (source_cos * delta_y - source_sin * delta_x);
    residual[2] =
        NormalizeAngleDifference(theta_ - (*target_theta - *source_theta));
    return true;
  }

 private:
  const double x_;
  const double y_;
  const double theta_;
};

class SpaCostFunctorAnalytic
    : public ceres::SizedCostFunction<3, 1, 1, 1, 1, 1, 1> {
 public:
  SpaCostFunctorAnalytic(const Pose& observed)
      : x_(observed.translation.x()),
        y_(observed.translation.y()),
        theta_(observed.rotation.angle()) {}
  virtual ~SpaCostFunctorAnalytic() {}

  bool Evaluate(const double* const* parameters, double* residuals,
                double** jacobians) const {
    const double source_cos = cos(parameters[2][0]);
    const double source_sin = sin(parameters[2][0]);
    const double delta_x = parameters[3][0] - parameters[0][0];
    const double delta_y = parameters[4][0] - parameters[1][0];

    residuals[0] = x_ - (source_cos * delta_x + source_sin * delta_y);
    residuals[1] = y_ - (source_cos * delta_y - source_sin * delta_x);
    residuals[2] = NormalizeAngleDifference(
        theta_ - (parameters[5][0] - parameters[2][0]));
    if (!jacobians) return true;

    double* grad_source_x = jacobians[0];
    double* grad_source_y = jacobians[1];
    double* grad_source_theta = jacobians[2];
    double* grad_target_x = jacobians[3];
    double* grad_target_y = jacobians[4];
    double* grad_target_theta = jacobians[5];

    if (grad_source_x) {
      grad_source_x[0] = source_cos;
      grad_source_x[1] = -source_sin;
      grad_source_x[2] = 0;
    }
    if (grad_source_y) {
      grad_source_y[0] = source_sin;
      grad_source_y[1] = source_cos;
      grad_source_y[2] = 0;
    }
    if (grad_source_theta) {
      grad_source_theta[0] = source_sin * delta_x - source_cos * delta_y;
      grad_source_theta[1] = source_cos * delta_x + source_sin * delta_y;
      grad_source_theta[2] = 1;
    }
    if (grad_target_x) {
      grad_target_x[0] = -source_cos;
      grad_target_x[1] = source_sin;
      grad_target_x[2] = 0;
    }
    if (grad_target_y) {
      grad_target_y[0] = -source_sin;
      grad_target_y[1] = -source_cos;
      grad_target_y[2] = 0;
    }
    if (grad_target_theta) {
      grad_target_theta[0] = grad_target_theta[1] = 0;
      grad_target_theta[2] = -1;
    }
    return true;
  }

 private:
  const double x_;
  const double y_;
  const double theta_;
};

int main(void) {
  Constraint constraint01, constraint12, constraint20;
  constraint01.source = 0;
  constraint01.target = 1;
  constraint12.source = 1;
  constraint12.target = 2;
  constraint20.source = 2;
  constraint20.target = 0;

  constraint01.relative_pose.translation = Eigen::Vector2d(4.0, 0.0);
  constraint01.relative_pose.rotation = Eigen::Rotation2Dd(M_PI / 2);
  constraint12.relative_pose.translation = Eigen::Vector2d(4.0, 4.0);
  constraint12.relative_pose.rotation = Eigen::Rotation2Dd(M_PI);
  constraint20.relative_pose.translation = Eigen::Vector2d(4.0, 0.0);
  constraint20.relative_pose.rotation = Eigen::Rotation2Dd(M_PI / 2);

  std::vector<Constraint> constraints;
  constraints.push_back(constraint01);
  constraints.push_back(constraint12);
  constraints.push_back(constraint20);

  std::map<int, Pose> poses;

  Pose p0, p1, p2;
  p0.translation = Eigen::Vector2d(0.0, 0.0);
  p0.rotation = Eigen::Rotation2Dd(0.0);
  p1.translation = Eigen::Vector2d(4.3, -0.2);
  p1.rotation = Eigen::Rotation2Dd(1.4208);
  p2.translation = Eigen::Vector2d(-0.5, 4.4);
  p2.rotation = Eigen::Rotation2Dd(-1.3708);

  poses[0] = p0;
  poses[1] = p1;
  poses[2] = p2;

  ceres::Problem problem;
  bool use_analytic_cost = true;
  for (const auto& constraint : constraints) {
    auto& source = poses[constraint.source];
    auto& target = poses[constraint.target];
    if (use_analytic_cost) {
      problem.AddResidualBlock(
          new SpaCostFunctorAnalytic(constraint.relative_pose),
          new ceres::HuberLoss(1.0), &source.translation.x(),
          &source.translation.y(), &source.rotation.angle(),
          &target.translation.x(), &target.translation.y(),
          &target.rotation.angle());
    } else {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<SpaCostFunctor, 3, 1, 1, 1, 1, 1, 1>(
              new SpaCostFunctor(constraint.relative_pose)),
          new ceres::HuberLoss(1.0), &source.translation.x(),
          &source.translation.y(), &source.rotation.angle(),
          &target.translation.x(), &target.translation.y(),
          &target.rotation.angle());
    }
  }
  auto& reference_pose = poses[0];
  problem.SetParameterBlockConstant(&reference_pose.translation.x());
  problem.SetParameterBlockConstant(&reference_pose.translation.y());
  problem.SetParameterBlockConstant(&reference_pose.rotation.angle());
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  bool verbose = true;
  ceres::Solve(options, &problem, &summary);
  if (verbose) {
    std::cout << "used "
              << (use_analytic_cost ? "analytic cost\n" : "autodiff cost\n");
    std::cout << "total time: " << summary.total_time_in_seconds << std::endl;
    std::cout << "num residuals: " << summary.num_residuals << std::endl;
    std::cout << "num parameters: " << summary.num_parameters << std::endl;
    std::cout << "num effective parameters: "
              << summary.num_effective_parameters << std::endl;
    std::cout << "num successful steps: " << summary.num_successful_steps
              << std::endl;
    std::cout << "initial/final costs: " << summary.initial_cost << ", "
              << summary.final_cost << std::endl;
  }

  for (const auto& pair : poses) {
    const auto& pose = pair.second;
    std::cout << "Pose " << pair.first << " is " << pose.translation.x() << ", "
              << pose.translation.y() << ", " << pose.rotation.angle()
              << std::endl;
  }

  return 0;
}
