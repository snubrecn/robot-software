#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <iostream>

typedef std::array<double, 3> Pose;

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

class SpaCostFunctorAnalytic : public ceres::SizedCostFunction<3, 3, 3> {
 public:
  SpaCostFunctorAnalytic(const Pose& observed)
      : x_(observed[0]), y_(observed[1]), theta_(observed[2]) {}
  virtual ~SpaCostFunctorAnalytic() {}

  bool Evaluate(const double* const* parameters, double* residuals,
                double** jacobians) const {
    double const* source = parameters[0];
    double const* target = parameters[1];

    const double source_cos = cos(source[2]);
    const double source_sin = sin(source[2]);
    const double delta_x = target[0] - source[0];
    const double delta_y = target[1] - source[1];

    residuals[0] = x_ - (source_cos * delta_x + source_sin * delta_y);
    residuals[1] = y_ - (source_cos * delta_y - source_sin * delta_x);
    residuals[2] = NormalizeAngleDifference(theta_ - (target[2] - source[2]));

    if (jacobians == NULL) return true;

    double* grad_source = jacobians[0];
    double* grad_target = jacobians[1];

    if (grad_source != NULL) {
      grad_source[0] = -source_cos;
      grad_source[1] = -source_sin;
      grad_source[2] = source_sin * delta_x - source_cos * delta_y;
      grad_source[3] = source_sin;
      grad_source[4] = -source_cos;
      grad_source[5] = source_cos * delta_x + source_sin * delta_y;
      grad_source[6] = grad_source[7] = 0;
      grad_source[8] = -1;
    }
    if (grad_target != NULL) {
      grad_target[0] = source_cos;
      grad_target[1] = source_sin;
      grad_target[3] = -source_sin;
      grad_target[4] = source_cos;
      grad_target[2] = grad_target[5] = grad_target[6] = grad_target[7] = 0;
      grad_target[8] = -1;
    }
    return true;
  }

 private:
  const double x_;
  const double y_;
  const double theta_;
};

class SpaCostFunctor {
 public:
  SpaCostFunctor(const Pose& observed)
      : x_(observed[0]), y_(observed[1]), theta_(observed[2]) {}
  ~SpaCostFunctor() {}

  template <typename T>
  bool operator()(const T* const source, const T* const target,
                  T* residual) const {
    const T source_cos = cos(source[2]);
    const T source_sin = sin(source[2]);
    const T delta_x = target[0] - source[0];
    const T delta_y = target[1] - source[1];
    residual[0] = x_ - (source_cos * delta_x + source_sin * delta_y);
    residual[1] = y_ - (source_cos * delta_y - source_sin * delta_x);
    residual[2] = NormalizeAngleDifference(theta_ - (target[2] - source[2]));
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

  constraint01.relative_pose = {4.0, 0.0, M_PI / 2};
  constraint12.relative_pose = {4.0, 4.0, M_PI};
  constraint20.relative_pose = {4.0, 0.0, M_PI / 2};

  std::vector<Constraint> constraints;
  constraints.push_back(constraint01);
  constraints.push_back(constraint12);
  constraints.push_back(constraint20);

  std::map<int, Pose> poses;

  Pose p0, p1, p2;
  p0 = {0.0, 0.0, 0.0};
  p1 = {4.3, -0.2, 1.4208};
  p2 = {-0.5, 4.4, -1.3708};

  poses[0] = p0;
  poses[1] = p1;
  poses[2] = p2;

  bool use_analytic_cost = true;
  ceres::Problem problem;
  for (const auto& constraint : constraints) {
    if (use_analytic_cost) {
      problem.AddResidualBlock(
          new SpaCostFunctorAnalytic(constraint.relative_pose),
          new ceres::HuberLoss(1.0), poses[constraint.source].data(),
          poses[constraint.target].data());
    } else {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<SpaCostFunctor, 3, 3, 3>(
              new SpaCostFunctor(constraint.relative_pose)),
          new ceres::HuberLoss(1.0), poses[constraint.source].data(),
          poses[constraint.target].data());
    }
  }
  problem.SetParameterBlockConstant(poses[0].data());
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solve(options, &problem, &summary);

  for (const auto& pair : poses) {
    const auto& pose = pair.second;
    std::cout << "Pose " << pair.first << " is " << pose[0] << ", " << pose[1]
              << ", " << pose[2] << std::endl;
  }

  return 0;
}
