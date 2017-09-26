#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  Eigen::VectorXd y = z - H_ * x_;
  Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
  Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + K * y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */

  static const double pi = 3.1415926535897;
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);
  VectorXd Hx(3);
  Hx <<
      sqrt(px * px + py * py),
      atan2(py, px),
      (px * vx + py * vy) / sqrt(px * px + py * py);

  Eigen::VectorXd y = z - Hx;
  if (y(1) < -pi) {
      y(1) = y(1) + 2 * pi;
  } else if (y(1) > pi) {
      y(1) = y(1) - 2 * pi;
  }

  MatrixXd Hj = Tools().CalculateJacobian(x_);

  Eigen::MatrixXd S = Hj * P_ * Hj.transpose() + R_;
  Eigen::MatrixXd K = P_ * Hj.transpose() * S.inverse();

  x_ = x_ + K * y;
  long x_size = x_.size();
  static const MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * Hj) * P_;
}
