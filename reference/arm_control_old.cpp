#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sstream>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>



/**
 * Controlling the robot using joint velocity commands;
 * Recording kinesthetic demonstrations on the robot
 * The robot sends back joint position and Jacobian
 * Dylan Losey, September 2020 - July 2024
 */



class VelocityController {

  public:

    VelocityController(franka::Model& model, int socket);
    franka::JointVelocities operator()(const franka::RobotState& robot_state, franka::Duration period);

  private:

    franka::Model* modelPtr;
    double time = 0.0;
    std::array<double, 100> buffer1;
    std::array<double, 100> buffer2;
    std::array<double, 100> buffer3;
    std::array<double, 100> buffer4;
    std::array<double, 100> buffer5;
    std::array<double, 100> buffer6;
    std::array<double, 100> buffer7;
    std::array<double, 100> bufferRate;
    std::array<double, 7> qdot;
    std::array<double, 7> control_input;
    double MovingAverage(std::array<double, 100>& buffer, double input);

    int sock = 0;
    int valread;
    char buffer[200] = {0};
    int steps = 0;

};


VelocityController::VelocityController(franka::Model& model, int socket) {

  modelPtr = &model;

  for (int i = 0; i < 7; i++) {
    control_input[i] = 0.0;
  }
  for (int i = 0; i < 100; i++) {
    buffer1[i] = 0.0;
    buffer2[i] = 0.0;
    buffer3[i] = 0.0;
    buffer4[i] = 0.0;
    buffer5[i] = 0.0;
    buffer6[i] = 0.0;
    buffer7[i] = 0.0;
    bufferRate[i] = 0.0;
  }

  sock = socket;

}


double VelocityController::MovingAverage(std::array<double, 100>& buffer, double input) {

  double filtered_input = 0.0;
  for (int i = 0; i < 100; i++) {
    filtered_input = filtered_input + buffer[i];
  }
  filtered_input = filtered_input / 100.0;
  for (int i = 0; i < 99; i++) {
    buffer[i] = buffer[i + 1];
  }
  buffer[99] = input;
  return filtered_input;

}


franka::JointVelocities VelocityController::operator()(const franka::RobotState& robot_state,franka::Duration period) {

  time += period.toSec();

  std::array<double, 7> joint_position = robot_state.q;
  std::array<double, 42> jacobian = modelPtr->zeroJacobian(franka::Frame::kEndEffector, robot_state);
  double send_rate = robot_state.control_command_success_rate;

  std::string state = "s,";
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(joint_position[i]));
    state.append(",");
  }
  for (int i = 0; i < 42; i++) {
    state.append(std::to_string(jacobian[i]));
    state.append(",");
  }
  char cstr[state.size() + 1];
  std::copy(state.begin(), state.end(), cstr);
  cstr[state.size()] = '\0';

  if (steps % 50 < 1) {
    valread = read(sock, buffer, 200);
    send(sock, cstr, strlen(cstr), 0);
    if (valread > 0) {
      std::stringstream ss(buffer);
      bool first = false;
      while (not first) {
        std::string substr;
        getline(ss, substr, ',');
        if (substr[0] == 's') {
          first = true;
        }
      }
      for (int i = 0; i < 7; i++) {
        std::string substr;
        getline(ss, substr, ',');
        double term = std::stod(substr);
        control_input[i] = term;
      }
    } else {
      for (int i = 0; i < 7; i++) {
        control_input[i] = 0.0;
      }
    }
  }

  double qdot1 = MovingAverage(buffer1, control_input[0]);
  double qdot2 = MovingAverage(buffer2, control_input[1]);
  double qdot3 = MovingAverage(buffer3, control_input[2]);
  double qdot4 = MovingAverage(buffer4, control_input[3]);
  double qdot5 = MovingAverage(buffer5, control_input[4]);
  double qdot6 = MovingAverage(buffer6, control_input[5]);
  double qdot7 = MovingAverage(buffer7, control_input[6]);
  double comm_rate = MovingAverage(bufferRate, send_rate);

  qdot = {{qdot1, qdot2, qdot3, qdot4, qdot5, qdot6, qdot7}};
  franka::JointVelocities velocity(qdot);

  steps = steps + 1;
  if (steps % 1000 < 1) {
    std::cout << "control_command_success_rate: " << comm_rate << std::endl;
  }
  return velocity;

}


void send2control(franka::Model& model, const franka::RobotState& robot_state, int sock) {

  franka::Model* modelPtr;
  modelPtr = &model;

  std::array<double, 7> joint_position = robot_state.q;
  std::array<double, 42> jacobian = modelPtr->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  std::string state = "s,";
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(joint_position[i]));
    state.append(",");
  }
  for (int i = 0; i < 42; i++) {
    state.append(std::to_string(jacobian[i]));
    state.append(",");
  }

  char cstr[state.size() + 1];
  std::copy(state.begin(), state.end(), cstr);
  cstr[state.size()] = '\0';
  send(sock, cstr, strlen(cstr), 0);

}


int connect2control(int port) {

  int sock = 0;
  struct sockaddr_in serv_addr;
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    std::cout << "Socket Creation Error!" << std::endl;
  }
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  if (inet_pton(AF_INET, "172.16.0.3", &serv_addr.sin_addr) <= 0) {
    std::cout << "Invalid Address / Address Not Supported" << std::endl;
  }
  if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    std::cout << "Connection Failed" << std::endl;
  }
  int status = fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK);
  if (status == -1) {
    std::cout << "Failed to Make the Socket Non-Blocking" << std::endl;
  }
  return sock;

}



int main() {

  int port = 8080;

  try {

    franka::Robot robot("172.16.0.2");
    franka::Model model = robot.loadModel();
    int sock = connect2control(port);
    int robot_mode = 0;
    char buffer[200] = {0};

    robot.setCollisionBehavior(
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});
    // robot.setJointImpedance({{6000, 6000, 6000, 6000, 250, 250, 250}});
    robot.setCartesianImpedance({{2000, 2000, 2000, 250, 250, 250}});

    while (true) {
      try {
        int valread = read(sock, buffer, 200);
        if (valread > 0){
          std::stringstream ss(buffer);
          std::string substr;
          bool last = false;
          while (not last) {
            getline(ss, substr, ',');
            if (substr[0] == 'v' || substr[0] == 'd') {
              last = true;
            }
          }
          std::string mode_string = substr.c_str();
          if (mode_string == "v" && robot_mode == 0){
            printf("%s\n", "Robot is in velocity control mode");
            robot_mode = 1;
          }
          else if (mode_string == "d" && robot_mode == 1){
            printf("%s\n", "Robot is in demonstration mode");
            robot_mode = 0;
          }
        }

        if (robot_mode == 1){
          robot.automaticErrorRecovery();
          VelocityController motion_generator(model, sock);
          robot.control(motion_generator, franka::ControllerMode::kCartesianImpedance);
        }
        else if (robot_mode == 0) {
          franka::RobotState current_state = robot.readOnce();
          send2control(model, current_state, sock);
        }

      } catch(...){}
    }

  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;

}
