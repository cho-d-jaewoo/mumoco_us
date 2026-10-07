#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sstream>

#include <franka/exception.h>
#include <franka/gripper.h>



/**
 * Opening and closing the robot gripper
 * sending "o" opens the gripper;
 * sending "c" closes the gripper
 * also prints tempurature for safety monitoring
 * Dylan Losey, September 2020 - July 2024
 */



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

  int port = 8081;

  try {

    franka::Gripper gripper("172.16.0.2");
    franka::GripperState gripper_state;
    gripper.stop();
    int sock = connect2control(port);
    int count = 0;
    bool is_open;

    while (true) {

      int open_gripper;
      char buffer[20] = {0};

      while (true) {
        gripper_state = gripper.readOnce();
        is_open = gripper_state.width > 0.0725;


        count++;
        uint16_t temperature = gripper_state.temperature;
        if (count % 10 < 1) {
          std::cout << "my temperature is: " << temperature << std::endl;
          std::cout << "is_open: " << is_open << std::endl;
          std::cout << "gripper_width:" << gripper_state.width << std::endl;
        }

        int valread = read(sock, buffer, 20);
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
          std::string substr;
          getline(ss, substr, ',');
          if (substr == "o") {
            open_gripper = 1;
          } else if (substr == "c") {
            open_gripper = 0;
          }
          break;
        }
      }

      if (is_open == 0 && open_gripper == 1) {
        gripper.move(0.075, 0.1);
        gripper.stop();
      } else if (is_open == 1 && open_gripper == 0) {
        gripper.grasp(0.0, 0.2, 5, 0.1, 0.1);
        //gripper.grasp(0.0, 0.2, 15, 0.1, 0.1);
      }
    }

  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;

}
