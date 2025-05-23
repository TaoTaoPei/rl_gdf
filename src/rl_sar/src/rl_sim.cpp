#include "rl_sim.hpp"

// #define PLOT
// #define CSV_LOGGER

RL_Sim::RL_Sim()
    : rclcpp::Node("rl_sim_node")
{
    this->ros_namespace = this->get_namespace();
    std::cout << LOGGER::INFO << "ros_namespace: " << this->ros_namespace << std::endl; // "/"

    // get params from param_node 传建客户端监听 param_node
    // *********************** 从 param_node 接受 两个参数****************************
    /**/
    param_client = this->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout << LOGGER::ERROR << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            return;
        }
        std::cout << LOGGER::WARNING << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};

    // Use a timeout for the future
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(this->get_node_base_interface(), future, std::chrono::seconds(5));

    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        std::cout << LOGGER::INFO << "Get param values.size() = " << result->values.size() << std::endl;
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
        }
        else
        {
            this->robot_name = result->values[0].string_value;
            this->gazebo_model_name = result->values[1].string_value;
            for(int i = 0; i < result->values.size(); i++){
                std::cout << LOGGER::INFO << "result->values[" << i << "] :" << "" << result->values[i].string_value << std::endl;
            }
            std::cout << LOGGER::INFO << "Get param robot_name: " << this->robot_name << std::endl;
            std::cout << LOGGER::INFO << "Get param gazebo_model_name: " << this->gazebo_model_name << std::endl;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "Failed to call param_node service" << std::endl;
    }
    /*
    if(true){
        this->robot_name = "a1_isaacsim";
        this->gazebo_model_name = "a1_gazebo";
    }
    */

    // read params from yaml
    this->ReadYaml(this->robot_name);
    for (std::string &observation : this->params.observations)
    {
        if (observation == "ang_vel")
        {
            observation = "ang_vel_body";
        }
    }

    if (this->params.framework == "isaacsim") {
        reorder_index = torch::tensor(new_order, torch::dtype(torch::kLong));
    }
    // init rl
    torch::autograd::GradMode::set_enabled(false);
    if (this->params.observations_history.size() != 0)
    {
        this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size()); // [1, 45, 6]
    }
    this->robot_command_publisher_msg.motor_command.resize(this->params.num_of_dofs); // 初始化电机cmd和电机state大小
    this->robot_state_subscriber_msg.motor_state.resize(this->params.num_of_dofs);
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
    running_state = STATE_RL_RUNNING; // 设置机器人初始状态

    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + this->robot_name + "/" + this->params.model_name;
    this->model = torch::jit::load(model_path);

    // publisher
    this->robot_command_publisher = this->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS()); // 控制器接口

    // subscriber
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {
            this->CmdvelCallback(msg); // 速度控制指令回调
        }
    );
    this->gazebo_imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {
            this->GazeboImuCallback(msg); // IMU数据回调
        }
    );
    this->robot_state_subscriber = this->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "robot_joint_controller/state", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {
            this->RobotStateCallback(msg); // 机器人状态回调
        }
    );

    // service
    this->gazebo_set_model_state_client = this->create_client<gazebo_msgs::srv::SetModelState>("/gazebo/set_model_state");
    this->gazebo_pause_physics_client = this->create_client<std_srvs::srv::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = this->create_client<std_srvs::srv::Empty>("/gazebo/unpause_physics");

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<double> *state)
{
    if (this->params.framework == "isaacgym")
    {
        state->imu.quaternion[3] = this->gazebo_imu.orientation.w;
        state->imu.quaternion[0] = this->gazebo_imu.orientation.x;
        state->imu.quaternion[1] = this->gazebo_imu.orientation.y;
        state->imu.quaternion[2] = this->gazebo_imu.orientation.z;
    }
    else if (this->params.framework == "isaacsim")
    {
        state->imu.quaternion[0] = this->gazebo_imu.orientation.w;
        state->imu.quaternion[1] = this->gazebo_imu.orientation.x;
        state->imu.quaternion[2] = this->gazebo_imu.orientation.y;
        state->imu.quaternion[3] = this->gazebo_imu.orientation.z;
    }

    state->imu.gyroscope[0] = this->gazebo_imu.angular_velocity.x;
    state->imu.gyroscope[1] = this->gazebo_imu.angular_velocity.y;
    state->imu.gyroscope[2] = this->gazebo_imu.angular_velocity.z;

    state->imu.accelerometer[0] = this->gazebo_imu.linear_acceleration.x;
    state->imu.accelerometer[1] = this->gazebo_imu.linear_acceleration.y;
    state->imu.accelerometer[2] = this->gazebo_imu.linear_acceleration.z;

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[i].q;
        state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[i].dq;
        state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[i].tau_est;
    }
    if(false){
        RCLCPP_ERROR(this->get_logger(), "[%d] %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,     %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", 
            __LINE__,
            state->motor_state.q[0], state->motor_state.q[1], state->motor_state.q[2],
            state->motor_state.q[3], state->motor_state.q[4], state->motor_state.q[5],
            state->motor_state.q[6], state->motor_state.q[7], state->motor_state.q[8],
            state->motor_state.q[9], state->motor_state.q[10], state->motor_state.q[11]);
    }

    /*
        float32 q            # motor current pos (rad)
        float32 dq           # motor current vel (rad/s)
        float32 ddq          # motor current acc (rad/s^2)
        float32 tau_est      # current estimated output torque (N*m)
        float32 cur          # current estimated output cur (N*m)
    */
    if (this->params.framework == "isaacsim") {
        /*  调整位置
            isaacgym:
                FL_hip_joint FL_thigh_joint FL_calf_joint
                FR_hip_joint FR_thigh_joint FR_calf_joint
                RL_hip_joint RL_thigh_joint RL_calf_joint
                RR_hip_joint RR_thigh_joint RR_calf_joint
            isaaclab:
                FR_hip_joint FR_thigh_joint FR_calf_joint
                FL_hip_joint FL_thigh_joint FL_calf_joint
                RR_hip_joint RR_thigh_joint RR_calf_joint
                RL_hip_joint RL_thigh_joint RL_calf_joint
        */

        std::swap_ranges(state->motor_state.q.begin(), state->motor_state.q.begin() + 3, state->motor_state.q.begin() + 3);
        std::swap_ranges(state->motor_state.q.begin() + 6, state->motor_state.q.begin() + 9, state->motor_state.q.begin() + 9);

        std::swap_ranges(state->motor_state.dq.begin(), state->motor_state.dq.begin() + 3, state->motor_state.dq.begin() + 3);
        std::swap_ranges(state->motor_state.dq.begin() + 6, state->motor_state.dq.begin() + 9, state->motor_state.dq.begin() + 9);

        std::swap_ranges(state->motor_state.tau_est.begin(), state->motor_state.tau_est.begin() + 3, state->motor_state.tau_est.begin() + 3);
        std::swap_ranges(state->motor_state.tau_est.begin() + 6, state->motor_state.tau_est.begin() + 9, state->motor_state.tau_est.begin() + 9);
    }
    if(false){
        RCLCPP_ERROR(this->get_logger(), "[%d] %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,     %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
            __LINE__,
            state->motor_state.q[0], state->motor_state.q[1], state->motor_state.q[2],
            state->motor_state.q[3], state->motor_state.q[4], state->motor_state.q[5],
            state->motor_state.q[6], state->motor_state.q[7], state->motor_state.q[8],
            state->motor_state.q[9], state->motor_state.q[10], state->motor_state.q[11]);
    }
}

void RL_Sim::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->robot_command_publisher_msg.motor_command[i].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[i].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[i].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[i].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[i].tau = command->motor_command.tau[i];
    }

    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
}

void RL_Sim::RobotControl()
{
    // std::lock_guard<std::mutex> lock(robot_state_mutex); // TODO will cause thread timeout

    // if (this->control.control_state == STATE_RESET_SIMULATION)
    // {
    //     auto set_model_state_request = std::make_shared<gazebo_msgs::srv::SetModelState::Request>();
    //     set_model_state_request->model_state.model_name = this->gazebo_model_name;
    //     set_model_state_request->model_state.pose.position.z = 1.0;
    //     set_model_state_request->model_state.reference_frame = "world";
    //     this->gazebo_set_model_state_client->async_send_request(set_model_state_request,
    //         [this](rclcpp::Client<gazebo_msgs::srv::SetModelState>::SharedFuture future)
    //         {
    //             if (future.get())
    //             {
    //                 std::cout << LOGGER::INFO << "Reset Simulation" << std::endl;
    //                 this->control.control_state = STATE_WAITING;
    //             }
    //             else
    //             {
    //                 std::cout << LOGGER::WARNING << "Failed to reset simulation" << std::endl;
    //             }
    //         }
    //     );
    // }
    // if (this->control.control_state == STATE_TOGGLE_SIMULATION)
    // {
    //     auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    //     if (simulation_running)
    //     {
    //         this->gazebo_pause_physics_client->async_send_request(empty_request,
    //             [this](rclcpp::Client<std_srvs::srv::Empty>::SharedFuture future)
    //             {
    //                 if (future.get())
    //                 {
    //                     std::cout << LOGGER::INFO << "Simulation Stop" << std::endl;
    //                 }
    //                 else
    //                 {
    //                     std::cout << LOGGER::WARNING << "Failed to toggle simulation" << std::endl;
    //                 }
    //             }
    //         );
    //     }
    //     else
    //     {
    //         this->gazebo_unpause_physics_client->async_send_request(empty_request,
    //             [this](rclcpp::Client<std_srvs::srv::Empty>::SharedFuture future)
    //             {
    //                 if (future.get())
    //                 {
    //                     std::cout << LOGGER::INFO << "Simulation Start" << std::endl;
    //                 }
    //                 else
    //                 {
    //                     std::cout << LOGGER::WARNING << "Failed to toggle simulation" << std::endl;
    //                 }
    //             }
    //         );
    //     }
    //     simulation_running = !simulation_running;
    //     this->control.control_state = STATE_WAITING;
    // }
    if (simulation_running)
    {
        this->motiontime++;
        this->GetState(&this->robot_state);
        this->StateController(&this->robot_state, &this->robot_command);
        this->SetCommand(&this->robot_command);
    }
}

void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->gazebo_imu = *msg;
}

void RL_Sim::CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    this->cmd_vel = *msg;
}

void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;
}

void RL_Sim::RunModel()
{
    // std::lock_guard<std::mutex> lock(robot_state_mutex); // TODO will cause thread timeout

    if (this->running_state == STATE_RL_RUNNING && simulation_running)
    {
        // this->obs.lin_vel NOT USE
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        // this->obs.commands = torch::tensor({{this->cmd_vel.linear.x, this->cmd_vel.linear.y, this->cmd_vel.angular.z}});
        this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        torch::Tensor clamped_actions = this->Forward();

        for (int i : this->params.hip_scale_reduction_indices)
        {
            clamped_actions[0][i] *= this->params.hip_scale_reduction;
        }

        this->obs.actions = clamped_actions; // 更新actions

        // 计算力矩
        torch::Tensor origin_output_torques = this->ComputeTorques(this->obs.actions);
        // this->TorqueProtect(origin_output_torques);
        this->output_torques = torch::clamp(origin_output_torques, -(this->params.torque_limits), this->params.torque_limits);

        // 计算位置
        this->output_dof_pos = this->ComputePosition(this->obs.actions);

        if (this->params.framework == "isaacsim") {
            this->output_torques = this->output_torques.index_select(1, reorder_index);
            this->output_dof_pos = this->output_dof_pos.index_select(1, reorder_index);
        }

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::empty({1, this->params.num_of_dofs}, torch::kFloat);
        for (int i = 0; i < this->params.num_of_dofs; ++i)
        {
            tau_est[0][i] = this->robot_state_subscriber_msg.motor_state[i].tau_est;
        }
        this->CSVLogger(this->output_torques, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Sim::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (this->params.observations_history.size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor(); // 直接将当前帧的观察数据输入模型
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
        plt::subplot(4, 3, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RL_Sim>());
    rclcpp::shutdown();
    return 0;
}
