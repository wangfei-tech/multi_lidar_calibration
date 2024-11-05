#include "multi_lidar_calibration.h"
#include <Eigen/Dense>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>
#include <chrono>

MultiLidarCalibration::MultiLidarCalibration(ros::NodeHandle &n) : nh_(n)
{
    ROS_INFO_STREAM("\033[1;32m----> Multi Lidar Calibration Use ICP...\033[0m");

    nh_.param<std::string>("/multi_lidar_calibration_node/source_lidar_topic", source_lidar_topic_str_, "/sick_back/scan");
    nh_.param<std::string>("/multi_lidar_calibration_node/target_lidar_topic", target_lidar_topic_str_, "/sick_front/scan");
    nh_.param<std::string>("/multi_lidar_calibration_node/source_lidar_frame", source_lidar_frame_str_, "sub_laser_link");
    nh_.param<std::string>("/multi_lidar_calibration_node/target_lidar_frame", target_lidar_frame_str_, "main_laser_link");
    nh_.param<std::string>("/multi_lidar_calibration_node/base_link", source_frame_str_, "base_link");
    nh_.param<float>("/multi_lidar_calibration_node/icp_score", icp_score_, 5.5487);
    nh_.param<float>("/multi_lidar_calibration_node/fitness_score", fitness_score_, 1.0);
    nh_.param<float>("/multi_lidar_calibration_node/main_to_base_transform_x", main_to_base_transform_x_, 0.352);
    nh_.param<float>("/multi_lidar_calibration_node/main_to_base_transform_y", main_to_base_transform_y_, 0.224);
    nh_.param<float>("/multi_lidar_calibration_node/main_to_base_transform_roll", main_to_base_transform_roll_, -3.1415926);

    nh_.param<float>("/multi_lidar_calibration_node/main_to_back_transform_x", main_to_back_transform_x_, 0.352);
    nh_.param<float>("/multi_lidar_calibration_node/main_to_back_transform_y", main_to_back_transform_y_, 0.224);
    nh_.param<float>("/multi_lidar_calibration_node/main_to_back_transform_roll", main_to_back_transform_roll_, -3.1415926);
    nh_.param<float>("/multi_lidar_calibration_node/main_to_back_transform_yaw", main_to_back_transform_yaw_, 2.35619);

    // 发布转换后的激光点云
    final_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/final_point_cloud", 10);

    // 订阅前后激光话题
    scan_front_subscriber_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, target_lidar_topic_str_, 1);
    scan_back_subscriber_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, source_lidar_topic_str_, 1);
    scan_synchronizer_ = new message_filters::Synchronizer<SyncPolicyT>(SyncPolicyT(10), *scan_front_subscriber_, *scan_back_subscriber_);//Synchronize topics
    scan_synchronizer_->registerCallback(boost::bind(&MultiLidarCalibration::ScanCallBack, this, _1, _2));

    // 参数赋值
    is_first_run_ = true;

    // 在front_laser_link下back_laser_link的坐标
    transform_martix_ = Eigen::Matrix4f::Identity(); //4 * 4 齐次坐标
    // 在base_link坐标系下front_laser_link的坐标
    front_to_base_link_ = Eigen::Matrix4f::Identity();
    // 在base_link坐标系下back_laser_link的坐标
    back_to_base_link_ = Eigen::Matrix4f::Identity();

    //点云指针赋值
    main_scan_pointcloud_ = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>(new pcl::PointCloud<pcl::PointXYZ>());
    sub_scan_pointcloud_ = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>(new pcl::PointCloud<pcl::PointXYZ>());
    final_registration_scan_ = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>(new pcl::PointCloud<pcl::PointXYZ>());
    // 使用在front_laser_link下back_laser_link的坐标，把back_laser_link下的激光转换到front_laser_link下
    sub_scan_pointcloud_init_transformed_ = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>(new pcl::PointCloud<pcl::PointXYZ>());
}

MultiLidarCalibration::~MultiLidarCalibration() {}

/**
 * @brief 获取激光雷达间的坐标变换(机械外参)
 * 
 * @param transform_martix_ 激光雷达间的转换矩阵
 * @param front_to_base_link_ 在front_laser_link下back_laser_link的坐标
 */
void MultiLidarCalibration::GetFrontLasertoBackLaserTf()
{
    tf2_ros::Buffer buffer;
    tf2_ros::TransformListener tfl(buffer);

    ros::Time time = ros::Time::now();
    ros::Duration timeout(0.1);

    geometry_msgs::TransformStamped tfGeom;
    geometry_msgs::TransformStamped transformStamped;
    try
    {
        // source_lidar_frame_str_: back_laser ||||  target_lidar_frame_str_: front_laser
        tfGeom = buffer.lookupTransform(source_lidar_frame_str_, target_lidar_frame_str_, ros::Time(0), ros::Duration(3.0));
    }
    catch (tf2::TransformException &e)
    {
        ROS_ERROR_STREAM("tf2::TransformException &e: "<<e.what());
        ROS_ERROR_STREAM("Lidar Transform Error ... ");
    }
    try
    {
        transformStamped = buffer.lookupTransform(source_frame_str_, target_lidar_frame_str_, ros::Time(0));
        T_base_to_front = transformToEigenMatrix(transformStamped);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
    }
    // tf2矩阵转换成Eigen::Matrix4f
    Eigen::Quaternionf qw_(tfGeom.transform.rotation.w, tfGeom.transform.rotation.x, tfGeom.transform.rotation.y, tfGeom.transform.rotation.z); //tf 获得的四元数
    qw = qw_;
    Eigen::Vector3f qt_(tfGeom.transform.translation.x, tfGeom.transform.translation.y, tfGeom.transform.translation.z);                        //tf获得的平移向量
    qt = qt_;
    transform_martix_.block<3, 3>(0, 0) = qw.toRotationMatrix();
    transform_martix_.block<3, 1>(0, 3) = qt;
    ROS_INFO_STREAM("back_laser_link in front_laser_link matrix=\n"
                    << transform_martix_);
    // 绝对标定的前向激光到base_link的坐标转换
    // 前激光雷达到baselink的坐标系转换
    Eigen::Vector3f rpy_front(main_to_base_transform_roll_, 0, main_to_base_transform_yaw_);
    Eigen::Matrix3f R_front;
    R_front = Eigen::AngleAxisf(rpy_front[0], Eigen::Vector3f::UnitX()) *
              Eigen::AngleAxisf(rpy_front[1], Eigen::Vector3f::UnitY()) *
              Eigen::AngleAxisf(rpy_front[2], Eigen::Vector3f::UnitZ());
    Eigen::Vector3f t_front(main_to_base_transform_x_, main_to_base_transform_y_, 0.0);
    front_to_base_link_.block<3, 3>(0, 0) = R_front;
    front_to_base_link_.block<3, 1>(0, 3) = t_front;
    ROS_INFO_STREAM("front_laser_link in base_link matrix=\n"
                    << front_to_base_link_);
    // 绝对标定的后向激光到base_link的坐标转换
    // 后激光雷达到baselink的坐标系转换
    Eigen::Vector3f rpy_back(main_to_back_transform_roll_, 0, main_to_back_transform_yaw_);
    Eigen::Matrix3f R_back;
    R_back = Eigen::AngleAxisf(rpy_back[0], Eigen::Vector3f::UnitX()) *
             Eigen::AngleAxisf(rpy_back[1], Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(rpy_back[2], Eigen::Vector3f::UnitZ());
    Eigen::Vector3f t_back(main_to_back_transform_x_, main_to_back_transform_y_, 0.0);
    back_to_base_link_.block<3, 3>(0, 0) = R_back;
    back_to_base_link_.block<3, 1>(0, 3) = t_back;
    ROS_INFO_STREAM("back_laser_link in base_link matrix=\n"
                    << back_to_base_link_);
}

/**
  * @brief 激光雷达发布点云
  * @param in_cloud_to_publish_ptr 输入icp转换后的激光点云数据
  */
void MultiLidarCalibration::PublishCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud_to_publish_ptr)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
    cloud_msg.header.frame_id = target_lidar_frame_str_;
    final_point_cloud_pub_.publish(cloud_msg);
}

/**
 * @brief 激光雷达消息类型转换 sensor_msg::Laser to pcl::PointCloud<pcl::PointXYZ>
 * 
 * @param scan_msg 输入sensor_msgs
 * @return pcl::PointCloud<pcl::PointXYZ> 输出pcl格式点云
 */
pcl::PointCloud<pcl::PointXYZ> MultiLidarCalibration::ConvertScantoPointCloud(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud_points;
    pcl::PointXYZ points;

    for (int i = 0; i < scan_msg->ranges.size(); ++i)
    {
        float range = scan_msg->ranges[i];
        if (!std::isfinite(range))
        {
            continue;
        }

        if (range > scan_msg->range_min && range < scan_msg->range_max)
        {
            float angle = scan_msg->angle_min + i * scan_msg->angle_increment;
            points.x = range * cos(angle);
            points.y = range * sin(angle);
            points.z = 0.0;
            cloud_points.push_back(points);
        }
    }
    return cloud_points;
}

/**
 * @brief 多个激光雷达数据同步
 * 
 * @param in_main_scan_msg 前激光雷达topic 1
 * @param in_sub_scan_msg 后激光雷达topic 2
 */
void MultiLidarCalibration::ScanCallBack(const sensor_msgs::LaserScan::ConstPtr &in_main_scan_msg, const sensor_msgs::LaserScan::ConstPtr &in_sub_scan_msg)
{
    main_scan_pointcloud_ = ConvertScantoPointCloud(in_main_scan_msg).makeShared();
    sub_scan_pointcloud_ = ConvertScantoPointCloud(in_sub_scan_msg).makeShared();
}

/**
 * @brief 两个激光雷达数据进行icp匹配
 * @example： 1.引入必要文件 2.创建源点云 3.设置icp对象 4.设置输入点云 5.执行配准 6.检查配准是否成功 7.使用最终变换 
 */
bool MultiLidarCalibration::ScanRegistration()
{
    if (0 == main_scan_pointcloud_->points.size() || 0 == sub_scan_pointcloud_->points.size())
    {
        return false;
    }

    // Initialize the point cloud rotation，back_link to front_link
    pcl::transformPointCloud(*sub_scan_pointcloud_, *sub_scan_pointcloud_init_transformed_, transform_martix_);

    //LiDAR uses mechanical external parameters to rotate

    // Maximum Euclidean distance difference
    icp_.setMaxCorrespondenceDistance(0.1);
    // Iteration threshold, when the difference between the current transformation matrix and the current iteration matrix is ​​less than the threshold,
    // it is considered converged
    icp_.setTransformationEpsilon(1e-10);
    // The iteration stops when the mean square error is less than the threshold
    icp_.setEuclideanFitnessEpsilon(0.005);
    // Maximum number of iterations
    icp_.setMaximumIterations(100);

    icp_.setInputSource(sub_scan_pointcloud_init_transformed_);

    icp_.setInputTarget(main_scan_pointcloud_);
    // matching
    icp_.align(*final_registration_scan_);
    // end or not
    std:cout<<"hasConverged: "<<icp_.hasConverged()<<"getFitnessScore():"<<icp_.getFitnessScore()<<std::endl;
    if (icp_.hasConverged() == false && icp_.getFitnessScore() > fitness_score_)//TEST: Expected scores are usually around 0.01 to 0.1.
    {
        ROS_WARN_STREAM("Not Converged ... ");
        return false;
    }
    return true;
}

/**
 * @brief 打印结果 
 * 
 */
void MultiLidarCalibration::PrintResult()
{
    if (icp_.getFitnessScore() > icp_score_)
    {
        ROS_WARN_STREAM("icp_.getFitnessScore(): "<<icp_.getFitnessScore());
        return;
    }
    // 前激光到base_link的坐标变换
    Eigen::Matrix3f R1 = front_to_base_link_.block<3, 3>(0, 0);
    Eigen::Vector3f t1 = front_to_base_link_.block<3, 1>(0, 3);
    ROS_INFO_STREAM("front_laser to base_link R1 : \n"<< R1);
    ROS_INFO_STREAM("front_laser to base_link t1 : \n"<< t1);
        
    // 后激光到base_link的坐标变换
    Eigen::Matrix3f R2 = back_to_base_link_.block<3, 3>(0, 0);
    Eigen::Vector3f t2 = back_to_base_link_.block<3, 1>(0, 3);
    ROS_INFO_STREAM("back_laser to base_link R1 : \n"<< R2);
    ROS_INFO_STREAM("back_laser to base_link t1 : \n"<< t2);

    // icp变换矩阵的含义：back_laser在经过tf变换后与front_laser匹配后得到的增量变换
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T = icp_.getFinalTransformation();
    Eigen::Matrix3f R3 = T.block<3, 3>(0, 0);
    Eigen::Vector3f t3 = T.block<3, 1>(0, 3);
    ROS_INFO_STREAM("back_laser to front_laser R3 : \n"<< R3);
    ROS_INFO_STREAM("back_laser to front_laser t3 : \n"<< t3);

    // 前激光雷达坐标系下后雷达的坐标位置,两个激光对称放置
    Eigen::Matrix3f R4 = qw.toRotationMatrix();
    Eigen::Vector3f t4 = qt;
    ROS_INFO_STREAM("front_laser to back_laser R4 : \n"<< R4);
    ROS_INFO_STREAM("front_laser to back_laser t4 : \n"<< t4);

    // // 变换结果是以base_link坐标系下的后激光雷达的坐标
    // Eigen::Matrix3f R5 = R4 * R1 * R3;
    // Eigen::Vector3f t5 = R1 * t3 + t1 + t4;

    Eigen::Matrix4f T_front_to_back_tf = transform_martix_.inverse();

    // 获取 ICP 结果 T_back_to_front_icp
    Eigen::Matrix4f T_back_to_front_icp = T; //ICP 计算的结果

    //计算 T_base_to_back
    Eigen::Matrix4f T_base_to_back = T_base_to_front * T_front_to_back_tf * T_back_to_front_icp;

    decomposeTransform(T_base_to_back);
}

/**
 * @brief  点云可视化
 * 
 */
void MultiLidarCalibration::View()
{
    // 点云可视化
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0); //背景色设置

    // 显示源点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(main_scan_pointcloud_, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(main_scan_pointcloud_, source_color, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");

    // 显示目标点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(sub_scan_pointcloud_, 255, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(sub_scan_pointcloud_, target_color, "target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

    // 显示变换后的源点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(final_registration_scan_, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(final_registration_scan_, source_trans_color, "source trans");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source trans");

    // 保存变换结果
    pcl::io::savePLYFile("final_registration_scan.pcd", *final_registration_scan_, false);
    viewer->spin();
}
/**
 * @brief 变换矩阵提取 旋转角与平移向量
 */
void MultiLidarCalibration::decomposeTransform(const Eigen::Matrix4f &T)
{
    // // 提取平移向量
    Eigen::Vector3f translation = T.block<3, 1>(0, 3);
    std::cout << "Translation: \n"
              << translation << std::endl;

    // 提取旋转矩阵
    Eigen::Matrix3f rotation = T.block<3, 3>(0, 0);

    // 计算欧拉角 (ZYX 顺序)
    Eigen::Vector3f euler_angles = rotation.eulerAngles(2, 1, 0); // yaw, pitch, roll
    std::cout << "Euler angles (yaw, pitch, roll): \n"
              << (euler_angles)/M_PI*180.0 << std::endl;
}
/**
 * @brief transfrom形式转换成Eigen矩阵的形式
 */
Eigen::Matrix4f MultiLidarCalibration::transformToEigenMatrix(const geometry_msgs::TransformStamped &transformStamped)
{
    Eigen::Affine3f affine = tf2::transformToEigen(transformStamped.transform).cast<float>();
    return affine.matrix();
}
/**
 * @brief 运行主函数
 * 
 */
void MultiLidarCalibration::Run()
{
    if (is_first_run_)
    {
        GetFrontLasertoBackLaserTf();
        is_first_run_ = false;
        return;
    }

    // 进行icp匹配，匹配失败返回
    if (!ScanRegistration())
    {
        ROS_ERROR_STREAM("ICP matching fails!!!");
        return;
    }

    PublishCloud(final_registration_scan_);

    PrintResult();

    // View(); 一般不要放开这个
}
