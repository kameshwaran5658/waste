-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 22, 2025 at 04:46 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `food_waste_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `activity_log`
--

CREATE TABLE `activity_log` (
  `id` int(11) NOT NULL,
  `admin_id` int(11) NOT NULL,
  `timestamp` datetime DEFAULT current_timestamp(),
  `type` varchar(100) NOT NULL,
  `details` text DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `activity_log`
--

INSERT INTO `activity_log` (`id`, `admin_id`, `timestamp`, `type`, `details`, `ip_address`) VALUES
(1, 2, '2025-05-21 00:10:27', 'Login', 'Admin kamesh logged in.', '127.0.0.1'),
(2, 2, '2025-05-21 00:10:48', 'Attendance', 'Recorded attendance for 2025-05-21 with 300 students.', '127.0.0.1'),
(3, 2, '2025-05-21 00:11:06', 'Attendance', 'Recorded attendance for 2025-05-21 with 300 students.', '127.0.0.1'),
(4, 2, '2025-05-21 00:11:42', 'Attendance', 'Recorded attendance for 2025-05-21 with 200 students.', '127.0.0.1'),
(5, 2, '2025-05-21 00:15:17', 'Attendance', 'Recorded attendance for 2025-05-21 with 300 students.', '127.0.0.1'),
(6, 2, '2025-05-21 00:16:06', 'Menu Edit', 'Edited menu ID 28 for Sunday, Dinner: Idly, Chutney, Sambar, White Rice, Rasam, Pickle.', '127.0.0.1'),
(7, 2, '2025-05-21 00:27:36', 'Logout', 'Admin kamesh logged out.', '127.0.0.1'),
(8, 2, '2025-05-21 00:27:59', 'Login', 'Admin kamesh logged in.', '127.0.0.1'),
(9, 2, '2025-05-21 10:14:20', 'Login', 'Admin kamesh logged in.', '127.0.0.1'),
(10, 2, '2025-05-21 10:16:41', 'Attendance', 'Recorded attendance for 2025-05-21 with 50 students.', '127.0.0.1'),
(11, 2, '2025-05-21 10:23:50', 'Settings Update', 'Notification alerts set to: True.', '127.0.0.1'),
(12, 2, '2025-05-21 10:25:49', 'Attendance', 'Recorded attendance for 2025-05-21 with 100 students.', '127.0.0.1'),
(13, 2, '2025-05-21 10:31:02', 'Attendance', 'Recorded attendance for 2025-05-21 with 100 students.', '127.0.0.1'),
(14, 2, '2025-05-21 10:36:44', 'Attendance', 'Recorded attendance for 2025-05-21 with 350 students.', '127.0.0.1'),
(15, 2, '2025-05-21 19:44:11', 'Attendance', 'Recorded attendance for 2025-05-21 with 80 students.', '127.0.0.1');

-- --------------------------------------------------------

--
-- Table structure for table `admins`
--

CREATE TABLE `admins` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `full_name` varchar(255) DEFAULT NULL,
  `email` varchar(255) NOT NULL,
  `email_alerts` tinyint(1) NOT NULL DEFAULT 1,
  `last_login` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `admins`
--

INSERT INTO `admins` (`id`, `username`, `password_hash`, `full_name`, `email`, `email_alerts`, `last_login`) VALUES
(1, 'admin', '$2b$12$TNEgPv7wUCeGqluboY.qzOR7XZjKCiFcs8XhVniqrl3gZpjpbv.Ae', 'System Administrator', 'admin@example.com', 1, '2025-04-26 06:13:47'),
(2, 'kamesh', '$2b$12$.bFPiDj5C5GM2pGt/1zAnO/s4dGVxcBvdD5Kdt5SYERcd.qk9LVlS', 'KAMESHWARAN', 'kameshwaranking5658@gmail.com', 1, '2025-04-26 06:38:47');

-- --------------------------------------------------------

--
-- Table structure for table `attendance`
--

CREATE TABLE `attendance` (
  `date` date NOT NULL,
  `student_count` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `attendance`
--

INSERT INTO `attendance` (`date`, `student_count`) VALUES
('2025-05-21', 80);

-- --------------------------------------------------------

--
-- Table structure for table `food_history`
--

CREATE TABLE `food_history` (
  `id` int(11) NOT NULL,
  `day` date NOT NULL,
  `meal_type` enum('Breakfast','Lunch','Snack','Dinner') NOT NULL,
  `item_name` varchar(255) NOT NULL,
  `actual_quantity_used` decimal(10,2) NOT NULL,
  `food_waste` decimal(10,2) NOT NULL,
  `student_count` int(11) NOT NULL
) ;

--
-- Dumping data for table `food_history`
--

INSERT INTO `food_history` (`id`, `day`, `meal_type`, `item_name`, `actual_quantity_used`, `food_waste`, `student_count`) VALUES
(1, '2025-05-08', 'Breakfast', 'Pongal', 500.00, 50.00, 800),
(2, '2025-05-08', 'Breakfast', 'Medu Vadai', 900.00, 20.00, 800),
(3, '2025-05-08', 'Breakfast', 'Sambar', 200.00, 20.00, 800),
(4, '0000-00-00', 'Breakfast', 'Pongal', 1200.00, 90.00, 300),
(5, '0000-00-00', 'Breakfast', 'Medu Vadai', 750.00, 60.00, 300),
(6, '0000-00-00', 'Breakfast', 'Sambar', 540.00, 40.00, 300),
(7, '0000-00-00', 'Breakfast', 'Coconut Chutney', 450.00, 30.00, 300),
(8, '0000-00-00', 'Breakfast', 'Tea', 300.00, 20.00, 300),
(9, '0000-00-00', 'Lunch', 'White Rice', 390.00, 30.00, 350),
(10, '0000-00-00', 'Lunch', 'Sambar', 540.00, 40.00, 350),
(11, '0000-00-00', 'Lunch', 'Rasam', 450.00, 35.00, 350),
(12, '0000-00-00', 'Lunch', 'Poriyal', 300.00, 20.00, 350),
(13, '0000-00-00', 'Lunch', 'Kootu', 420.00, 30.00, 350),
(14, '0000-00-00', 'Lunch', 'Papad', 280.00, 20.00, 350),
(15, '0000-00-00', 'Lunch', 'Butter Milk', 330.00, 25.00, 350),
(16, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 350),
(17, '0000-00-00', '', 'Biscuit', 360.00, 30.00, 300),
(18, '0000-00-00', '', 'Tea', 270.00, 18.00, 300),
(19, '0000-00-00', 'Dinner', 'Chapathi', 1680.00, 120.00, 350),
(20, '0000-00-00', 'Dinner', 'Channa Masala', 700.00, 50.00, 350),
(21, '0000-00-00', 'Dinner', 'White Rice', 420.00, 30.00, 350),
(22, '0000-00-00', 'Dinner', 'Rasam', 525.00, 40.00, 350),
(23, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 350),
(24, '0000-00-00', 'Breakfast', 'Idly', 1450.00, 105.00, 290),
(25, '0000-00-00', 'Breakfast', 'Medu Vadai', 720.00, 55.00, 290),
(26, '0000-00-00', 'Breakfast', 'Sambar', 510.00, 38.00, 290),
(27, '0000-00-00', 'Breakfast', 'Chutney', 464.00, 32.00, 290),
(28, '0000-00-00', 'Breakfast', 'Tea', 290.00, 20.00, 290),
(29, '0000-00-00', 'Lunch', 'White Rice', 425.00, 32.00, 340),
(30, '0000-00-00', 'Lunch', 'Kara Kulambu', 680.00, 50.00, 340),
(31, '0000-00-00', 'Lunch', 'Rasam', 510.00, 38.00, 340),
(32, '0000-00-00', 'Lunch', 'Poriyal', 340.00, 25.00, 340),
(33, '0000-00-00', 'Lunch', 'Kootu', 460.00, 35.00, 340),
(34, '0000-00-00', 'Lunch', 'Papad', 340.00, 25.00, 340),
(35, '0000-00-00', 'Lunch', 'Butter Milk', 380.00, 28.00, 340),
(36, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 340),
(37, '0000-00-00', '', 'Sundal', 390.00, 30.00, 270),
(38, '0000-00-00', '', 'Tea', 270.00, 18.00, 270),
(39, '0000-00-00', 'Dinner', 'Parotta', 1550.00, 110.00, 310),
(40, '0000-00-00', 'Dinner', 'Chicken Gravy with piece', 770.00, 60.00, 310),
(41, '0000-00-00', 'Dinner', 'Veg Kuruma', 460.00, 35.00, 310),
(42, '0000-00-00', 'Dinner', 'Gopi Manjurian', 500.00, 40.00, 310),
(43, '0000-00-00', 'Dinner', 'White Rice', 330.00, 25.00, 310),
(44, '0000-00-00', 'Dinner', 'Rasam', 420.00, 32.00, 310),
(45, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 310),
(46, '0000-00-00', 'Breakfast', 'Kal Dosa', 1200.00, 90.00, 300),
(47, '0000-00-00', 'Breakfast', 'Kesari', 600.00, 45.00, 300),
(48, '0000-00-00', 'Breakfast', 'Rava Uppuma', 1050.00, 80.00, 300),
(49, '0000-00-00', 'Breakfast', 'Sambar', 480.00, 35.00, 300),
(50, '0000-00-00', 'Breakfast', 'Chutney', 450.00, 30.00, 300),
(51, '0000-00-00', 'Breakfast', 'Tea', 300.00, 20.00, 300),
(52, '0000-00-00', 'Lunch', 'White Rice', 450.00, 35.00, 360),
(53, '0000-00-00', 'Lunch', 'Mor Kulambu', 680.00, 50.00, 360),
(54, '0000-00-00', 'Lunch', 'Rasam', 510.00, 38.00, 360),
(55, '0000-00-00', 'Lunch', 'Poriyal', 340.00, 25.00, 360),
(56, '0000-00-00', 'Lunch', 'Kootu', 440.00, 33.00, 360),
(57, '0000-00-00', 'Lunch', 'Papad', 340.00, 25.00, 360),
(58, '0000-00-00', 'Lunch', 'Butter Milk', 380.00, 28.00, 360),
(59, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 360),
(60, '0000-00-00', '', 'Valaikai Baji', 420.00, 30.00, 290),
(61, '0000-00-00', '', 'Tea', 290.00, 18.00, 290),
(62, '0000-00-00', 'Dinner', 'Idly', 960.00, 70.00, 320),
(63, '0000-00-00', 'Dinner', 'Tomato Thokku', 576.00, 43.00, 320),
(64, '0000-00-00', 'Dinner', 'Sambar', 480.00, 35.00, 320),
(65, '0000-00-00', 'Dinner', 'White Rice', 368.00, 27.00, 320),
(66, '0000-00-00', 'Dinner', 'Rasam', 448.00, 33.00, 320),
(67, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 320),
(68, '0000-00-00', 'Breakfast', 'Rava Kichadi', 1320.00, 99.00, 330),
(69, '0000-00-00', 'Breakfast', 'Bread', 660.00, 50.00, 330),
(70, '0000-00-00', 'Breakfast', 'Jam', 330.00, 25.00, 330),
(71, '0000-00-00', 'Breakfast', 'Chutney', 495.00, 35.00, 330),
(72, '0000-00-00', 'Breakfast', 'Tea', 330.00, 22.00, 330),
(73, '0000-00-00', 'Lunch', 'White Rice', 437.00, 33.00, 350),
(74, '0000-00-00', 'Lunch', 'Sambar', 680.00, 50.00, 350),
(75, '0000-00-00', 'Lunch', 'Rasam', 510.00, 38.00, 350),
(76, '0000-00-00', 'Lunch', 'Poriyal', 340.00, 25.00, 350),
(77, '0000-00-00', 'Lunch', 'Kootu', 460.00, 35.00, 350),
(78, '0000-00-00', 'Lunch', 'Papad', 340.00, 25.00, 350),
(79, '0000-00-00', 'Lunch', 'Butter Milk', 385.00, 28.00, 350),
(80, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 350),
(81, '0000-00-00', '', 'Bonda', 420.00, 30.00, 300),
(82, '0000-00-00', '', 'Tea', 300.00, 20.00, 300),
(83, '0000-00-00', 'Dinner', 'Variety Rice', 680.00, 50.00, 340),
(84, '0000-00-00', 'Dinner', 'White Rice', 374.00, 28.00, 340),
(85, '0000-00-00', 'Dinner', 'Rasam', 476.00, 35.00, 340),
(86, '0000-00-00', 'Dinner', 'Egg', 680.00, 50.00, 340),
(87, '0000-00-00', 'Dinner', 'Gopi Manjurian', 510.00, 38.00, 340),
(88, '0000-00-00', 'Dinner', 'Brinjal Thalicha', 476.00, 35.00, 340),
(89, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 340),
(90, '0000-00-00', 'Breakfast', 'Kal Dosai', 1120.00, 85.00, 280),
(91, '0000-00-00', 'Breakfast', 'Semiya', 896.00, 67.00, 280),
(92, '0000-00-00', 'Breakfast', 'Sambar', 448.00, 32.00, 280),
(93, '0000-00-00', 'Breakfast', 'Chutney', 420.00, 30.00, 280),
(94, '0000-00-00', 'Breakfast', 'Tea', 280.00, 20.00, 280),
(95, '0000-00-00', 'Lunch', 'White Rice', 400.00, 30.00, 320),
(96, '0000-00-00', 'Lunch', 'Chicken Kulambu', 704.00, 55.00, 320),
(97, '0000-00-00', 'Lunch', 'Rasam', 480.00, 35.00, 320),
(98, '0000-00-00', 'Lunch', 'Poriyal', 320.00, 25.00, 320),
(99, '0000-00-00', 'Lunch', 'Kootu', 368.00, 28.00, 320),
(100, '0000-00-00', 'Lunch', 'Papad', 320.00, 25.00, 320),
(101, '0000-00-00', 'Lunch', 'Butter Milk', 352.00, 26.00, 320),
(102, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 320),
(103, '0000-00-00', '', 'Sundal', 420.00, 30.00, 300),
(104, '0000-00-00', '', 'Tea', 300.00, 20.00, 300),
(105, '0000-00-00', 'Dinner', 'Chapathi', 1550.00, 115.00, 310),
(106, '0000-00-00', 'Dinner', 'Tomato Thokku', 529.00, 38.00, 310),
(107, '0000-00-00', 'Dinner', 'White Rice', 330.00, 25.00, 310),
(108, '0000-00-00', 'Dinner', 'Rasam', 420.00, 32.00, 310),
(109, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 310),
(110, '0000-00-00', 'Breakfast', 'Idly', 1080.00, 80.00, 270),
(111, '0000-00-00', 'Breakfast', 'Medu Vadai', 648.00, 50.00, 270),
(112, '0000-00-00', 'Breakfast', 'Sambar', 405.00, 30.00, 270),
(113, '0000-00-00', 'Breakfast', 'Chutney', 378.00, 28.00, 270),
(114, '0000-00-00', 'Breakfast', 'Tea', 270.00, 20.00, 270),
(115, '0000-00-00', 'Lunch', 'White Rice', 400.00, 30.00, 320),
(116, '0000-00-00', 'Lunch', 'Sambar', 680.00, 50.00, 320),
(117, '0000-00-00', 'Lunch', 'Rasam', 480.00, 35.00, 320),
(118, '0000-00-00', 'Lunch', 'Poriyal', 320.00, 25.00, 320),
(119, '0000-00-00', 'Lunch', 'Kootu', 460.00, 35.00, 320),
(120, '0000-00-00', 'Lunch', 'Papad', 320.00, 25.00, 320),
(121, '0000-00-00', 'Lunch', 'Butter Milk', 352.00, 26.00, 320),
(122, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 320),
(123, '0000-00-00', '', 'Sweet Bonda', 420.00, 30.00, 290),
(124, '0000-00-00', '', 'Tea', 290.00, 20.00, 290),
(125, '0000-00-00', 'Dinner', 'Dosa', 1240.00, 90.00, 310),
(126, '0000-00-00', 'Dinner', 'Sambar', 480.00, 35.00, 310),
(127, '0000-00-00', 'Dinner', 'Chutney', 420.00, 30.00, 310),
(128, '0000-00-00', 'Dinner', 'White Rice', 345.00, 25.00, 310),
(129, '0000-00-00', 'Dinner', 'Rasam', 420.00, 32.00, 310),
(130, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 310),
(131, '0000-00-00', 'Breakfast', 'Poori', 1200.00, 90.00, 300),
(132, '0000-00-00', 'Breakfast', 'Alu Masala', 600.00, 45.00, 300),
(133, '0000-00-00', 'Breakfast', 'Tea', 300.00, 20.00, 300),
(134, '0000-00-00', 'Lunch', 'Chicken Briyani', 700.00, 55.00, 350),
(135, '0000-00-00', 'Lunch', 'Veg Briyani', 625.00, 50.00, 350),
(136, '0000-00-00', 'Lunch', 'White Rice', 350.00, 25.00, 350),
(137, '0000-00-00', 'Lunch', 'Rasam', 510.00, 38.00, 350),
(138, '0000-00-00', 'Lunch', 'Gobi Manjurian', 560.00, 40.00, 350),
(139, '0000-00-00', 'Lunch', 'Onion Raita', 455.00, 34.00, 350),
(140, '0000-00-00', 'Lunch', 'Pickle', 105.00, 7.00, 350),
(141, '0000-00-00', '', 'Groundnut', 360.00, 30.00, 300),
(142, '0000-00-00', '', 'Tea', 300.00, 20.00, 300),
(143, '0000-00-00', 'Dinner', 'Idly', 1280.00, 90.00, 320),
(144, '0000-00-00', 'Dinner', 'Chutney', 480.00, 35.00, 320),
(145, '0000-00-00', 'Dinner', 'Sambar', 480.00, 35.00, 320),
(146, '0000-00-00', 'Dinner', 'White Rice', 352.00, 25.00, 320),
(147, '0000-00-00', 'Dinner', 'Rasam', 448.00, 32.00, 320),
(148, '0000-00-00', 'Dinner', 'Pickle', 105.00, 7.00, 320),
(149, '2025-05-05', 'Breakfast', 'Pongal', 1200.00, 90.00, 300),
(150, '2025-05-05', 'Breakfast', 'Medu Vadai', 750.00, 60.00, 300),
(151, '2025-05-05', 'Breakfast', 'Sambar', 540.00, 40.00, 300),
(152, '2025-05-05', 'Breakfast', 'Coconut Chutney', 450.00, 30.00, 300),
(153, '2025-05-05', 'Breakfast', 'Tea', 300.00, 20.00, 300),
(154, '2025-05-05', 'Lunch', 'White Rice', 390.00, 30.00, 350),
(155, '2025-05-05', 'Lunch', 'Sambar', 540.00, 40.00, 350),
(156, '2025-05-05', 'Lunch', 'Rasam', 450.00, 35.00, 350),
(157, '2025-05-05', 'Lunch', 'Poriyal', 300.00, 20.00, 350),
(158, '2025-05-05', 'Lunch', 'Kootu', 420.00, 30.00, 350),
(159, '2025-05-05', 'Lunch', 'Papad', 280.00, 20.00, 350),
(160, '2025-05-05', 'Lunch', 'Butter Milk', 330.00, 25.00, 350),
(161, '2025-05-05', 'Lunch', 'Pickle', 105.00, 7.00, 350),
(162, '2025-05-05', '', 'Biscuit', 360.00, 30.00, 300),
(163, '2025-05-05', '', 'Tea', 270.00, 18.00, 300),
(164, '2025-05-05', 'Dinner', 'Chapathi', 1680.00, 120.00, 350),
(165, '2025-05-05', 'Dinner', 'Channa Masala', 700.00, 50.00, 350),
(166, '2025-05-05', 'Dinner', 'White Rice', 420.00, 30.00, 350),
(167, '2025-05-05', 'Dinner', 'Rasam', 525.00, 40.00, 350),
(168, '2025-05-05', 'Dinner', 'Pickle', 105.00, 7.00, 350),
(169, '2025-05-06', 'Breakfast', 'Idly', 1450.00, 105.00, 290),
(170, '2025-05-06', 'Breakfast', 'Medu Vadai', 720.00, 55.00, 290),
(171, '2025-05-06', 'Breakfast', 'Sambar', 510.00, 38.00, 290),
(172, '2025-05-06', 'Breakfast', 'Chutney', 464.00, 32.00, 290),
(173, '2025-05-06', 'Breakfast', 'Tea', 290.00, 20.00, 290),
(174, '2025-05-06', 'Lunch', 'White Rice', 425.00, 32.00, 340),
(175, '2025-05-06', 'Lunch', 'Kara Kulambu', 680.00, 50.00, 340),
(176, '2025-05-06', 'Lunch', 'Rasam', 510.00, 38.00, 340),
(177, '2025-05-06', 'Lunch', 'Poriyal', 340.00, 25.00, 340),
(178, '2025-05-06', 'Lunch', 'Kootu', 460.00, 35.00, 340),
(179, '2025-05-06', 'Lunch', 'Papad', 340.00, 25.00, 340),
(180, '2025-05-06', 'Lunch', 'Butter Milk', 380.00, 28.00, 340),
(181, '2025-05-06', 'Lunch', 'Pickle', 105.00, 7.00, 340),
(182, '2025-05-06', '', 'Sundal', 390.00, 30.00, 270),
(183, '2025-05-06', '', 'Tea', 270.00, 18.00, 270),
(184, '2025-05-06', 'Dinner', 'Parotta', 1550.00, 110.00, 310),
(185, '2025-05-06', 'Dinner', 'Chicken Gravy with piece', 770.00, 60.00, 310),
(186, '2025-05-06', 'Dinner', 'Veg Kuruma', 460.00, 35.00, 310),
(187, '2025-05-06', 'Dinner', 'Gopi Manjurian', 500.00, 40.00, 310),
(188, '2025-05-06', 'Dinner', 'White Rice', 330.00, 25.00, 310),
(189, '2025-05-06', 'Dinner', 'Rasam', 420.00, 32.00, 310),
(190, '2025-05-06', 'Dinner', 'Pickle', 105.00, 7.00, 310),
(191, '2025-05-16', 'Breakfast', 'Pongal', 200.00, 10.00, 500),
(192, '2025-05-21', 'Breakfast', 'Pongal', 273.00, 229.00, 273),
(193, '2025-05-21', 'Breakfast', 'Medu Vadai', 263.00, 308.00, 263),
(194, '2025-05-21', 'Breakfast', 'Idly', 81.00, 91.00, 81),
(195, '2025-05-21', 'Breakfast', 'Kesari', 232.00, 201.00, 232),
(196, '2025-05-21', 'Breakfast', 'Rava Uppuma', 221.00, 256.00, 221),
(197, '2025-05-21', 'Breakfast', 'Kal Dosa', 169.00, 179.00, 169),
(198, '2025-05-21', 'Breakfast', 'Rava Kichadi', 272.00, 264.00, 272),
(199, '2025-05-21', 'Breakfast', 'Bread', 180.00, 165.00, 180),
(200, '2025-05-21', 'Breakfast', 'Semiya', 288.00, 327.00, 288),
(201, '2025-05-21', 'Breakfast', 'Poori', 140.00, 127.00, 140),
(202, '2025-05-21', 'Breakfast', 'Alu Masala', 125.00, 133.00, 125),
(203, '2025-05-21', 'Lunch', 'White Rice', 161.00, 142.00, 161),
(204, '2025-05-21', 'Lunch', 'Sambar', 236.00, 195.00, 236),
(205, '2025-05-21', 'Lunch', 'Rasam', 86.00, 86.00, 86),
(206, '2025-05-21', 'Lunch', 'Kara Kulambu', 71.00, 82.00, 71),
(207, '2025-05-21', 'Lunch', 'Mor Kulambu', 124.00, 136.00, 124),
(208, '2025-05-21', 'Lunch', 'Chicken Kulambu', 276.00, 247.00, 276),
(209, '2025-05-21', 'Lunch', 'Chicken Briyani', 170.00, 183.00, 170),
(210, '2025-05-21', 'Lunch', 'Veg Briyani', 55.00, 44.00, 55),
(211, '2025-05-21', 'Lunch', 'Poriyal', 61.00, 52.00, 61),
(212, '2025-05-21', 'Lunch', 'Kootu', 221.00, 258.00, 221),
(213, '2025-05-21', 'Lunch', 'Papad', 188.00, 150.00, 188),
(214, '2025-05-21', 'Lunch', 'Butter Milk', 141.00, 158.00, 141),
(215, '2025-05-21', 'Lunch', 'Pickle', 296.00, 291.00, 296),
(216, '2025-05-21', 'Snack', 'Sundal', 140.00, 135.00, 140),
(217, '2025-05-21', 'Snack', 'Valaikai Baji', 58.00, 67.00, 58),
(218, '2025-05-21', 'Snack', 'Bonda', 140.00, 119.00, 140),
(219, '2025-05-21', 'Snack', 'Sweet Bonda', 188.00, 179.00, 188),
(220, '2025-05-21', 'Snack', 'Groundnut', 117.00, 102.00, 117),
(221, '2025-05-21', 'Dinner', 'Chapathi', 84.00, 71.00, 84),
(222, '2025-05-21', 'Dinner', 'Channa Masala', 102.00, 109.00, 102),
(223, '2025-05-21', 'Dinner', 'Parotta', 234.00, 248.00, 234),
(224, '2025-05-21', 'Dinner', 'Chicken Gravy', 73.00, 72.00, 73),
(225, '2025-05-21', 'Dinner', 'Veg Kuruma', 80.00, 69.00, 80),
(226, '2025-05-21', 'Dinner', 'Gopi Manjurian', 171.00, 199.00, 171),
(227, '2025-05-21', 'Dinner', 'Tomato Thokku', 79.00, 89.00, 79),
(228, '2025-05-21', 'Dinner', 'Dosa', 266.00, 295.00, 266),
(229, '2025-05-21', 'Dinner', 'Variety Rice', 125.00, 108.00, 125),
(230, '2025-05-21', 'Dinner', 'Egg', 284.00, 246.00, 284),
(231, '2025-05-21', 'Dinner', 'Brinjal Thalicha', 60.00, 64.00, 60),
(232, '2025-05-21', 'Dinner', 'Onion Raita', 143.00, 158.00, 143);

-- --------------------------------------------------------

--
-- Table structure for table `ingredients_inventory`
--

CREATE TABLE `ingredients_inventory` (
  `id` int(11) NOT NULL,
  `ingredient_name` varchar(255) NOT NULL,
  `current_stock_kg` decimal(10,2) NOT NULL,
  `last_updated` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `menu`
--

CREATE TABLE `menu` (
  `id` int(11) NOT NULL,
  `day` varchar(10) NOT NULL,
  `meal_type` varchar(20) NOT NULL,
  `items` text NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `menu`
--

INSERT INTO `menu` (`id`, `day`, `meal_type`, `items`, `created_at`) VALUES
(1, 'Monday', 'Breakfast', 'Pongal, Medu Vadai, Sambar, Coconut Chutney, Tea', '2025-05-06 06:03:52'),
(2, 'Monday', 'Lunch', 'White Rice, Sambar, Rasam, Poriyal, Kootu, Papad, Butter Milk, Pickle', '2025-05-06 06:03:52'),
(3, 'Monday', 'Snacks', 'Biscuit, Tea', '2025-05-06 06:03:52'),
(4, 'Monday', 'Dinner', 'Chapathi, Channa Masala, White Rice, Rasam, Pickle', '2025-05-06 06:03:52'),
(5, 'Tuesday', 'Breakfast', 'Idly, Medu Vadai, Sambar, Chutney, Tea', '2025-05-06 06:03:52'),
(6, 'Tuesday', 'Lunch', 'White Rice, Kara Kulambu, Rasam, Poriyal, Kootu, Papad, Butter Milk, Pickle', '2025-05-06 06:03:52'),
(7, 'Tuesday', 'Snacks', 'Sundal, Tea', '2025-05-06 06:03:52'),
(8, 'Tuesday', 'Dinner', 'Parotta, Chicken Gravy with piece, Veg Kuruma, Gopi Manjurian, White Rice, Rasam, Pickle', '2025-05-06 06:03:52'),
(9, 'Wednesday', 'Breakfast', 'Kal Dosa, Kesari, Rava Uppuma, Sambar, Chutney, Tea', '2025-05-06 06:03:52'),
(10, 'Wednesday', 'Lunch', 'White Rice, Mor Kulambu, Rasam, Poriyal, Kootu, Papad, Butter Milk, Pickle', '2025-05-06 06:03:52'),
(11, 'Wednesday', 'Snacks', 'Valaikai Baji, Tea', '2025-05-06 06:03:52'),
(12, 'Wednesday', 'Dinner', 'Idly, Tomato Thokku, Sambar, White Rice, Rasam, Pickle', '2025-05-06 06:03:52'),
(13, 'Thursday', 'Breakfast', 'Rava Kichadi, Bread, Jam, Chutney, Tea', '2025-05-06 06:03:52'),
(14, 'Thursday', 'Lunch', 'White Rice, Sambar, Rasam, Poriyal, Kootu, Papad, Butter Milk, Pickle', '2025-05-06 06:03:52'),
(15, 'Thursday', 'Snacks', 'Bonda, Tea', '2025-05-06 06:03:52'),
(16, 'Thursday', 'Dinner', 'Variety Rice, White Rice, Rasam, Egg, Gopi Manjurian, Brinjal Thalicha, Pickle', '2025-05-06 06:03:52'),
(17, 'Friday', 'Breakfast', 'Kal Dosai, Semiya, Sambar, Chutney, Tea', '2025-05-06 06:03:52'),
(18, 'Friday', 'Lunch', 'White Rice, Chicken Kulambu, Rasam, Poriyal, Kootu, Papad, Butter Milk, Pickle', '2025-05-06 06:03:52'),
(19, 'Friday', 'Snacks', 'Sundal, Tea', '2025-05-06 06:03:52'),
(20, 'Friday', 'Dinner', 'Chapathi, Tomato Thokku, White Rice, Rasam, Pickle', '2025-05-06 06:03:52'),
(21, 'Saturday', 'Breakfast', 'Idly, Medu Vadai, Sambar, Chutney, Tea', '2025-05-06 06:03:52'),
(22, 'Saturday', 'Lunch', 'White Rice, Sambar, Rasam, Poriyal, Kootu, Papad, Butter Milk, Pickle', '2025-05-06 06:03:52'),
(23, 'Saturday', 'Snacks', 'Sweet Bonda, Tea', '2025-05-06 06:03:52'),
(24, 'Saturday', 'Dinner', 'Dosa, Sambar, Chutney, White Rice, Rasam, Pickle', '2025-05-06 06:03:52'),
(25, 'Sunday', 'Breakfast', 'Poori, Alu Masala, Tea', '2025-05-06 06:03:52'),
(26, 'Sunday', 'Lunch', 'Chicken Briyani, Veg Briyani, White Rice, Rasam, Gobi Manjurian, Onion Raita, Pickle', '2025-05-06 06:03:52'),
(27, 'Sunday', 'Snacks', 'Groundnut, Tea', '2025-05-06 06:03:52'),
(28, 'Sunday', 'Dinner', 'Idly, Chutney, Sambar, White Rice, Rasam, Pickle', '2025-05-06 06:03:52');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `activity_log`
--
ALTER TABLE `activity_log`
  ADD PRIMARY KEY (`id`),
  ADD KEY `admin_id` (`admin_id`);

--
-- Indexes for table `admins`
--
ALTER TABLE `admins`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `ux_admins_email` (`email`);

--
-- Indexes for table `attendance`
--
ALTER TABLE `attendance`
  ADD PRIMARY KEY (`date`);

--
-- Indexes for table `food_history`
--
ALTER TABLE `food_history`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_day` (`day`);

--
-- Indexes for table `ingredients_inventory`
--
ALTER TABLE `ingredients_inventory`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `ingredient_name` (`ingredient_name`);

--
-- Indexes for table `menu`
--
ALTER TABLE `menu`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `activity_log`
--
ALTER TABLE `activity_log`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=16;

--
-- AUTO_INCREMENT for table `admins`
--
ALTER TABLE `admins`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `food_history`
--
ALTER TABLE `food_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `ingredients_inventory`
--
ALTER TABLE `ingredients_inventory`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `menu`
--
ALTER TABLE `menu`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=30;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `activity_log`
--
ALTER TABLE `activity_log`
  ADD CONSTRAINT `activity_log_ibfk_1` FOREIGN KEY (`admin_id`) REFERENCES `admins` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
