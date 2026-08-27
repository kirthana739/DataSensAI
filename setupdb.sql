-- Run this in MySQL Workbench or phpMyAdmin

-- Step 1: Create database
CREATE DATABASE IF NOT EXISTS datasense_ai;
USE datasense_ai;

-- Step 2: Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id   INT PRIMARY KEY AUTO_INCREMENT,
    username  VARCHAR(100) NOT NULL UNIQUE,
    password  VARCHAR(255) NOT NULL,       -- bcrypt hashed
    role      VARCHAR(50)  NOT NULL DEFAULT 'user'
);

-- Note: Do NOT insert passwords as plain text.
-- Run `python create_admin.py` after this to create the first admin user with proper bcrypt hashing.