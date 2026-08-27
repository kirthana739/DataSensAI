-- Run this against your existing DataSense AI database before using the
-- new Image & Video Analysis module. Mirrors the structure of the existing
-- `uploaded_files` table.

CREATE TABLE IF NOT EXISTS uploaded_media (
    media_id        INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT NOT NULL,
    file_name       VARCHAR(255) NOT NULL,
    file_path       VARCHAR(500) NOT NULL,
    media_type      ENUM('image', 'video') NOT NULL,
    file_size_kb    DECIMAL(10, 1),
    upload_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_uploaded_media_user ON uploaded_media(user_id);