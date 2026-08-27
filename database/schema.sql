CREATE TABLE IF NOT EXISTS uploaded_files (
    file_id       INT PRIMARY KEY AUTO_INCREMENT,
    user_id       INT NOT NULL,
    file_name     VARCHAR(255) NOT NULL,
    file_path     VARCHAR(500) NOT NULL,
    upload_date   DATETIME DEFAULT CURRENT_TIMESTAMP,
    record_count  INT,
    column_count  INT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);