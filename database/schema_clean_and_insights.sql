-- Run this AFTER setupdb.sql and database/schema.sql (needs `users` and
-- `uploaded_files` to already exist, for the foreign keys below).
--
-- These two tables were missing from the original SQL files: app.py reads
-- and writes them (Module 4 - Clean Data, Module 9 - AI Insights), but
-- there was no CREATE TABLE anywhere for either one. On a fresh database
-- the app will throw "table doesn't exist" the moment you open Clean Data
-- or Insights for a file.
--
-- USE datasense_ai;   (uncomment if running this file standalone)

CREATE TABLE IF NOT EXISTS clean_data (
    clean_id                INT PRIMARY KEY AUTO_INCREMENT,
    file_id                 INT NOT NULL,
    user_id                 INT NOT NULL,
    clean_file_path         VARCHAR(500) NOT NULL,
    original_record_count   INT,
    clean_record_count      INT,
    duplicates_removed      INT,
    nulls_fixed             INT,
    unknowns_fixed          INT,
    outliers_capped         INT,
    cleaned_date             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES uploaded_files(file_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_clean_data_file_user ON clean_data(file_id, user_id);


CREATE TABLE IF NOT EXISTS ai_insights (
    insight_id      INT PRIMARY KEY AUTO_INCREMENT,
    file_id         INT NOT NULL,
    user_id         INT NOT NULL,
    recommendation  TEXT,
    created_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES uploaded_files(file_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_ai_insights_file_user ON ai_insights(file_id, user_id);
