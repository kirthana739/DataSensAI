-- Run this in your mysql> client, same as before:
--   USE datasense_ai;
--   (then paste this)

ALTER TABLE uploaded_media
    ADD COLUMN thumbnail_path VARCHAR(500) NULL AFTER file_path;