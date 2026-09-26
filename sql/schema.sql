-- Schema definition for VQA predictions
-- Table: vqa_predictions

CREATE TABLE IF NOT EXISTS vqa_predictions (
    run_tag VARCHAR(50) NOT NULL DEFAULT 'local',
    model_name VARCHAR(50) NOT NULL DEFAULT 'vqa_mul',
    question_id BIGINT NOT NULL,
    image_id BIGINT NOT NULL,
    question TEXT NOT NULL,
    predicted_answer VARCHAR(100) NOT NULL,
    confidence REAL NOT NULL,
    is_correct BOOLEAN NOT NULL,
    answer_type VARCHAR(20) NOT NULL,
    PRIMARY KEY (run_tag, model_name, question_id)
);

CREATE INDEX IF NOT EXISTS idx_vqa_predictions_run_tag_answer_type
ON vqa_predictions (run_tag, answer_type);
