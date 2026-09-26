-- VQA Predictions Analysis Queries
-- Default run_tag variable to 'kaggle_v3' if not provided

\if :{?run_tag}
    -- run_tag variable is already defined
\else
    \set run_tag kaggle_v3
\endif

-- ============================================================================
-- Query (a): Accuracy (%) and count by model_name and answer_type
-- ============================================================================
SELECT
    model_name,
    answer_type,
    COUNT(*) AS total_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_correct) / COUNT(*), 2) AS accuracy_pct
FROM vqa_predictions
WHERE run_tag = :'run_tag'
GROUP BY model_name, answer_type
ORDER BY model_name, answer_type;

-- ============================================================================
-- Query (b): Calibration: confidence buckets (0-0.2 ... 0.8-1.0) -> count, mean confidence, accuracy
-- ============================================================================
SELECT
    CASE
        WHEN confidence < 0.2 THEN '0.0 - 0.2'
        WHEN confidence < 0.4 THEN '0.2 - 0.4'
        WHEN confidence < 0.6 THEN '0.4 - 0.6'
        WHEN confidence < 0.8 THEN '0.6 - 0.8'
        ELSE '0.8 - 1.0'
    END AS confidence_bucket,
    COUNT(*) AS total_count,
    ROUND(AVG(confidence)::numeric, 4) AS mean_confidence,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_correct) / COUNT(*), 2) AS accuracy_pct
FROM vqa_predictions
WHERE run_tag = :'run_tag'
GROUP BY
    CASE
        WHEN confidence < 0.2 THEN '0.0 - 0.2'
        WHEN confidence < 0.4 THEN '0.2 - 0.4'
        WHEN confidence < 0.6 THEN '0.4 - 0.6'
        WHEN confidence < 0.8 THEN '0.6 - 0.8'
        ELSE '0.8 - 1.0'
    END
ORDER BY MIN(confidence);

-- ============================================================================
-- Query (c): Top-20 predicted answers: count, accuracy
-- ============================================================================
SELECT
    predicted_answer,
    COUNT(*) AS total_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_correct) / COUNT(*), 2) AS accuracy_pct
FROM vqa_predictions
WHERE run_tag = :'run_tag'
GROUP BY predicted_answer
ORDER BY total_count DESC, predicted_answer ASC
LIMIT 20;

-- ============================================================================
-- Query (d): Top-20 confident errors (is_correct = false, confidence DESC)
-- ============================================================================
SELECT
    question_id,
    image_id,
    question,
    predicted_answer,
    confidence,
    answer_type
FROM vqa_predictions
WHERE run_tag = :'run_tag'
  AND is_correct = FALSE
ORDER BY confidence DESC, question_id ASC
LIMIT 20;

-- ============================================================================
-- Query (e): Accuracy by first word of question (lower(split_part(question, ' ', 1))), top-15 by count
-- ============================================================================
SELECT
    LOWER(SPLIT_PART(question, ' ', 1)) AS first_word,
    COUNT(*) AS total_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_correct) / COUNT(*), 2) AS accuracy_pct
FROM vqa_predictions
WHERE run_tag = :'run_tag'
GROUP BY LOWER(SPLIT_PART(question, ' ', 1))
ORDER BY total_count DESC, first_word ASC
LIMIT 15;
