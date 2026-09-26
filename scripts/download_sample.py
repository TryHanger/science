import json
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw

def create_sample_dataset(data_dir: str = "./data", num_train: int = 300, num_val: int = 100):
    data_path = Path(data_dir)
    train_img_dir = data_path / "train2014"
    val_img_dir = data_path / "val2014"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Generating images in {train_img_dir} and {val_img_dir}...")
    colors = [
        ("red", (255, 50, 50)),
        ("blue", (50, 50, 255)),
        ("green", (50, 255, 50)),
        ("yellow", (255, 255, 50)),
        ("white", (250, 250, 250)),
        ("black", (20, 20, 20)),
    ]

    # Generate synthetic images for train and val splits
    def generate_images(img_dir: Path, prefix: str, count: int):
        img_info = {}
        for idx in range(1, count + 1):
            img_id = idx
            filename = f"COCO_{prefix}_{img_id:012d}.jpg"
            img_file = img_dir / filename
            
            # Draw 256x256 image with geometric shapes
            img = Image.new("RGB", (256, 256), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            color_name, rgb = colors[idx % len(colors)]
            shape = "circle" if idx % 2 == 0 else "rectangle"
            num_objects = (idx % 3) + 1
            
            for o in range(num_objects):
                offset = o * 40
                if shape == "circle":
                    draw.ellipse([40 + offset, 40, 100 + offset, 100], fill=rgb, outline=(0, 0, 0))
                else:
                    draw.rectangle([40 + offset, 40, 100 + offset, 100], fill=rgb, outline=(0, 0, 0))
            
            img.save(img_file, format="JPEG")
            img_info[img_id] = {
                "color": color_name,
                "shape": shape,
                "count": str(num_objects)
            }
        return img_info

    train_meta = generate_images(train_img_dir, "train2014", 30)
    val_meta = generate_images(val_img_dir, "val2014", 15)

    print("[2/4] Generating questions and annotations following VQA v2 protocol...")
    
    def generate_vqa_pair(meta_dict, total_questions, split_name):
        questions_list = []
        annotations_list = []
        
        img_ids = list(meta_dict.keys())
        
        for q_idx in range(1, total_questions + 1):
            img_id = random.choice(img_ids)
            info = meta_dict[img_id]
            q_type_choice = random.choice(["color", "shape", "count", "yesno"])
            
            if q_type_choice == "color":
                q_text = "what color is the object?"
                true_ans = info["color"]
                ans_type = "other"
            elif q_type_choice == "shape":
                q_text = "is this a circle or a rectangle?"
                true_ans = info["shape"]
                ans_type = "other"
            elif q_type_choice == "count":
                q_text = "how many objects are there?"
                true_ans = info["count"]
                ans_type = "number"
            else:
                q_text = f"is the object {info['color']}?"
                true_ans = "yes" if random.random() > 0.3 else "no"
                ans_type = "yes/no"
            
            # In VQA v2 each question contains 10 answers from different annotators
            # Most answers match the ground truth, with occasional variance
            answers_10 = []
            for a_i in range(10):
                # 80-90% ground-truth answer
                if random.random() < 0.85:
                    ans_val = true_ans
                else:
                    # random noise
                    ans_val = random.choice(["yes", "no", "red", "blue", "1", "2"])
                answers_10.append({
                    "answer": ans_val,
                    "answer_confidence": "yes",
                    "answer_id": a_i + 1
                })
            
            q_id = 100000 + q_idx if split_name == "train" else 200000 + q_idx
            
            questions_list.append({
                "image_id": img_id,
                "question": q_text,
                "question_id": q_id
            })
            
            annotations_list.append({
                "question_type": q_type_choice,
                "multiple_choice_answer": true_ans,
                "answers": answers_10,
                "image_id": img_id,
                "answer_type": ans_type,
                "question_id": q_id
            })
            
        return (
            {"questions": questions_list, "data_type": "mscoco", "data_subtype": f"{split_name}2014"},
            {"annotations": annotations_list, "data_type": "mscoco", "data_subtype": f"{split_name}2014"}
        )

    random.seed(42)
    train_q, train_a = generate_vqa_pair(train_meta, num_train, "train")
    val_q, val_a = generate_vqa_pair(val_meta, num_val, "val")

    print("[3/4] Saving JSON files...")
    with open(data_path / "v2_OpenEnded_mscoco_train2014_questions.json", "w", encoding="utf-8") as f:
        json.dump(train_q, f, ensure_ascii=False, indent=2)
    with open(data_path / "v2_mscoco_train2014_annotations.json", "w", encoding="utf-8") as f:
        json.dump(train_a, f, ensure_ascii=False, indent=2)
        
    with open(data_path / "v2_OpenEnded_mscoco_val2014_questions.json", "w", encoding="utf-8") as f:
        json.dump(val_q, f, ensure_ascii=False, indent=2)
    with open(data_path / "v2_mscoco_val2014_annotations.json", "w", encoding="utf-8") as f:
        json.dump(val_a, f, ensure_ascii=False, indent=2)

    print(f"[4/4] Done! Generated:")
    print(f"  - Train: {len(train_q['questions'])} questions, {len(list(train_img_dir.glob('*.jpg')))} images")
    print(f"  - Val:   {len(val_q['questions'])} questions, {len(list(val_img_dir.glob('*.jpg')))} images")

if __name__ == "__main__":
    create_sample_dataset()
