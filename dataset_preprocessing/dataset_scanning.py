
import os
import pandas as pd

DATASET_ROOT = "/home/ahad/Documents/fyp_dataset"
OUTPUT_WORKSPACE = "/home/ahad/Documents/fyp_pipeline_outputs"
ML_DATASET_DIR = os.path.join(OUTPUT_WORKSPACE, "fyp_processed_dataset")
LANDMARKS_SAVE_DIR = os.path.join(ML_DATASET_DIR, "normalized_landmarks")

os.makedirs(LANDMARKS_SAVE_DIR, exist_ok=True)

print("📁 Local pipeline output workspace initialized.")
print(f"📦 Dedicated ML Dataset Folder established at:\n   └─> {ML_DATASET_DIR}\n")

print("🔍 Scanning your local 20-sentence dataset structure (cleaning hidden files)...")

manifest_records = []
ignored_eaf_count = 0
ghost_files_ignored = 0

for folder_name in sorted(os.listdir(DATASET_ROOT)):
    folder_path = os.path.join(DATASET_ROOT, folder_name)

    if not os.path.isdir(folder_path) or folder_name == "env_urdu_sl":
        continue

    if folder_name.startswith("S") and "_" in folder_name:
        sentence_id = folder_name.split("_")[0]
        sentence_text = folder_name[len(sentence_id)+1:]

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.startswith("._"):
                ghost_files_ignored += 1
                continue

            if file_name.endswith(".avi"):
                base_name = os.path.splitext(file_name)[0]
                npy_filename = f"{sentence_id}_{base_name}.npy"
                landmark_output_path = os.path.join(LANDMARKS_SAVE_DIR, npy_filename)

                manifest_records.append({
                    "sentence_id": sentence_id,
                    "sentence_text": sentence_text,
                    "raw_video_name": file_name,
                    "raw_video_path": file_path,
                    "landmark_output_path": landmark_output_path
                })
            elif file_name.endswith(".eaf"):
                ignored_eaf_count += 1

df_manifest = pd.DataFrame(manifest_records)
df_manifest = df_manifest.sort_values(by=["sentence_id"]).reset_index(drop=True)

print("📊 Scan Complete!")
print(f"🎥 Real Videos (.avi) Detected: {len(df_manifest)}")
print(f"🗣️ Total Sentence Categories Found: {df_manifest['sentence_id'].nunique()}")
print(f"📝 Total ELAN Annotation Files (.eaf) safely ignored: {ignored_eaf_count}")
print(f"👻 Apple hidden ghost files (`._*`) filtered out: {ghost_files_ignored}\n")

if len(df_manifest) > 0:
    print("👀 Verified Data Manifest Preview:")
    print(df_manifest[['sentence_id', 'sentence_text', 'raw_video_name']].head(5).to_string())

    backup_path = os.path.join(ML_DATASET_DIR, "initial_scan_manifest.csv")
    df_manifest.to_csv(backup_path, index=False)
    print(f"\n💾 Saved clean manifest configuration to:\n   └─> {backup_path}")
